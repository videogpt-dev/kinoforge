from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from kinoforge.contract import ModelRef
from kinoforge.observ import active, logged, with_context
from kinoforge.segments.clips.moments.filter import MomentFilter
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.moments.transcript import TranscriptText

Completer = Callable[..., str]
Renderer = Callable[..., str]

_CHARS_PER_TOKEN = 4
_RESERVE_TOKENS = 8000
_MIN_BUDGET_CHARS = 4000
_CHUNK_OVERLAP = 3


@dataclass(frozen=True)
class MomentTuning:
    """Engine-owned AI moment knobs; a caller may override any, unset/zero keeps the default."""

    max_workers: int = 4
    batch_size: int = 8
    max_candidates: int = 40
    context_pad: float = 2.0

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "MomentTuning":
        value = value or {}
        defaults = cls()
        return cls(**{
            f.name: type(getattr(defaults, f.name))(value[f.name])
            for f in fields(cls)
            if value.get(f.name) not in (None, 0, "")
        })


class AiMomentEngine:
    """Finds moments by reading the transcript with an LLM, and scores extracted candidates
    in batches when discovery comes back empty."""

    def __init__(
        self,
        route: ModelRef,
        *,
        min_clip_length: float,
        max_clip_length: float,
        tuning: Optional[Mapping[str, Any]],
        complete: Completer,
        render: Renderer,
        context_window: int = 1_000_000,
    ):
        self.name = str(route)
        self._route = route
        self._min_len = float(min_clip_length)
        self._max_len = float(max_clip_length)
        self._tuning = MomentTuning.from_mapping(tuning)
        self._complete = complete
        self._render = render
        usable = max(int(context_window) - _RESERVE_TOKENS, 0)
        self._budget_chars = max(usable * _CHARS_PER_TOKEN, _MIN_BUDGET_CHARS)

    @logged
    def discover_moments(
        self, transcript: List[Dict], min_len: float, max_len: float, target_clips: int
    ) -> List[Dict]:
        """Scored moments (ai_scored=True), snapped to segment edges, sized into the window."""
        segs = self._fit_budget([s for s in transcript or [] if s.get("text")])
        if not segs:
            return []
        active().info(f"  Reading the full transcript to find moments ({self._route})...")
        chunks = self._chunks(segs)
        want = max(target_clips, 5)
        picks = [p for found in self._parallel(
            lambda chunk: self._discover_in(chunk, min_len, max_len, want), chunks
        ) if found for p in found]
        moments = [m for m in (self._to_moment(p, segs, min_len, max_len) for p in picks) if m]
        kept: List[Moment] = []
        for moment in sorted((Moment(m) for m in moments), key=lambda m: m.score, reverse=True):
            if all(moment.overlap_ratio(k) <= 0.5 for k in kept):
                kept.append(moment)
        active().success(f"  Chose {len(kept)} moments from the full transcript "
                         f"({len(chunks)} pass(es), {len(picks)} raw picks)")
        return [m.data for m in kept]

    @logged
    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Pre-filter, then one batched analysis that scores and re-bounds each candidate."""
        if not candidates:
            return []
        active().info(f"  Analyzing moments ({self._route})...")
        top = MomentFilter.run(candidates, transcript)[: self._tuning.max_candidates]
        if not top:
            return []
        kept: List[Dict] = []
        adjusted = 0
        for moment, analysis in zip(top, self._analyze(top, transcript)):
            adjusted += self._apply(moment, analysis)
            if analysis["worthy"]:
                kept.append(moment)
        active().info(f"  Analyzed {len(top)} moments, kept {len(kept)}, "
                      f"adjusted {adjusted} boundaries")
        return kept or top[:10]

    @logged
    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Ranked by score; only moments filter_moments did not already score cost a call."""
        unscored = [m for m in moments if not m.get("ai_scored")]
        for moment, analysis in zip(unscored, self._analyze(unscored, transcript)):
            self._apply(moment, analysis)
        ranked = sorted(moments, key=lambda m: m.get("score", 0), reverse=True)
        if ranked:
            active().success(f"  Ranked {len(ranked)} moments (top: "
                             f"{ranked[0].get('score', 0):.0f}, low: "
                             f"{ranked[-1].get('score', 0):.0f})")
        return ranked

    def _fit_budget(self, segs: List[Dict]) -> List[Dict]:
        kept: List[Dict] = []
        size = 0
        for seg in segs:
            size += len(str(seg.get("text") or "")) + 12
            if size > self._budget_chars:
                break
            kept.append(seg)
        if len(kept) < len(segs):
            active().warning(
                f"  Transcript over the {self._budget_chars // _CHARS_PER_TOKEN}-token budget: "
                f"kept the first {len(kept)}/{len(segs)} segments, tail truncated."
            )
        return kept

    def _chunks(self, segs: List[Dict]) -> List[List[Dict]]:
        chunks: List[List[Dict]] = []
        current: List[Dict] = []
        size = 0
        for seg in segs:
            current.append(seg)
            size += len(str(seg.get("text") or "")) + 12
            if size >= self._budget_chars:
                chunks.append(current)
                current = current[-_CHUNK_OVERLAP:]
                size = sum(len(str(s.get("text") or "")) + 12 for s in current)
        if current and (not chunks or len(current) > _CHUNK_OVERLAP):
            chunks.append(current)
        return chunks or [segs]

    def _discover_in(self, chunk: List[Dict], min_len: float, max_len: float,
                     want: int) -> List[Dict]:
        lines = "\n".join(
            f"[{Moment(s).start:.1f}] {str(s.get('text') or '').strip()[:200]}" for s in chunk
        )
        prompt = self._render(
            "prompts.agents.moment_discovery", count=str(want), min_len=f"{min_len:.0f}",
            max_len=f"{max_len:.0f}", transcript=lines,
        )
        if not prompt.strip():
            return []
        reply = self._ask(prompt, 120 * want + 300, 0.4)
        return [p for p in reply if isinstance(p, dict)]

    def _to_moment(self, pick: Dict, segs: List[Dict], min_len: float,
                   max_len: float) -> Optional[Dict]:
        try:
            start, end = float(pick["start"]), float(pick["end"])
        except (KeyError, ValueError, TypeError):
            return None
        if end <= start:
            return None
        s, e, text = TranscriptText.snap(segs, start, end, min_len, max_len)
        if e - s < 2.0:
            return None
        return Moment.new(
            s, e, text=text,
            score=min(max(float(pick.get("score") or 60), 0), 100),
            ai_reason=str(pick.get("reason", ""))[:200],
            ai_hook=str(pick.get("hook", ""))[:120],
            title=str(pick.get("title", ""))[:120],
            language=TranscriptText.detect_language(text),
            ai_scored=True, provider=self._route.provider, source="transcript_discovery",
        )

    def _analyze(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        size = self._tuning.batch_size
        batches = [moments[i:i + size] for i in range(0, len(moments), size)]
        results = self._parallel(lambda batch: self._analyze_batch(batch, transcript or []),
                                 batches)
        return [a for batch, result in zip(batches, results)
                for a in (result if result and len(result) == len(batch)
                          else [self.analysis({}, m, (0.0, 0.0)) for m in batch])]

    def _analyze_batch(self, batch: List[Dict], transcript: List[Dict]) -> List[Dict]:
        blocks = [self._clip_block(i, m, transcript) for i, m in enumerate(batch)]
        prompt = self._render(
            "prompts.agents.clip_analysis", min_len=f"{self._min_len:.0f}",
            max_len=f"{self._max_len:.0f}", clips="\n\n".join(b for _, b in blocks),
        )
        reply = self._ask(prompt, 160 * len(batch) + 200, 0.3)
        by_id = {}
        for obj in reply:
            try:
                by_id[int(obj["id"])] = obj
            except (KeyError, ValueError, TypeError):
                continue
        return [
            self.analysis(by_id.get(i + 1) or (reply[i] if i < len(reply) else {}), m,
                          blocks[i][0])
            for i, m in enumerate(batch)
        ]

    def _clip_block(self, index: int, moment: Dict,
                    transcript: List[Dict]) -> Tuple[Tuple[float, float], str]:
        start = float(moment.get("start", 0.0))
        end = float(moment.get("end", start + self._max_len))
        pad = self._tuning.context_pad
        win_start, win_end = max(0.0, start - pad), end + pad
        context = [s for s in transcript
                   if s.get("end", 0) > win_start and s.get("start", 0) < win_end]
        if context:
            win_end = max(win_end, context[-1].get("end", win_end))
            lines = "\n".join(f"[{s.get('start', 0):.1f}-{s.get('end', 0):.1f}] "
                              f"{s.get('text', '')[:160]}" for s in context)
        else:
            lines = f"[{start:.1f}-{end:.1f}] {moment.get('text', '')[:280]}"
        return (win_start, win_end), (
            f"CLIP {index + 1}: rough candidate {start:.1f}-{end:.1f}s\n"
            f"Transcript segments (use these exact timestamps for boundaries):\n{lines}"
        )

    @staticmethod
    def analysis(obj: Any, moment: Dict, window: Tuple[float, float]) -> Dict:
        """The model's verdict on one moment in standard shape; boundaries are clamped to the
        context window and kept as-is when missing, invalid or under 2s."""
        obj = obj if isinstance(obj, dict) else {}
        try:
            score = float(obj.get("score", 60))
        except (ValueError, TypeError):
            score = 60.0
        worthy = obj.get("worthy", True)
        if isinstance(worthy, str):
            worthy = worthy.strip().lower() in ("true", "yes", "1")
        start, end = moment.get("start"), moment.get("end")
        try:
            new_start, new_end = (max(window[0], min(float(obj[k]), window[1]))
                                  for k in ("start", "end"))
            if new_end - new_start >= 2.0:
                start, end = new_start, new_end
        except (KeyError, ValueError, TypeError):
            pass
        return {
            "worthy": bool(worthy), "score": min(max(score, 0), 100),
            "reason": str(obj.get("reason", "fallback" if not obj else ""))[:200],
            "hook": str(obj.get("hook", ""))[:120], "start": start, "end": end,
        }

    def _apply(self, moment: Dict, analysis: Dict) -> bool:
        moment.update(score=analysis["score"], ai_reason=analysis["reason"],
                      ai_hook=analysis["hook"], ai_scored=True, provider=self._route.provider)
        start, end = analysis["start"], analysis["end"]
        if start is None or end is None or end <= start:
            return False
        view = Moment(moment)
        moved = abs(start - view.start) > 0.05 or abs(end - view.end) > 0.05
        view.set_span(start, end)
        return moved

    def _ask(self, prompt: str, max_tokens: int, temperature: float) -> List[Any]:
        """The JSON array in the model's reply (code fences and extra text tolerated), or []."""
        content = self._complete(prompt, max_tokens=max_tokens, temperature=temperature) or ""
        for chunk in re.split(r"```(?:json)?", content) if "```" in content else []:
            if chunk.strip().startswith("["):
                content = chunk.strip()
                break
        start, end = content.find("["), content.rfind("]")
        try:
            data = json.loads(content[start:end + 1] if 0 <= start < end else content)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _parallel(self, fn: Callable[[Any], Any], items: List[Any]) -> List[Any]:
        """fn over items concurrently, in order; a failed item yields None."""
        tasks = [(with_context(self._attempt), item) for item in items]
        with ThreadPoolExecutor(max_workers=self._tuning.max_workers) as pool:
            return list(pool.map(lambda task: task[0](fn, task[1]), tasks))

    @staticmethod
    def _attempt(fn: Callable[[Any], Any], item: Any) -> Any:
        try:
            return fn(item)
        except Exception as exc:
            active().warning(f"  AI request failed: {exc}")
            return None
