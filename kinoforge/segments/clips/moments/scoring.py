from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

from kinoforge.contract import ModelRef
from kinoforge.segments.clips.moments.analysis import MomentAnalysis
from kinoforge.segments.clips.moments.llm_client import LlmClient
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.observ import active, logged


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


class MomentScorer:
    """Candidate filter + score: pre-filter, then batched LLM analysis that judges, scores, and
    picks natural clip boundaries within each candidate's context window."""

    def __init__(
        self,
        llm: LlmClient,
        route: ModelRef,
        min_clip_length: float,
        max_clip_length: float,
        tuning: MomentTuning,
    ) -> None:
        self._llm = llm
        self._route = route
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
        self._batch_size = tuning.batch_size
        self._max_candidates = tuning.max_candidates
        self._context_pad = tuning.context_pad

    @logged
    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Filter + score candidate moments in one combined batched pass, attaching
        score/reason/hook/worthy so a later score_moments() makes zero extra calls."""
        if not candidates:
            return []

        active().info(f"  Analyzing moments ({self._route})...")
        from kinoforge.segments.clips.moments.filter import MomentFilter

        pre_filtered = MomentFilter.run(candidates, transcript)
        if not pre_filtered:
            return []
        top = pre_filtered[: self._max_candidates]
        analyses = self._analyze_moments_batched(top, transcript)

        kept: List[Dict] = []
        adjusted = 0
        for moment, a in zip(top, analyses):
            adjusted += self._apply(moment, a)
            if a.get("worthy", True):
                kept.append(moment)

        n_batches = (len(top) + self._batch_size - 1) // self._batch_size
        active().info(f"  Analyzed {len(top)} moments in {n_batches} request(s), "
              f"kept {len(kept)}, adjusted {adjusted} boundaries")
        return kept if kept else top[:10]

    @logged
    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Rank moments. Reuses scores from filter_moments; scores any stragglers."""
        unscored = [m for m in moments if not m.get("ai_scored")]
        if unscored:
            active().info(f"  Scoring {len(unscored)} moments ({self._route})...")
            analyses = self._analyze_moments_batched(unscored, transcript)
            for moment, a in zip(unscored, analyses):
                self._apply(moment, a)
        else:
            active().info(
                f"  Ranking {len(moments)} moments (scores from analysis pass, no extra calls)"
            )

        ranked = sorted(moments, key=lambda m: m.get("score", 0), reverse=True)
        if ranked:
            active().success(f"  Ranked {len(ranked)} moments "
                  f"(top: {ranked[0].get('score', 0):.0f}, low: {ranked[-1].get('score', 0):.0f})")
        return ranked

    def _apply(self, moment: Dict, analysis: Dict) -> bool:
        """Write one analysis onto its moment; True when it moved the clip boundaries."""
        moment.update(
            score=analysis.get("score", 60.0), ai_reason=analysis.get("reason", ""),
            ai_hook=analysis.get("hook", ""), ai_scored=True, provider=self._route.provider,
        )
        start, end = analysis.get("start"), analysis.get("end")
        if start is None or end is None or end <= start:
            return False
        view = Moment(moment)
        moved = abs(start - view.start) > 0.05 or abs(end - view.end) > 0.05
        view.set_span(start, end)
        return moved

    def _analyze_moments_batched(
        self, moments: List[Dict], transcript: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Analyze moments in batches (one request per batch), parallel across batches.
        Returns a list aligned to `moments`, each {worthy, score, reason, hook, start, end}."""
        transcript = transcript or []
        size = self._batch_size
        batches = [moments[i : i + size] for i in range(0, len(moments), size)]
        batch_results = self._llm.parallel_map(
            lambda b: self._analyze_batch(b, transcript), batches, desc="  Analyzing"
        )
        out: List[Dict] = []
        for batch, result in zip(batches, batch_results):
            if result and len(result) == len(batch):
                out.extend(result)
            else:
                out.extend(MomentAnalysis.fallback(batch))
        return out

    def _analyze_batch(self, batch: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Single request: judge + score a batch and pick natural clip boundaries, each clip
        shown with its surrounding timestamped transcript so cuts land on segment edges."""
        pieces = [self._clip_block(i, moment, transcript) for i, moment in enumerate(batch)]
        prompt = self._llm.render(
            "prompts.agents.clip_analysis", min_len=f"{self.min_clip_length:.0f}",
            max_len=f"{self.max_clip_length:.0f}", clips="\n\n".join(b for _, b in pieces),
        )
        content = self._llm.complete(prompt, max_tokens=160 * len(batch) + 200, temperature=0.3)
        parsed = self._llm.parse_json_array(content) if content else None
        if not parsed:
            return MomentAnalysis.fallback(batch)
        by_id = self._by_id(parsed)
        return [
            MomentAnalysis.normalize(
                by_id.get(i + 1) or (parsed[i] if i < len(parsed) else {}), moment, pieces[i][0]
            )
            for i, moment in enumerate(batch)
        ]

    def _clip_block(
        self, index: int, moment: Dict, transcript: List[Dict]
    ) -> Tuple[Tuple[float, float], str]:
        """(context window, prompt block) for one candidate: its padded span of transcript."""
        start = float(moment.get("start", 0.0))
        end = float(moment.get("end", start + self.max_clip_length))
        win_start, win_end = max(0.0, start - self._context_pad), end + self._context_pad
        context = [s for s in transcript
                   if s.get("end", 0) > win_start and s.get("start", 0) < win_end]
        if context:
            win_end = max(win_end, context[-1].get("end", win_end))
            lines = "\n".join(
                f"[{s.get('start', 0):.1f}-{s.get('end', 0):.1f}] {s.get('text', '')[:160]}"
                for s in context
            )
        else:
            lines = f"[{start:.1f}-{end:.1f}] {moment.get('text', '')[:280]}"
        block = (f"CLIP {index + 1}: rough candidate {start:.1f}-{end:.1f}s\n"
                 f"Transcript segments (use these exact timestamps for boundaries):\n{lines}")
        return (win_start, win_end), block

    @staticmethod
    def _by_id(parsed: List) -> Dict[int, Dict]:
        """Model answers keyed by their 1-based clip id (answers without a usable id skipped)."""
        by_id: Dict[int, Dict] = {}
        for obj in parsed:
            if isinstance(obj, dict) and "id" in obj:
                try:
                    by_id[int(obj["id"])] = obj
                except (ValueError, TypeError):
                    continue
        return by_id
