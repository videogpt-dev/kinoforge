from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from kinoforge.contract import ModelRef
from kinoforge.observ import active, logged, with_context
from kinoforge.segments.clips.moments.finders.base import MomentFinder, MomentSpec
from kinoforge.segments.clips.moments.finders.offline import OfflineMomentFinder
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.moments.transcript import TranscriptText

Completer = Callable[..., str]
Renderer = Callable[..., str]

_CHARS_PER_TOKEN = 4
_RESERVE_TOKENS = 8000
_MIN_BUDGET_CHARS = 4000
_REPLY_TOKENS = 4000


class AiMomentFinder(MomentFinder):
    """Gives the LLM the whole timestamped transcript and asks for up to `count` moments.
    A transcript over the context budget is split by time into labelled parts that overlap
    by max_len seconds, so nothing is dropped. When the model picks nothing, the offline
    finder ranks the transcript instead."""

    def __init__(
        self,
        route: ModelRef,
        *,
        complete: Completer,
        render: Renderer,
        context_window: int = 1_000_000,
        max_workers: int = 4,
    ):
        self.name = str(route)
        self._route = route
        self._complete = complete
        self._render = render
        self._max_workers = max(int(max_workers), 1)
        usable = max(int(context_window) - _RESERVE_TOKENS, 0)
        self._budget_chars = max(usable * _CHARS_PER_TOKEN, _MIN_BUDGET_CHARS)

    @logged
    def find(self, transcript: List[Dict], spec: MomentSpec) -> List[Dict]:
        segs = [s for s in transcript or [] if str(s.get("text") or "").strip()]
        moments = self._discover(segs, spec) if segs else []
        if moments:
            return moments
        active().warning("  The model picked no moments; ranking the transcript offline")
        return OfflineMomentFinder().find(transcript, spec)

    def _discover(self, segs: List[Dict], spec: MomentSpec) -> List[Dict]:
        parts = self._parts(segs, spec.max_len)
        active().info(f"  Reading the full transcript in {len(parts)} part(s) ({self._route})...")
        labels = [self._label(i, len(parts), part, segs) for i, part in enumerate(parts)]
        picks = [p for found in self._parallel(
            lambda job: self._ask_part(job[0], job[1], spec), list(zip(parts, labels))
        ) if found for p in found]
        moments = [m for m in (self._to_moment(p, segs, spec) for p in picks) if m]
        kept: List[Moment] = []
        for moment in sorted((Moment(m) for m in moments), key=lambda m: m.score, reverse=True):
            if all(moment.overlap_ratio(k) <= 0.5 for k in kept):
                kept.append(moment)
        active().success(f"  Chose {len(kept)} moments ({len(picks)} raw picks)")
        return [m.data for m in kept[: spec.count]]

    @staticmethod
    def _line(seg: Dict) -> str:
        return f"[{Moment(seg).start:.1f}] {str(seg.get('text') or '').strip()}"

    def _parts(self, segs: List[Dict], overlap: float) -> List[List[Dict]]:
        """Segments cut at whole-segment boundaries whenever a part fills the budget; each
        part after the first repeats the previous part's last `overlap` seconds."""
        parts: List[List[Dict]] = []
        current: List[Dict] = []
        size = 0
        for seg in segs:
            line = len(self._line(seg)) + 1
            if current and size + line > self._budget_chars:
                parts.append(current)
                current = self._tail(current, overlap)
                size = sum(len(self._line(s)) + 1 for s in current)
            current.append(seg)
            size += line
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _tail(part: List[Dict], seconds: float) -> List[Dict]:
        """The part's last `seconds`, at most half its segments so each part moves forward."""
        cut = Moment(part[-1]).end - seconds
        tail = [s for s in part if Moment(s).start >= cut]
        return tail[len(tail) - len(part) // 2:] if len(tail) > len(part) // 2 else tail

    @staticmethod
    def _label(index: int, total: int, part: List[Dict], segs: List[Dict]) -> str:
        if total == 1:
            return "This is the whole transcript."
        return (f"This is part {index + 1} of {total} of the transcript, covering "
                f"{TranscriptText.clock(Moment(part[0]).start)}-"
                f"{TranscriptText.clock(Moment(part[-1]).end)} of a "
                f"{TranscriptText.clock(Moment(segs[-1]).end)} video. "
                f"Only pick moments inside this part.")

    def _ask_part(self, part: List[Dict], label: str, spec: MomentSpec) -> List[Dict]:
        prompt = self._render(
            "prompts.agents.moment_discovery",
            transcript="\n".join(self._line(s) for s in part), part=label,
            count=str(spec.count), min_len=f"{spec.min_len:.0f}", max_len=f"{spec.max_len:.0f}",
        )
        return self._ask(prompt, _REPLY_TOKENS + 150 * spec.count, 0.4)

    def _to_moment(self, pick: Dict, segs: List[Dict], spec: MomentSpec) -> Optional[Dict]:
        try:
            start, end = float(pick["start"]), float(pick["end"])
        except (KeyError, ValueError, TypeError):
            return None
        if end <= start:
            return None
        s, e, text = TranscriptText.snap(segs, start, end, spec.min_len, spec.max_len)
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

    def _ask(self, prompt: str, max_tokens: int, temperature: float) -> List[Dict]:
        """The last JSON array of objects in the reply, so fences, prose and a reasoning
        model's thinking before it are tolerated; [] when there is none."""
        content = self._complete(prompt, max_tokens=max_tokens, temperature=temperature) or ""
        decoder = json.JSONDecoder()
        found: List[Dict] = []
        pos = content.find("[")
        while pos >= 0:
            try:
                data, end = decoder.raw_decode(content, pos)
            except json.JSONDecodeError:
                pos = content.find("[", pos + 1)
                continue
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                found = data
            pos = content.find("[", end)
        if not found:
            active().warning(f"  No JSON array in the model reply ({len(content)} chars): "
                             f"{content[-300:]!r}")
        return found

    def _parallel(self, fn: Callable[[Any], Any], items: List[Any]) -> List[Any]:
        """fn over items concurrently, in order; a failed item yields None."""
        tasks = [(with_context(self._attempt), item) for item in items]
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(lambda task: task[0](fn, task[1]), tasks))

    @staticmethod
    def _attempt(fn: Callable[[Any], Any], item: Any) -> Any:
        try:
            return fn(item)
        except Exception as exc:
            active().warning(f"  AI request failed: {exc}")
            return None
