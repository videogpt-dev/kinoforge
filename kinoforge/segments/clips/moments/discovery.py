from __future__ import annotations

from typing import Dict, List, Optional

from kinoforge.contract import ModelRef
from kinoforge.segments.clips.moments.extractor import TranscriptText
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.moments.llm_client import LlmClient
from kinoforge.observ import active, logged

_CHARS_PER_TOKEN = 4  # rough average for transcript prose, not code
_RESERVE_TOKENS = 8000  # held back for the prompt template and the model's own answer
_MIN_BUDGET_CHARS = 4000


class MomentDiscoverer:
    """Transcript-first discovery: read the transcript (chunked when long) and pick
    self-contained moments snapped to segment edges. Capped to the context window
    (context_window tokens, default 1M); an over-long transcript is tail-truncated."""

    def __init__(self, llm: LlmClient, route: ModelRef, *, context_window: int = 1_000_000) -> None:
        self._llm = llm
        self._route = route
        usable = max(int(context_window) - _RESERVE_TOKENS, 0)
        self._budget_chars = max(usable * _CHARS_PER_TOKEN, _MIN_BUDGET_CHARS)

    @logged
    def discover_moments(
        self, transcript: List[Dict], min_len: float, max_len: float, target_clips: int
    ) -> List[Dict]:
        """Moments come back already scored (ai_scored=True), snapped to segment edges and
        sized into [min_len, max_len]."""
        segs = [s for s in (transcript or []) if s.get("text")]
        if not segs:
            return []
        segs = self._fit_budget(segs)

        active().info(f"  Reading the full transcript to find moments ({self._route})...")
        chunks = self._chunk_transcript(segs)
        want = max(target_clips, 5)
        raw_lists = self._llm.parallel_map(
            lambda c: self._discover_in_chunk(c, min_len, max_len, want), chunks, desc="  Reading"
        )
        raw = [m for lst in raw_lists if lst for m in lst]

        moments = [
            moment for moment in (self._to_moment(r, segs, min_len, max_len) for r in raw)
            if moment is not None
        ]

        deduped = self._dedupe_overlaps(moments)
        active().success(f"  Chose {len(deduped)} moments from the full transcript "
              f"({len(chunks)} pass(es), {len(raw)} raw picks)")
        return deduped

    def _fit_budget(self, segs: List[Dict]) -> List[Dict]:
        """Cap the transcript to the context-window budget, dropping the tail on overflow."""
        kept: List[Dict] = []
        size = 0
        for s in segs:
            size += len(str(s.get("text") or "")) + 12  # + room for the "[123.4] " prefix
            if size > self._budget_chars:
                break
            kept.append(s)
        if len(kept) < len(segs):
            active().warning(
                f"  Transcript over the {self._budget_chars // _CHARS_PER_TOKEN}-token budget: "
                f"kept the first {len(kept)}/{len(segs)} segments, tail truncated."
            )
        return kept

    def _chunk_transcript(self, segs: List[Dict]) -> List[List[Dict]]:
        """Split into windows that fit one request, overlapping a few segments so a moment
        straddling a boundary is still seen whole. Short transcripts return one window."""
        overlap = 3
        chunks: List[List[Dict]] = []
        cur: List[Dict] = []
        size = 0
        for s in segs:
            text = str(s.get("text") or "")
            cur.append(s)
            size += len(text) + 12  # + room for the "[123.4] " prefix
            if size >= self._budget_chars:
                chunks.append(cur)
                cur = cur[-overlap:]
                size = sum(len(str(x.get("text") or "")) + 12 for x in cur)
        if cur and (not chunks or len(cur) > overlap):
            chunks.append(cur)
        return chunks or [segs]

    def _discover_in_chunk(
        self, chunk: List[Dict], min_len: float, max_len: float, want: int
    ) -> List[Dict]:
        """One request: hand the model a transcript window, get back its best moments."""
        lines = "\n".join(
            f"[{float(s.get('start') or 0):.1f}] {str(s.get('text') or '').strip()[:200]}"
            for s in chunk
        )
        prompt = self._llm.render(
            "prompts.agents.moment_discovery", count=str(want), min_len=f"{min_len:.0f}",
            max_len=f"{max_len:.0f}", transcript=lines,
        )
        if not prompt.strip():
            return []
        content = self._llm.complete(prompt, max_tokens=120 * want + 300, temperature=0.4)
        parsed = self._llm.parse_json_array(content)
        return [p for p in parsed if isinstance(p, dict)] if parsed else []

    def _to_moment(
        self, pick: Dict, segs: List[Dict], min_len: float, max_len: float
    ) -> Optional[Dict]:
        """One model pick snapped to segment edges and sized into [min_len, max_len], or None
        when it has no usable span."""
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

    @staticmethod
    def _dedupe_overlaps(moments: List[Dict]) -> List[Dict]:
        """Keep the highest-scoring moment out of any overlapping set. Greedy, best first."""
        kept: List[Moment] = []
        for moment in sorted((Moment(m) for m in moments), key=lambda m: m.score, reverse=True):
            if all(moment.overlap_ratio(k) <= 0.5 for k in kept):
                kept.append(moment)
        return [m.data for m in kept]
