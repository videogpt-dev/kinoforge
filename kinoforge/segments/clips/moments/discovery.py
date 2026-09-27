from __future__ import annotations

from typing import Dict, List

from kinoforge.segments.clips.moments.llm_client import LlmClient
from kinoforge.observ import active, logged

_CHARS_PER_TOKEN = 4  # rough average for transcript prose, not code
_RESERVE_TOKENS = 8000  # held back for the prompt template and the model's own answer
_MIN_BUDGET_CHARS = 4000


class MomentDiscoverer:
    """Transcript-first discovery: read the transcript (chunked when long) and pick
    self-contained moments with real start/end, snapped to segment edges.

    The transcript is capped to the model's context window (context_window tokens, default
    1M): it fits in a single pass in the common case, and an over-long transcript has its
    tail truncated rather than overflowing the window or fanning out into many requests."""

    def __init__(
        self, llm: LlmClient, provider: str, name: str, *, context_window: int = 1_000_000
    ) -> None:
        self._llm = llm
        self.provider = provider
        self.name = name
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

        from kinoforge.segments.clips.moments.extractor import TranscriptText

        active().info(f"  Reading the full transcript to find moments ({self.name})...")
        chunks = self._chunk_transcript(segs)
        want = max(target_clips, 5)
        raw_lists = self._llm.parallel_map(
            lambda c: self._discover_in_chunk(c, min_len, max_len, want), chunks, desc="  Reading"
        )
        raw = [m for lst in raw_lists if lst for m in lst]

        moments: List[Dict] = []
        for r in raw:
            try:
                rs, re_ = float(r["start"]), float(r["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if re_ <= rs:
                continue
            s, e, text = TranscriptText.snap(segs, rs, re_, min_len, max_len)
            if e - s < 2.0:
                continue
            moments.append({
                "start": s, "end": e, "duration": e - s, "text": text,
                "score": min(max(float(r.get("score") or 60), 0), 100),
                "ai_reason": str(r.get("reason", ""))[:200],
                "ai_hook": str(r.get("hook", ""))[:120],
                "title": str(r.get("title", ""))[:120],
                "language": TranscriptText.detect_language(text),
                "ai_scored": True, "provider": self.provider,
                "source": "transcript_discovery",
            })

        deduped = self._dedupe_overlaps(moments)
        active().success(f"  Chose {len(deduped)} moments from the full transcript "
              f"({len(chunks)} pass(es), {len(raw)} raw picks)")
        return deduped

    def _fit_budget(self, segs: List[Dict]) -> List[Dict]:
        """Cap the transcript to the context-window budget, dropping the tail when it overflows.
        Keeps the whole clip in one pass in the common case (a 1M window holds hours of speech)."""
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

    @staticmethod
    def _dedupe_overlaps(moments: List[Dict]) -> List[Dict]:
        """Keep the highest-scoring moment out of any overlapping set. Greedy, best first."""
        ordered = sorted(moments, key=lambda m: float(m.get("score") or 0), reverse=True)
        kept: List[Dict] = []
        for m in ordered:
            ms, me = float(m["start"]), float(m["end"])
            clash = False
            for k in kept:
                ks, ke = float(k["start"]), float(k["end"])
                overlap = max(0.0, min(me, ke) - max(ms, ks))
                shorter = min(me - ms, ke - ks) or 1.0
                if overlap / shorter > 0.5:
                    clash = True
                    break
            if not clash:
                kept.append(m)
        return kept
