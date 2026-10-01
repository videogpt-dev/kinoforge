from __future__ import annotations

import re
from typing import Dict, List, Tuple

from kinoforge.segments.clips.moments.moment import Moment


class TranscriptText:
    """Transcript utilities: boundary snapping, candidate windows, language guess."""

    @classmethod
    def snap(cls, transcript: List[Dict], start: float, end: float,
             min_len: float = 0.0, max_len: float = 0.0) -> Tuple[float, float, str]:
        """(start, end, text) of [start, end] snapped onto whole segments and sized into
        [min_len, max_len], so a clip opens and closes on a complete sentence."""
        segs = sorted((s for s in transcript if s.get("text")), key=lambda s: Moment(s).start)
        if not segs:
            return start, end, ""
        first = max([i for i, s in enumerate(segs) if Moment(s).start <= start] or [0])
        last = next((i for i in range(first, len(segs)) if Moment(segs[i]).end >= end),
                    len(segs) - 1)
        while min_len > 0 and cls._span(segs, first, last) < min_len and (
                last < len(segs) - 1 or first > 0):
            if last < len(segs) - 1:
                last += 1
            else:
                first -= 1
        while max_len > 0 and cls._span(segs, first, last) > max_len and last > first:
            last -= 1
        text = " ".join((s.get("text") or "").strip() for s in segs[first:last + 1]).strip()
        return Moment(segs[first]).start, Moment(segs[last]).end, text

    @classmethod
    def windows(cls, transcript: List[Dict], min_len: float, max_len: float) -> List[Dict]:
        """Non-overlapping candidate windows across the whole transcript, each between min_len
        and max_len seconds with at least 5 words."""
        candidates = []
        i, n = 0, len(transcript)
        while i < n:
            start = end = transcript[i]["start"]
            parts = []
            j = i
            while j < n and transcript[j]["end"] - start <= max_len:
                parts.append(transcript[j]["text"])
                end = transcript[j]["end"]
                j += 1
                if end - start >= min_len:
                    break
            text = " ".join(parts).strip()
            if min_len <= end - start <= max_len and len(text.split()) >= 5:
                candidates.append(Moment.new(start, end, text=text,
                                             language=cls.detect_language(text)))
                i = j
            else:
                i += 1
        return candidates

    @staticmethod
    def detect_language(text: str) -> str:
        """The script with the most characters wins, so a stray glyph cannot flip the label."""
        counts = {
            "hindi": len(re.findall(r"[ऀ-ॿ]", text)),
            "chinese": len(re.findall(r"[一-鿿]", text)),
            "arabic": len(re.findall(r"[؀-ۿ]", text)),
        }
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else "english"

    @staticmethod
    def _span(segs: List[Dict], first: int, last: int) -> float:
        return Moment(segs[last]).end - Moment(segs[first]).start
