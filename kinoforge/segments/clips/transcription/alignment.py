from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

_ALNUM = re.compile(r"[^a-z0-9]+")

Heard = List[Tuple[str, float, float]]
Times = List[Optional[float]]


class WordAligner:
    """The recognizer's timing, never its text: each script word is anchored to the matching
    heard word and the rest are interpolated, giving correctly spelled captions that track the
    voice."""

    @classmethod
    def align(cls, text: str, segments: List[Dict], duration: float) -> List[Dict]:
        """One segment: [{start, end, text, words: [{word, start, end}]}]."""
        truth = [t for t in text.split() if t]
        if not truth:
            return []
        duration = max(float(duration or 0), 0.1)
        starts, ends = cls._interpolate(*cls._anchor(truth, cls._heard(segments)), duration)
        words = [{"word": w, "start": round(s, 3), "end": round(e, 3)}
                 for w, s, e in zip(truth, starts, ends)]
        return [{"start": 0.0, "end": round(duration, 3), "text": " ".join(truth),
                 "words": words}]

    @staticmethod
    def _norm(token: str) -> str:
        return _ALNUM.sub("", token.lower())

    @classmethod
    def _heard(cls, segments: List[Dict]) -> Heard:
        heard = []
        for seg in segments or []:
            for word in seg.get("words") or []:
                start, end = word.get("start"), word.get("end")
                if start is not None and end is not None:
                    heard.append((cls._norm(word.get("word") or ""), float(start), float(end)))
        return heard

    @classmethod
    def _anchor(cls, truth: List[str], heard: Heard) -> Tuple[Times, Times]:
        starts: Times = [None] * len(truth)
        ends: Times = [None] * len(truth)
        if not heard:
            return starts, ends
        matcher = difflib.SequenceMatcher(
            a=[h[0] for h in heard], b=[cls._norm(t) for t in truth], autojunk=False
        )
        for tag, i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(j2 - j1):
                    starts[j1 + k], ends[j1 + k] = heard[i1 + k][1], heard[i1 + k][2]
        return starts, ends

    @staticmethod
    def _interpolate(starts: Times, ends: Times, duration: float
                     ) -> Tuple[List[float], List[float]]:
        n = len(starts)
        out_starts, out_ends = [0.0] * n, [0.0] * n
        i, prev = 0, 0.0
        while i < n:
            start, end = starts[i], ends[i]
            if start is not None and end is not None:
                out_starts[i], out_ends[i] = start, end
                prev = end
                i += 1
                continue
            j = i
            while j < n and (starts[j] is None or ends[j] is None):
                j += 1
            next_start = starts[j] if j < n else None
            right = next_start if next_start is not None else duration
            left = min(prev, right)
            width = (right - left) / (j - i)
            for k in range(j - i):
                out_starts[i + k], out_ends[i + k] = left + k * width, left + (k + 1) * width
            prev = out_ends[j - 1]
            i = j
        return out_starts, out_ends
