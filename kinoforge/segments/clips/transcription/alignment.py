"""Fit a known narration script to recognized word timings, for captions."""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(token: str) -> str:
    """Lowercased letters/digits only, so 'AI', 'to-do', "Moon's" match on their content."""
    return _ALNUM.sub("", token.lower())


def _heard_words(segments: List[Dict]) -> List[Tuple[str, float, float]]:
    heard = []
    for seg in segments or []:
        for word in seg.get("words") or []:
            start, end = word.get("start"), word.get("end")
            if start is not None and end is not None:
                heard.append((_norm(word.get("word") or ""), float(start), float(end)))
    return heard


def _anchor(truth: List[str], heard: List[Tuple[str, float, float]]
            ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Timing for each script word the recognizer heard verbatim; None where it didn't."""
    starts: List[Optional[float]] = [None] * len(truth)
    ends: List[Optional[float]] = [None] * len(truth)
    if not heard:
        return starts, ends
    matcher = difflib.SequenceMatcher(
        a=[h[0] for h in heard], b=[_norm(t) for t in truth], autojunk=False
    )
    for tag, i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                starts[j1 + k], ends[j1 + k] = heard[i1 + k][1], heard[i1 + k][2]
    return starts, ends


def _interpolate(starts: List[Optional[float]], ends: List[Optional[float]],
                 duration: float) -> Tuple[List[float], List[float]]:
    """Spread each run of unanchored words (merged/dropped/misheard) evenly across the gap
    between its neighbours: 0 on the left, the clip duration on the right."""
    n = len(starts)
    out_starts, out_ends = [0.0] * n, [0.0] * n
    i, prev = 0, 0.0
    while i < n:
        if starts[i] is not None and ends[i] is not None:
            out_starts[i], out_ends[i] = starts[i], ends[i]
            prev = ends[i]
            i += 1
            continue
        j = i
        while j < n and (starts[j] is None or ends[j] is None):
            j += 1
        right = starts[j] if j < n and starts[j] is not None else duration
        left = min(prev, right)
        width = (right - left) / (j - i)
        for k in range(j - i):
            out_starts[i + k], out_ends[i + k] = left + k * width, left + (k + 1) * width
        prev = out_ends[j - 1]
        i = j
    return out_starts, out_ends


def align_words(text: str, segments: List[Dict], duration: float) -> List[Dict]:
    """The recognizer's timing, never its text: each script word is anchored to the matching
    heard word and the rest are interpolated, giving correctly spelled captions that track the
    voice. One segment: [{start, end, text, words: [{word, start, end}]}]."""
    truth = [t for t in text.split() if t]
    if not truth:
        return []
    duration = max(float(duration or 0), 0.1)
    starts, ends = _interpolate(*_anchor(truth, _heard_words(segments)), duration)
    words = [{"word": w, "start": round(s, 3), "end": round(e, 3)}
             for w, s, e in zip(truth, starts, ends)]
    return [{"start": 0.0, "end": round(duration, 3), "text": " ".join(truth), "words": words}]
