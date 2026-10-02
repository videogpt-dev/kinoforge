from __future__ import annotations

import re
from typing import Dict, List

from kinoforge.observ import active
from kinoforge.segments.clips.moments.filter import MomentFilter
from kinoforge.segments.clips.moments.finders.base import MomentFinder, MomentSpec
from kinoforge.segments.clips.moments.transcript import TranscriptText

_KEYWORDS = (
    (3.0, ("shocking", "unbelievable", "crazy", "insane", "mind blown", "did not expect",
           "never saw that coming", "plot twist")),
    (3.0, ("happened", "crashed", "exploded", "collapsed", "broke", "failed", "succeeded",
           "won", "beaten", "destroyed")),
    (2.0, ("love", "hate", "proud", "ashamed", "happy", "sad", "angry", "hilarious",
           "awkward", "embarrassing")),
    (2.5, ("actually", "turns out", "secret", "truth", "never knew", "didn't know",
           "find out", "discover", "exposed")),
)
_NUMBERS = r"\d+(?:%|k|m|billion|million|thousand)"
_ENERGY = ((r"\b(wow|omg|oh my god|amazing|incredible|shocking)\b", 3), (r"\b(wow)\b", 5),
           (r"!!!", 2), (r"\?\?", 2))
_HOOKS = (
    (r"\bwait\b.*\b(what|how|why)\b", 5),
    (r"\b(what if|imagine|picture this)\b", 4),
    (r"\b(would you|could you|can you)\b", 3),
    (r"\b(have you ever|did you know)\b", 4),
    (r"\bhold on\b", 3),
    (r"\b(listen|trust me|watch this)\b", 3),
    (r"\b(this is|here\'s|you won\'t|you\'ll|you\'re)\b.*\b(crazy|insane|amazing)\b", 5),
)


class OfflineMomentFinder(MomentFinder):
    """Keyless: standalone transcript windows, kept and ranked by energy, keyword and hook
    text scores."""

    name = "Offline (Smart)"

    def find(self, transcript: List[Dict], spec: MomentSpec) -> List[Dict]:
        active().info("  Finding moments with Offline Smart Analysis...")
        candidates = self.candidates(transcript, spec)
        worthy = [m for m in candidates[:20] if self._worthiness(m.get("text", "")) >= 35]
        moments = worthy or candidates[:10]
        for moment in moments:
            moment["score"] = self._score(moment.get("text", ""))
        return self.ranked(moments)

    @staticmethod
    def candidates(transcript: List[Dict], spec: MomentSpec) -> List[Dict]:
        """Transcript windows that stand alone; every window when none do."""
        windows = TranscriptText.windows(transcript, spec.min_len, spec.max_len)
        active().info(f"  {len(windows)} candidate windows")
        return MomentFilter.run(windows, transcript) or windows

    def _worthiness(self, text: str) -> float:
        return (self._energy(text) + self._keywords(text) + self._hooks(text) * 1.5
                + self._pacing(text) * 0.5 + self._clarity(text) * 0.5)

    def _score(self, text: str) -> float:
        hooks = self._hooks(text)
        score = (self._energy(text) * 0.3 + self._keywords(text) * 0.35 + hooks * 0.25
                 + self._pacing(text) * 0.05 + self._clarity(text) * 0.05)
        if hooks > 10:
            score *= 1.15
        if len(text.split()) > 200:
            score *= 0.8
        return min(max(score, 0), 100)

    @staticmethod
    def _energy(text: str) -> float:
        lower = text.lower()
        score = sum(min(len(re.findall(p, lower, re.IGNORECASE)) * pts, 15) for p, pts in _ENERGY)
        if re.search(r"\b(um|uh|like)\b.*\b(what|wait|stop|hold on)\b", lower):
            score += 5
        return min(score, 30)

    @staticmethod
    def _keywords(text: str) -> float:
        lower = text.lower()
        score = min(len(re.findall(_NUMBERS, lower)) * 2.0, 10)
        score += sum(weight for weight, words in _KEYWORDS for word in words if word in lower)
        return min(score, 30)

    @staticmethod
    def _hooks(text: str) -> float:
        lower = text.lower()
        return min(sum(pts for p, pts in _HOOKS if re.search(p, lower, re.IGNORECASE)), 20)

    @staticmethod
    def _pacing(text: str) -> float:
        sentences = max(len(re.split(r"[.!?]+", text.strip())), 1)
        pacing = (10 if len(text) / sentences < 30 else 5) if text else 0
        intensity = (text.count("!") * 2 + text.count("?")) / sentences
        return min(pacing + min(intensity * 2, 5), 10)

    @staticmethod
    def _clarity(text: str) -> float:
        words = len(text.split())
        if words < 5:
            return 2
        if words > 150:
            return 5
        return 10 if text.lower().endswith((".", "!", "?")) else 7
