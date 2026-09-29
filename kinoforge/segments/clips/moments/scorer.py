"""Keyless fallback ranking for moments when the moment provider cannot score them."""

from __future__ import annotations

import re
from typing import Dict, List

from kinoforge.segments.clips.moments.moment import Moment

_VAGUE_REFS = ("this", "that", "it", "they", "those", "these")
_SENTENCE_ENDINGS = (".", "!", "?", "।")
_HOOKS = {
    "english": [
        (r"\b(secret|hidden|truth|reality)\b", 3.0),
        (r"\b(never|always|nobody|everyone)\b", 2.5),
        (r"^(why|how|what)", 2.0),
        (r"\b(mistake|wrong|problem)\b", 2.0),
    ],
    "hindi": [
        (r"(रहस्य|सच|वास्तविकता)", 3.0),
        (r"(क्यों|कैसे|क्या)", 2.0),
        (r"(गलती|समस्या|गलत)", 2.0),
    ],
    "spanish": [
        (r"(secreto|verdad|realidad)", 3.0),
        (r"(por qué|cómo|qué)", 2.0),
    ],
}
_ENGAGEMENT = (
    r"\b(you|your)\b",                    # direct address
    r"\b(imagine|picture|think about)\b",  # mental imagery
    r"\?\s*\w+",                           # a question answered
    r"\b(first|second|finally)\b",         # structure
)


def _clamp(score: float) -> float:
    return max(0, min(10, score))


class HeuristicScorer:
    """Four 0-10 dimensions averaged into `score` (with the parts under `scores`): context
    clarity, hook strength, standalone understanding, retention. Language comes from the
    first moment."""

    @classmethod
    def rank(cls, moments: List[Dict]) -> List[Dict]:
        """Scored copies, best first."""
        if not moments:
            return []
        language = moments[0].get("language", "english")
        scored = [cls._scored(moment, language) for moment in moments]
        return sorted(scored, key=lambda m: m["score"], reverse=True)

    @classmethod
    def _scored(cls, moment: Dict, language: str) -> Dict:
        scores = {
            "context_clarity": cls.context_clarity(moment["text"], language),
            "hook_strength": cls.hook_strength(moment["text"], language),
            "standalone": cls.standalone(moment["text"]),
            "retention": cls.retention(moment["text"], Moment(moment).span),
        }
        return {**moment, "scores": scores, "score": round(sum(scores.values()) / len(scores), 2)}

    @staticmethod
    def context_clarity(text: str, language: str) -> float:
        """Self-contained: questions and numbers help; opening on a vague reference hurts."""
        score = 10.0
        if "?" in text or "？" in text:
            score += 1.0
        if re.search(r"\d+", text):
            score += 0.5
        if language == "english":
            opening = set(re.findall(r"[a-z']+", " ".join(text.split()[:20]).lower()))
            score -= 1.5 * sum(1 for ref in _VAGUE_REFS if ref in opening)
        return _clamp(score)

    @staticmethod
    def hook_strength(text: str, language: str) -> float:
        """How attention-grabbing the first ten words are; one strong hook counts."""
        opening = " ".join(text.split()[:10]).lower()
        score = 5.0
        if "?" in opening or "？" in opening:
            score += 2.0
        if re.search(r"\d+", opening):
            score += 1.5
        for pattern, points in _HOOKS.get(language, []):
            if re.search(pattern, opening, re.IGNORECASE):
                score += points
                break
        return _clamp(score)

    @staticmethod
    def standalone(text: str) -> float:
        """A new viewer follows it: complete sentences and answered questions."""
        score = 8.0
        if text.strip().endswith(_SENTENCE_ENDINGS):
            score += 1.0
        if "?" in text and len(text.split("?")) > 1:
            score += 1.5
        return _clamp(score)

    @staticmethod
    def retention(text: str, duration: float) -> float:
        """Kept watching: 30-45s is the sweet spot, engagement cues and 3-5 sentences help."""
        score = 7.0
        if 30 <= duration <= 45:
            score += 2.0
        elif 45 < duration <= 60:
            score += 1.0
        if duration > 55:
            score -= 1.0
        score += 0.5 * sum(1 for p in _ENGAGEMENT if re.search(p, text, re.IGNORECASE))
        if 3 <= text.count(".") + text.count("!") + text.count("?") <= 5:
            score += 1.0
        return _clamp(score)
