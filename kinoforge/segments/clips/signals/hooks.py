from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional


@dataclass
class HookSignal:
    hook_type: str
    strength: float
    text: str
    confidence: float
    reasons: List[str] = field(default_factory=list)


class HookDetector:
    QUESTION_MARKS: ClassVar[tuple] = ("?", "？", "¿")
    QUESTION_WORDS: ClassVar[List[str]] = [
        "what", "why", "how", "when", "where", "who", "which",
        "can", "could", "would", "should", "will", "did", "does",
        "is", "are", "was", "were",
    ]
    SURPRISING: ClassVar[Dict[str, List[str]]] = {
        "strong": ["actually", "surprisingly", "shocking", "incredible",
                   "unbelievable", "secret", "truth", "reality", "wrong"],
        "medium": ["wait", "but", "however", "though", "never knew",
                   "didn't know", "realize", "turns out", "plot twist"],
        "weak": ["interesting", "fascinating", "curious", "unusual"],
    }
    SURPRISING_STRENGTH: ClassVar[Dict[str, float]] = {"strong": 9.0, "medium": 8.0, "weak": 7.0}
    NUMBER_PATTERNS: ClassVar[list] = [
        (r"\d+%", "percentage", 8.0),
        (r"\d{4,}", "large number", 7.5),
        (r"\d+\s*(?:million|billion|thousand)", "magnitude", 8.0),
        (r"\d+[-–]\d+", "range", 7.0),
        (r"#?\d+\s+(?:ways|reasons|tips|secrets|facts)", "listicle", 9.0),
        (r"\d+", "number", 7.0),
    ]
    CTA: ClassVar[Dict[str, List[str]]] = {
        "direct": ["watch this", "check this out", "look at this", "see this"],
        "instructive": ["here's", "let me show", "let me tell", "i'll show"],
        "imperative": ["listen", "understand", "learn", "discover", "find out"],
        "engaging": ["imagine", "picture this", "think about", "consider"],
    }
    CTA_STRENGTH: ClassVar[Dict[str, float]] = {
        "direct": 8.0, "instructive": 7.0, "imperative": 6.5, "engaging": 7.5,
    }
    EMOTIONAL: ClassVar[List[str]] = [
        "love", "hate", "fear", "worry", "excited", "angry",
        "frustrated", "amazing", "terrible", "best", "worst",
        "dangerous", "safe", "risky", "genius", "stupid",
    ]
    URGENCY: ClassVar[List[str]] = [
        "now", "today", "immediately", "quickly", "before",
        "limited", "only", "last chance", "hurry",
    ]
    VAGUE: ClassVar[List[str]] = [
        "so", "and", "um", "uh", "like", "basically", "literally", "you know", "i mean",
    ]

    def analyze(
        self,
        transcript: List[Dict],
        moment_start: float,
        moment_end: Optional[float] = None,
    ) -> HookSignal:
        if moment_end is None:
            moment_end = moment_start + 3.0

        opening_text = self._opening_text(transcript, moment_start, moment_end)
        if not opening_text:
            return HookSignal("none", 0.0, "", 1.0, ["No text in opening 3 seconds"])

        lower_text = opening_text.lower()
        detectors = (
            self._question,
            self._surprising,
            self._numeric,
            self._cta,
            self._emotional,
            self._urgency,
        )
        signals = [sig for sig in (probe(opening_text, lower_text) for probe in detectors) if sig]

        if signals and any(lower_text.startswith(v) for v in self.VAGUE):
            for signal in signals:
                signal.strength *= 0.8
                signal.reasons.append("Penalty: vague start")

        if signals:
            return max(signals, key=lambda s: s.strength * s.confidence)
        return HookSignal("none", 0.0, opening_text[:80], 1.0, ["No hook patterns detected"])

    @staticmethod
    def _opening_text(transcript: List[Dict], start: float, end: float) -> str:
        parts: List[str] = []
        for segment in transcript:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            if (seg_start >= start and seg_end <= end) or (seg_start < end and seg_end > start):
                parts.append(segment.get("text", ""))
        return " ".join(parts).strip()

    def _question(self, opening_text: str, lower_text: str) -> Optional[HookSignal]:
        has_question = any(q in opening_text for q in self.QUESTION_MARKS)
        starts_with_question = any(lower_text.startswith(qw) for qw in self.QUESTION_WORDS)
        if not (has_question or starts_with_question):
            return None
        confidence = 1.0 if has_question else 0.85
        reason = "Question mark detected" if has_question else "Question word at start"
        return HookSignal("question", 10.0, opening_text[:80], confidence, [reason])

    def _surprising(self, opening_text: str, lower_text: str) -> Optional[HookSignal]:
        for level, words in self.SURPRISING.items():
            for word in words:
                if word in lower_text:
                    return HookSignal(
                        "surprising", self.SURPRISING_STRENGTH[level], opening_text[:80],
                        0.85, [f'Surprising word: "{word}"'],
                    )
        return None

    def _numeric(self, opening_text: str, lower_text: str) -> Optional[HookSignal]:
        for pattern, desc, strength in self.NUMBER_PATTERNS:
            if re.search(pattern, opening_text, re.IGNORECASE):
                return HookSignal(
                    "data", strength, opening_text[:80], 0.9, [f"Numeric pattern: {desc}"],
                )
        return None

    def _cta(self, opening_text: str, lower_text: str) -> Optional[HookSignal]:
        for category, phrases in self.CTA.items():
            for phrase in phrases:
                if phrase in lower_text:
                    return HookSignal(
                        "cta", self.CTA_STRENGTH[category], opening_text[:80],
                        0.75, [f'CTA phrase: "{phrase}"'],
                    )
        return None

    def _emotional(self, opening_text: str, lower_text: str) -> Optional[HookSignal]:
        for trigger in self.EMOTIONAL:
            if trigger in lower_text:
                return HookSignal(
                    "emotional", 7.5, opening_text[:80], 0.7, [f'Emotional trigger: "{trigger}"'],
                )
        return None

    def _urgency(self, opening_text: str, lower_text: str) -> Optional[HookSignal]:
        if any(word in lower_text for word in self.URGENCY):
            return HookSignal(
                "urgency", 7.0, opening_text[:80], 0.7, ["Urgency/scarcity language"],
            )
        return None
