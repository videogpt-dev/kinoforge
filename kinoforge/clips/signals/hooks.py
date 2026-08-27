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

        opening_text = ""
        for segment in transcript:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            if (seg_start >= moment_start and seg_end <= moment_end) or (
                seg_start < moment_end and seg_end > moment_start
            ):
                opening_text += " " + segment.get("text", "")

        opening_text = opening_text.strip()
        if not opening_text:
            return HookSignal("none", 0.0, "", 1.0, ["No text in opening 3 seconds"])

        signals: List[HookSignal] = []
        lower_text = opening_text.lower()

        has_question = any(q in opening_text for q in self.QUESTION_MARKS)
        starts_with_question = any(lower_text.startswith(qw) for qw in self.QUESTION_WORDS)
        if has_question or starts_with_question:
            confidence = 1.0 if has_question else 0.85
            reason = "Question mark detected" if has_question else "Question word at start"
            signals.append(HookSignal("question", 10.0, opening_text[:80], confidence, [reason]))

        for level, words in self.SURPRISING.items():
            for word in words:
                if word in lower_text:
                    signals.append(HookSignal(
                        "surprising", self.SURPRISING_STRENGTH[level], opening_text[:80],
                        0.85, [f'Surprising word: "{word}"'],
                    ))
                    break
            if signals and signals[-1].hook_type == "surprising":
                break

        for pattern, desc, strength in self.NUMBER_PATTERNS:
            if re.search(pattern, opening_text, re.IGNORECASE):
                signals.append(HookSignal(
                    "data", strength, opening_text[:80], 0.9, [f"Numeric pattern: {desc}"],
                ))
                break

        for category, phrases in self.CTA.items():
            for phrase in phrases:
                if phrase in lower_text:
                    signals.append(HookSignal(
                        "cta", self.CTA_STRENGTH[category], opening_text[:80],
                        0.75, [f'CTA phrase: "{phrase}"'],
                    ))
                    break
            if signals and signals[-1].hook_type == "cta":
                break

        for trigger in self.EMOTIONAL:
            if trigger in lower_text:
                signals.append(HookSignal(
                    "emotional", 7.5, opening_text[:80], 0.7, [f'Emotional trigger: "{trigger}"'],
                ))
                break

        if any(word in lower_text for word in self.URGENCY):
            signals.append(HookSignal(
                "urgency", 7.0, opening_text[:80], 0.7, ["Urgency/scarcity language"],
            ))

        if any(lower_text.startswith(v) for v in self.VAGUE) and signals:
            for signal in signals:
                signal.strength *= 0.8
                signal.reasons.append("Penalty: vague start")

        if signals:
            return max(signals, key=lambda s: s.strength * s.confidence)

        return HookSignal("none", 0.0, opening_text[:80], 1.0, ["No hook patterns detected"])
