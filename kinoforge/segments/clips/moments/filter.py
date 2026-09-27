from __future__ import annotations

import re
from typing import Any, Dict, List

from kinoforge.observ import active


class MomentFilter:
    """Keeps only clips that stand alone. Bound to one language, it runs seven rejection rules
    against each candidate; a moment that trips any rule requires external context and is cut."""

    _TOPIC_PATTERNS: Dict[str, List[str]] = {
        "english": [
            r"^\s*(why|how|what|when|where|who)",
            r"^\s*(do you know|have you ever|did you know)",
            r"^\s*the (secret|truth|reality|key|problem|issue|thing) (is|to|about)",
            r"^\s*(here\'s|let me (tell|show|explain))",
            r"^\s*(\d+\s+(ways|reasons|things|tips))",
            r"\b(the (secret|truth|reality|key|problem|issue) (is|of|to))\b",
            r"\b(actually|really|surprisingly|interestingly|basically)\s+",
            r"\b(one of the|the most|the best|the worst)\b",
        ],
        "hindi": [
            r"(क्यों|कैसे|क्या|कब|कहाँ|कौन)",
            r"(रहस्य|सच|वास्तविकता)",
            r"\d+\s*(तरीके|कारण|टिप्स)",
        ],
        "spanish": [
            r"(por qué|cómo|qué|cuándo|dónde)",
            r"(secreto|verdad|realidad)",
        ],
    }
    _FILLER = {"and", "the", "a", "or", "is", "are", "was", "were", "this", "that", "it", "be"}

    _MID_THOUGHT_PATTERNS: Dict[str, List[str]] = {
        "english": [
            r"^\s*so\s+(i|we|he|she|they|you)",
            r"^\s*because",
            r"^\s*as\s+i\s+(said|mentioned)",
            r"^\s*going back to",
        ],
        "hindi": [r"^\s*(तो|क्योंकि)"],
        "spanish": [r"^\s*(entonces|porque)"],
    }

    _UNCLEAR_PRONOUNS = ["this", "that", "it", "they", "them", "these", "those"]

    _PROBLEM_KEYWORDS: Dict[str, List[str]] = {
        "english": ["why", "how", "what", "problem", "reason", "secret", "truth",
                    "solution", "key", "mistake"],
        "hindi": ["क्यों", "कैसे", "समस्या", "कारण", "समाधान"],
        "spanish": ["por qué", "cómo", "problema", "razón", "solución"],
    }
    _BARE_EXPLANATION = r"^(because|since|due to|as a result|therefore|thus|so|hence)\s+"

    _CONTEXT_PATTERNS: Dict[str, List[str]] = {
        "english": [
            r"\b(remember when|as (i|we) said|earlier|previously)\b",
            r"\b(in (this|that) (video|episode|podcast))\b",
            r"\b(like i mentioned|as discussed)\b",
            r"\b(the other day|last (week|time))\b",
        ],
        "hindi": [
            r"\b(याद है|जैसा मैंने कहा|पहले|पिछले)\b",
            r"\b(इस (वीडियो|एपिसोड|पॉडकास्ट) में)\b",
        ],
        "spanish": [
            r"\b(recuerda cuando|como (yo|nosotros) dijimos|antes|previamente)\b",
            r"\b(en (este|ese) (video|episodio|podcast))\b",
        ],
    }

    _PODCAST_PATTERNS: Dict[str, List[str]] = {
        "english": [
            r"\b(on (this|the) (show|podcast|episode))\b",
            r"\b(my guest|our guest|the guest)\b",
            r"\b(we\'re talking (about|with))\b",
            r"\b(thanks for (having|joining))\b",
        ],
        "hindi": [
            r"\b(इस (शो|पॉडकास्ट|एपिसोड) पर)\b",
            r"\b(मेरे अतिथि|हमारे अतिथि)\b",
        ],
        "spanish": [
            r"\b(en (este|el) (show|podcast|episodio))\b",
            r"\b(mi invitado|nuestro invitado)\b",
        ],
    }

    _BRANDING_PATTERNS: Dict[str, List[str]] = {
        "english": [
            r"\b(subscribe|like|comment|follow|check out)\b",
            r"\b(my (channel|podcast|show|course))\b",
            r"\b(link in (bio|description))\b",
        ],
        "hindi": [
            r"\b(सब्सक्राइब|लाइक|कमेंट|फॉलो)\b",
            r"\b(मेरे (चैनल|पॉडकास्ट|शो))\b",
        ],
        "spanish": [
            r"\b(suscríbete|like|comenta|sigue)\b",
            r"\b(mi (canal|podcast|show))\b",
        ],
    }

    def __init__(self, language: str = "english") -> None:
        self.language = language

    @classmethod
    def run(cls, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Filter a candidate list. Language is taken from the first candidate."""
        if not candidates:
            return []

        language = candidates[0].get("language", "english")
        engine = cls(language)
        active().info(f"  Filtering for language: {language}")

        filtered = []
        rejection_log: List[Dict[str, Any]] = []
        for idx, moment in enumerate(candidates):
            reasons = engine.rejections(moment, transcript)
            if not reasons:
                filtered.append(moment)
            else:
                rejection_log.append({
                    "moment_id": idx,
                    "reasons": reasons,
                    "text_preview": moment["text"][:100],
                })

        active().info(f"  Rejected {len(rejection_log)}/{len(candidates)} moments:")
        for log in rejection_log[:5]:
            active().info(f"    - {', '.join(log['reasons'])}")
        if len(rejection_log) > 5:
            active().info(f"    ... and {len(rejection_log) - 5} more")

        return filtered

    def rejections(self, moment: Dict, transcript: List[Dict]) -> List[str]:
        """Run the 7 rejection rules against one moment. Empty list means keep it."""
        text = moment["text"]

        # First 2s of speech, widened to 4s when the opening 2s is silent.
        first_2s_text = self.window_text(transcript, moment["start"], moment["start"] + 2)
        if not first_2s_text:
            first_2s_text = self.window_text(transcript, moment["start"], moment["start"] + 4)

        reasons: List[str] = []
        if not self.has_clear_topic(first_2s_text):
            reasons.append("No clear topic/problem in first 2s, Requires external context")
        if self.starts_mid_thought(text):
            reasons.append("Starts mid-thought")
        if self.language == "english" and self.has_unclear_pronouns(first_2s_text):
            reasons.append("Unclear pronouns without reference")
        if self.is_explanation_without_question(text):
            reasons.append("Explanation without stated question/problem")
        if self.requires_context(text):
            reasons.append("Requires external context")
        if self.has_podcast_dependency(text):
            reasons.append("Requires podcast context")
        if self.has_branding_before_insight(text):
            reasons.append("Branding appears before insight")
        return reasons

    @staticmethod
    def window_text(transcript: List[Dict], start_time: float, end_time: float) -> str:
        """Text from segments that OVERLAP the time window (not just fully contained)."""
        if not transcript or start_time >= end_time:
            return ""
        text_parts = [
            segment.get("text", "")
            for segment in transcript
            if segment.get("start", 0) < end_time and segment.get("end", 0) > start_time
        ]
        return " ".join(text_parts).strip()

    def has_clear_topic(self, text: str) -> bool:
        """A clear topic, question, or problem: question mark, number, a known pattern, or a
        substantive statement (3+ meaningful words)."""
        if len(text.strip()) < 5:
            return False
        if "?" in text or "？" in text or bool(re.search(r"\d+", text)):
            return True
        for pattern in self._TOPIC_PATTERNS.get(self.language, []):
            if re.search(pattern, text, re.IGNORECASE):
                return True
        words = text.lower().split()
        if len(text.strip()) >= 10 and len(words) >= 3:
            meaningful = [w for w in words if w not in self._FILLER and len(w) > 2]
            if len(meaningful) >= 2:
                return True
        return False

    def starts_mid_thought(self, text: str) -> bool:
        """Catch only OBVIOUS mid-thought openers; conservative to avoid false positives."""
        for pattern in self._MID_THOUGHT_PATTERNS.get(self.language, []):
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def has_unclear_pronouns(self, text: str) -> bool:
        """A pronoun with no antecedent in the first three words of the first sentence."""
        first_sentence = text.split(".")[0] if "." in text else text
        words = first_sentence.lower().split()[:10]
        for pronoun in self._UNCLEAR_PRONOUNS:
            if pronoun in words and words.index(pronoun) < 3:
                return True
        return False

    def is_explanation_without_question(self, text: str) -> bool:
        """Bare explanation with no stated problem: no question, no problem keyword, and it
        opens on a connective (because/since/...)."""
        if "?" in text or "？" in text:
            return False
        if any(kw in text.lower() for kw in self._PROBLEM_KEYWORDS.get(self.language, [])):
            return False
        return bool(re.match(self._BARE_EXPLANATION, text.lower().strip()))

    def requires_context(self, text: str) -> bool:
        for pattern in self._CONTEXT_PATTERNS.get(self.language, []):
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def has_podcast_dependency(self, text: str) -> bool:
        for pattern in self._PODCAST_PATTERNS.get(self.language, []):
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def has_branding_before_insight(self, text: str) -> bool:
        first_sentence = text.split(".")[0] if "." in text else text[:100]
        for pattern in self._BRANDING_PATTERNS.get(self.language, []):
            if re.search(pattern, first_sentence, re.IGNORECASE):
                return True
        return False
