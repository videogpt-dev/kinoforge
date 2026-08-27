from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class _KeywordSet:
    """One scored category: literal words, or a regex when the signal is a shape rather than
    a vocabulary (numbers)."""

    weight: float
    words: Tuple[str, ...] = ()
    pattern: str = ""


_KEYWORD_SETS: Tuple[_KeywordSet, ...] = (
    _KeywordSet(3, ('shocking', 'unbelievable', 'crazy', 'insane', 'mind blown',
                    'did not expect', 'never saw that coming', 'plot twist')),
    _KeywordSet(3, ('happened', 'crashed', 'exploded', 'collapsed', 'broke',
                    'failed', 'succeeded', 'won', 'beaten', 'destroyed')),
    _KeywordSet(2, ('love', 'hate', 'proud', 'ashamed', 'happy', 'sad',
                    'angry', 'hilarious', 'awkward', 'embarrassing')),
    _KeywordSet(2.5, ('actually', 'turns out', 'secret', 'truth', 'never knew',
                      "didn't know", 'find out', 'discover', 'exposed')),
    _KeywordSet(2, pattern=r'\d+(?:%|k|m|billion|million|thousand)'),
)


class OfflineMomentEngine:
    """Keyless moment scoring with energy + keyword + hook analysis."""

    def __init__(self):
        self.name = "Offline (Smart)"
        self.provider = "local"

    def health_check(self) -> bool:
        return True

    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Smart local filtering using energy + keywords + hooks."""
        if not candidates:
            return []

        print("  Filtering with Offline Smart Analysis...")
        from kinoforge.clips.moments.filter import filter_moments_aggressively

        pre_filtered = filter_moments_aggressively(candidates, transcript)
        if not pre_filtered:
            return []

        filtered = [m for m in pre_filtered[:20] if self._is_viral_worthy_local(m, transcript)]
        return filtered if filtered else pre_filtered[:10]

    def _is_viral_worthy_local(self, moment: Dict, transcript: List[Dict]) -> bool:
        signals = {
            'energy': self._check_energy_spike(moment),
            'keywords': self._check_viral_keywords(moment),
            'hooks': self._check_hook_pattern(moment),
            'pacing': self._check_pacing_energy(moment),
            'clarity': self._check_clarity(moment),
        }
        total_score = (
            signals['energy'] * 1.0
            + signals['keywords'] * 1.0
            + signals['hooks'] * 1.5
            + signals['pacing'] * 0.5
            + signals['clarity'] * 0.5
        )
        return total_score >= 35

    def _check_energy_spike(self, moment: Dict) -> float:
        text = moment.get('text', '').lower()
        energy_markers = [
            (r'\b(wow|omg|oh my god|amazing|incredible|shocking)\b', 3),
            (r'\b(wow)\b', 5),
            (r'!!!', 2),
            (r'\?\?', 2),
            (r'(all caps words)', 3),
        ]
        score = 0
        for pattern, points in energy_markers:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += min(matches * points, 15)
        if re.search(r'\b(um|uh|like)\b.*\b(what|wait|stop|hold on)\b', text):
            score += 5
        return min(score, 30)

    def _check_viral_keywords(self, moment: Dict) -> float:
        text = moment.get('text', '').lower()
        score = 0.0
        for config in _KEYWORD_SETS:
            if config.pattern:
                matches = len(re.findall(config.pattern, text))
                score += min(matches * config.weight, 10)
            else:
                for word in config.words:
                    if word in text:
                        score += config.weight
        return min(score, 30)

    def _check_hook_pattern(self, moment: Dict) -> float:
        text = moment.get('text', '').lower()
        hook_patterns = [
            (r'\bwait\b.*\b(what|how|why)\b', 5),
            (r'\b(what if|imagine|picture this)\b', 4),
            (r'\b(would you|could you|can you)\b', 3),
            (r'\b(have you ever|did you know)\b', 4),
            (r'\bhold on\b', 3),
            (r'\b(listen|trust me|watch this)\b', 3),
            (r'\b(this is|here\'s|you won\'t|you\'ll|you\'re)\b.*\b(crazy|insane|amazing)\b', 5),
        ]
        score = 0
        for pattern, points in hook_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += points
        return min(score, 20)

    def _check_pacing_energy(self, moment: Dict) -> float:
        text = moment.get('text', '')
        sentences = len(re.split(r'[.!?]+', text.strip()))
        exclamations = len(re.findall(r'!', text))
        questions = len(re.findall(r'\?', text))
        if len(text) > 0:
            avg_sentence_length = len(text) / max(sentences, 1)
            pacing_score = 10 if avg_sentence_length < 30 else 5
        else:
            pacing_score = 0
        punctuation_intensity = (exclamations * 2 + questions) / max(sentences, 1)
        intensity_score = min(punctuation_intensity * 2, 5)
        return min(pacing_score + intensity_score, 10)

    def _check_clarity(self, moment: Dict) -> float:
        text = moment.get('text', '').lower()
        word_count = len(text.split())
        if word_count < 5:
            return 2
        if word_count > 150:
            return 5
        return 10 if text.endswith(('.', '!', '?')) else 7

    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        print("  Scoring with Offline Smart Analysis...")
        for moment in moments:
            moment['score'] = self._calculate_smart_score(moment, transcript)
            moment['ai_method'] = 'local_smart'
            moment['scoring_factors'] = self._get_scoring_explanation(moment)
        return sorted(moments, key=lambda m: m['score'], reverse=True)

    def _calculate_smart_score(self, moment: Dict, transcript: List[Dict]) -> float:
        energy_score = self._check_energy_spike(moment)
        keyword_score = self._check_viral_keywords(moment)
        hook_score = self._check_hook_pattern(moment)
        pacing_score = self._check_pacing_energy(moment)
        clarity_score = self._check_clarity(moment)
        base_score = (
            energy_score * 0.3
            + keyword_score * 0.35
            + hook_score * 0.25
            + pacing_score * 0.05
            + clarity_score * 0.05
        )
        if self._check_hook_pattern(moment) > 10:
            base_score *= 1.15
        if len(moment.get('text', '').split()) > 200:
            base_score *= 0.8
        return min(max(base_score, 0), 100)

    def _get_scoring_explanation(self, moment: Dict) -> Dict:
        return {
            'energy': self._check_energy_spike(moment),
            'keywords': self._check_viral_keywords(moment),
            'hooks': self._check_hook_pattern(moment),
            'pacing': self._check_pacing_energy(moment),
            'clarity': self._check_clarity(moment),
            'method': 'local_smart_analysis',
        }
