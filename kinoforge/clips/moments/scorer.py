"""
Score and rank moments by quality metrics
"""

import re
from typing import Dict, List


def score_and_rank_moments(
        moments: List[Dict],
        transcript: List[Dict]
) -> List[Dict]:
    """
    Score moments on multiple dimensions and rank by total score

    Scoring dimensions (0-10 each):
    1. Context clarity - How standalone is it?
    2. Hook strength - How attention-grabbing?
    3. Standalone understanding - Can new viewer understand?
    4. Retention potential - Will they watch to the end?

    Args:
        moments: Filtered candidate moments
        transcript: Full transcript

    Returns:
        Scored and sorted moments (highest first)
    """
    if not moments:
        return []

    # Get language from first moment
    language = moments[0].get('language', 'english')

    scored_moments = []

    for moment in moments:
        scores = {
            'context_clarity': score_context_clarity(moment, language),
            'hook_strength': score_hook_strength(moment, language),
            'standalone': score_standalone_understanding(moment, language),
            'retention': score_retention_potential(moment, language)
        }

        # Calculate weighted total (all equal weight)
        total_score = sum(scores.values()) / len(scores)

        moment_with_score = moment.copy()
        moment_with_score['scores'] = scores
        moment_with_score['score'] = round(total_score, 2)

        scored_moments.append(moment_with_score)

    # Sort by total score (descending)
    scored_moments.sort(key=lambda x: x['score'], reverse=True)

    return scored_moments


def score_context_clarity(moment: Dict, language: str = 'english') -> float:
    """
    Score how clear the context is (0-10)
    Higher = more self-contained
    """
    text = moment['text']
    score = 10.0

    # Universal: Check for question marks (always good)
    if '?' in text or '？' in text:
        score += 1.0

    # Universal: Check for numbers (often indicates structure)
    if re.search(r'\d+', text):
        score += 0.5

    # Language-specific deductions for vague references
    if language == 'english':
        vague_refs = ['this', 'that', 'it', 'they', 'those', 'these']
        first_20_words = ' '.join(text.split()[:20]).lower()
        for ref in vague_refs:
            if ref in first_20_words:
                score -= 1.5

    return max(0, min(10, score))


def score_hook_strength(moment: Dict, language: str = 'english') -> float:
    """
    Score how attention-grabbing the opening is (0-10)
    """
    text = moment['text']
    first_10_words = ' '.join(text.split()[:10]).lower()
    score = 5.0  # Base score

    # Universal indicators
    if '?' in first_10_words or '？' in first_10_words:
        score += 2.0

    if re.search(r'\d+', first_10_words):
        score += 1.5

    # Language-specific hook patterns
    hooks = {
        'english': [
            (r'\b(secret|hidden|truth|reality)\b', 3.0),
            (r'\b(never|always|nobody|everyone)\b', 2.5),
            (r'^(why|how|what)', 2.0),
            (r'\b(mistake|wrong|problem)\b', 2.0),
        ],
        'hindi': [
            (r'(रहस्य|सच|वास्तविकता)', 3.0),
            (r'(क्यों|कैसे|क्या)', 2.0),
            (r'(गलती|समस्या|गलत)', 2.0),
        ],
        'spanish': [
            (r'(secreto|verdad|realidad)', 3.0),
            (r'(por qué|cómo|qué)', 2.0),
        ]
    }

    lang_hooks = hooks.get(language, [])
    for pattern, points in lang_hooks:
        if re.search(pattern, first_10_words, re.IGNORECASE):
            score += points
            break  # Only count one strong hook

    return max(0, min(10, score))


def score_standalone_understanding(moment: Dict, language: str = 'english') -> float:
    """
    Score how well a new viewer can understand this (0-10)
    """
    text = moment['text']
    score = 8.0  # Start optimistic (already passed filters)

    # Universal: Complete sentences are good
    sentence_endings = ['.', '!', '?', '।']  # Added Devanagari danda
    if any(text.strip().endswith(end) for end in sentence_endings):
        score += 1.0

    # Universal: Questions and answers are good
    if '?' in text and len(text.split('?')) > 1:
        score += 1.5

    return max(0, min(10, score))


def score_retention_potential(moment: Dict, language: str = 'english') -> float:
    """
    Score likelihood of keeping viewer engaged (0-10)
    """
    text = moment['text']
    duration = moment['duration']
    score = 7.0  # Base retention score

    # Optimal length bonus (30-45s is sweet spot)
    if 30 <= duration <= 45:
        score += 2.0
    elif 45 < duration <= 60:
        score += 1.0

    # Deduct if too long
    if duration > 55:
        score -= 1.0

    # Engagement patterns
    engagement_patterns = [
        r'\b(you|your)\b',  # Direct address
        r'\b(imagine|picture|think about)\b',  # Mental imagery
        r'\?\s*\w+',  # Questions followed by answers
        r'\b(first|second|finally)\b',  # Structure
    ]

    for pattern in engagement_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.5

    # Sentence count (good pacing = 3-5 sentences)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    if 3 <= sentence_count <= 5:
        score += 1.0

    return max(0, min(10, score))


def print_score_summary(moments: List[Dict], top_n: int = 5):
    """
    Print scoring summary for debugging
    """
    print(f"\nTop {top_n} Moments by Score:")
    print("=" * 80)

    for i, moment in enumerate(moments[:top_n], 1):
        print(f"\n#{i} | Score: {moment['score']:.1f}/10")
        print(f"Duration: {moment['duration']:.1f}s")
        print(f"Scores: Context={moment['scores']['context_clarity']:.1f}, "
              f"Hook={moment['scores']['hook_strength']:.1f}, "
              f"Standalone={moment['scores']['standalone']:.1f}, "
              f"Retention={moment['scores']['retention']:.1f}")
        print(f"Text: {moment['text'][:100]}...")
        print("-" * 80)
