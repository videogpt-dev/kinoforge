"""
Aggressive moment filtering - reject unclear clips
THIS IS THE MOST CRITICAL MODULE
"""

import re
from typing import Any, Dict, List


def filter_moments_aggressively(
        candidates: List[Dict],
        transcript: List[Dict]
) -> List[Dict]:
    """
    Apply aggressive rejection rules to candidate moments

    REJECT if ANY condition is true:
    1. First 2 seconds don't state clear topic/problem
    2. Starts mid-thought
    3. Pronouns without clear reference
    4. Explanation without question/problem
    5. Context dependency
    6. Requires podcast knowledge
    7. Branding before insight

    Args:
        candidates: List of candidate moments
        transcript: Full transcript for context

    Returns:
        Filtered moments (ONLY high-quality)
    """
    if not candidates:
        return []

    # Get language from first candidate
    language = candidates[0].get('language', 'english')
    print(f"  Filtering for language: {language}")

    filtered = []
    rejection_log: List[Dict[str, Any]] = []

    for idx, moment in enumerate(candidates):
        text = moment['text']
        rejection_reasons = []

        # Get first 2 seconds of text (with fallback if empty)
        first_2s_text = get_text_in_time_window(
            transcript,
            moment['start'],
            moment['start'] + 2
        )

        # Fallback: if first 2s is empty, extend to 4s to capture speech
        if not first_2s_text:
            first_2s_text = get_text_in_time_window(
                transcript,
                moment['start'],
                moment['start'] + 4
            )

        # RULE 1: First 2 seconds must state topic/problem
        if not has_clear_topic_or_hook(first_2s_text, language):
            rejection_reasons.append("No clear topic/problem in first 2s, Requires external context")

        # RULE 2: Must NOT start mid-thought (language-specific)
        if starts_mid_thought(text, language):
            rejection_reasons.append("Starts mid-thought")

        # RULE 3: Pronouns without reference (mostly for English)
        if language == 'english' and has_unclear_pronouns(first_2s_text):
            rejection_reasons.append("Unclear pronouns without reference")

        # RULE 4: Explanation without question (universal)
        if is_explanation_without_question(text, language):
            rejection_reasons.append("Explanation without stated question/problem")

        # RULE 5: Context dependency indicators
        if requires_context(text, language):
            rejection_reasons.append("Requires external context")

        # RULE 6: Podcast-specific references
        if has_podcast_context_dependency(text, language):
            rejection_reasons.append("Requires podcast context")

        # RULE 7: Branding before value
        if has_branding_before_insight(text, language):
            rejection_reasons.append("Branding appears before insight")

        # If NO rejection reasons, keep it
        if not rejection_reasons:
            filtered.append(moment)
        else:
            rejection_log.append({
                'moment_id': idx,
                'reasons': rejection_reasons,
                'text_preview': text[:100]
            })

    # Print rejection summary
    print(f"  Rejected {len(rejection_log)}/{len(candidates)} moments:")
    for log in rejection_log[:5]:  # Show first 5
        print(f"    - {', '.join(log['reasons'])}")
    if len(rejection_log) > 5:
        print(f"    ... and {len(rejection_log) - 5} more")

    return filtered


def get_text_in_time_window(
        transcript: List[Dict],
        start_time: float,
        end_time: float
) -> str:
    """Extract text from segments that OVERLAP the time window (not just fully contained)"""
    if not transcript or start_time >= end_time:
        return ""

    text_parts = []
    for segment in transcript:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)

        # Include if segment overlaps window: seg_start < window_end AND seg_end > window_start
        if seg_start < end_time and seg_end > start_time:
            text_parts.append(segment.get('text', ''))

    return ' '.join(text_parts).strip()


def has_clear_topic_or_hook(text: str, language: str = 'english') -> bool:
    """
    Check if text states a clear topic, question, or problem
    Accepts questions, numbers, specific patterns, or substantive statements
    """
    if len(text.strip()) < 5:
        return False

    # Universal indicators - always accept
    if '?' in text or '？' in text or bool(re.search(r'\d+', text)):
        return True

    # Language-specific patterns
    patterns = {
        'english': [
            r'^\s*(why|how|what|when|where|who)',
            r'^\s*(do you know|have you ever|did you know)',
            r'^\s*the (secret|truth|reality|key|problem|issue|thing) (is|to|about)',
            r'^\s*(here\'s|let me (tell|show|explain))',
            r'^\s*(\d+\s+(ways|reasons|things|tips))',
            r'\b(the (secret|truth|reality|key|problem|issue) (is|of|to))\b',
            r'\b(actually|really|surprisingly|interestingly|basically)\s+',
            r'\b(one of the|the most|the best|the worst)\b',
        ],
        'hindi': [
            r'(क्यों|कैसे|क्या|कब|कहाँ|कौन)',
            r'(रहस्य|सच|वास्तविकता)',
            r'\d+\s*(तरीके|कारण|टिप्स)',
        ],
        'spanish': [
            r'(por qué|cómo|qué|cuándo|dónde)',
            r'(secreto|verdad|realidad)',
        ]
    }

    lang_patterns = patterns.get(language, [])
    for pattern in lang_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Fallback: Accept if substantive (3+ meaningful words, 10+ chars)
    words = text.lower().split()
    if len(text.strip()) >= 10 and len(words) >= 3:
        filler = {'and', 'the', 'a', 'or', 'is', 'are', 'was', 'were', 'this', 'that', 'it', 'be'}
        meaningful = [w for w in words if w not in filler and len(w) > 2]
        if len(meaningful) >= 2:
            return True

    return False


def starts_mid_thought(text: str, language: str = 'english') -> bool:
    """
    Detect if clip starts mid-thought - only catch OBVIOUS cases
    Be conservative to avoid false positives
    """
    patterns = {
        'english': [
            r'^\s*so\s+(i|we|he|she|they|you)',
            r'^\s*because',
            r'^\s*as\s+i\s+(said|mentioned)',
            r'^\s*going back to',
        ],
        'hindi': [
            r'^\s*(तो|क्योंकि)',
        ],
        'spanish': [
            r'^\s*(entonces|porque)',
        ]
    }

    lang_patterns = patterns.get(language, [])
    for pattern in lang_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def has_unclear_pronouns(text: str) -> bool:
    """
    Check for pronouns without clear antecedents in first sentence
    """
    first_sentence = text.split('.')[0] if '.' in text else text
    words = first_sentence.lower().split()[:10]  # First 10 words

    # Problematic pronouns
    unclear_pronouns = ['this', 'that', 'it', 'they', 'them', 'these', 'those']

    for pronoun in unclear_pronouns:
        if pronoun in words:
            # Check if there's a clear referent before it
            pronoun_idx = words.index(pronoun)
            if pronoun_idx < 3:  # Pronoun appears too early
                return True

    return False


def is_explanation_without_question(text: str, language: str = 'english') -> bool:
    """
    Detect if this is ONLY bare explanation with no substance
    Accept if it has problem-solving or value statements
    """
    # Accept any statement with question mark or problem keywords
    if '?' in text or '？' in text:
        return False

    problem_keywords = {
        'english': ['why', 'how', 'what', 'problem', 'reason', 'secret', 'truth', 'solution', 'key', 'solution', 'mistake'],
        'hindi': ['क्यों', 'कैसे', 'समस्या', 'कारण', 'समाधान'],
        'spanish': ['por qué', 'cómo', 'problema', 'razón', 'solución']
    }

    keywords = problem_keywords.get(language, [])
    if any(kw in text.lower() for kw in keywords):
        return False

    # Only reject pure bare explanations (start with because/since/etc with nothing else)
    bare_pattern = r'^(because|since|due to|as a result|therefore|thus|so|hence)\s+'
    return bool(re.match(bare_pattern, text.lower().strip()))


def requires_context(text: str, language: str = 'english') -> bool:
    """
    Detect if clip requires external context to understand
    """
    patterns = {
        'english': [
            r'\b(remember when|as (i|we) said|earlier|previously)\b',
            r'\b(in (this|that) (video|episode|podcast))\b',
            r'\b(like i mentioned|as discussed)\b',
            r'\b(the other day|last (week|time))\b',
        ],
        'hindi': [
            r'\b(याद है|जैसा मैंने कहा|पहले|पिछले)\b',
            r'\b(इस (वीडियो|एपिसोड|पॉडकास्ट) में)\b',
        ],
        'spanish': [
            r'\b(recuerda cuando|como (yo|nosotros) dijimos|antes|previamente)\b',
            r'\b(en (este|ese) (video|episodio|podcast))\b',
        ]
    }

    lang_patterns = patterns.get(language, [])
    for pattern in lang_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def has_podcast_context_dependency(text: str, language: str = 'english') -> bool:
    """
    Check for podcast-specific references
    """
    patterns = {
        'english': [
            r'\b(on (this|the) (show|podcast|episode))\b',
            r'\b(my guest|our guest|the guest)\b',
            r'\b(we\'re talking (about|with))\b',
            r'\b(thanks for (having|joining))\b',
        ],
        'hindi': [
            r'\b(इस (शो|पॉडकास्ट|एपिसोड) पर)\b',
            r'\b(मेरे अतिथि|हमारे अतिथि)\b',
        ],
        'spanish': [
            r'\b(en (este|el) (show|podcast|episodio))\b',
            r'\b(mi invitado|nuestro invitado)\b',
        ]
    }

    lang_patterns = patterns.get(language, [])
    for pattern in lang_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def has_branding_before_insight(text: str, language: str = 'english') -> bool:
    """
    Detect if branding/CTA appears before value
    """
    first_sentence = text.split('.')[0] if '.' in text else text[:100]

    patterns = {
        'english': [
            r'\b(subscribe|like|comment|follow|check out)\b',
            r'\b(my (channel|podcast|show|course))\b',
            r'\b(link in (bio|description))\b',
        ],
        'hindi': [
            r'\b(सब्सक्राइब|लाइक|कमेंट|फॉलो)\b',
            r'\b(मेरे (चैनल|पॉडकास्ट|शो))\b',
        ],
        'spanish': [
            r'\b(suscríbete|like|comenta|sigue)\b',
            r'\b(mi (canal|podcast|show))\b',
        ]
    }

    lang_patterns = patterns.get(language, [])
    for pattern in lang_patterns:
        if re.search(pattern, first_sentence, re.IGNORECASE):
            return True

    return False
