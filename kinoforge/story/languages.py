"""Narration languages and caption-alignment codes.

Shared by the whole story engine: generation writes narration in one of these,
the translation pass reshapes it into another, and caption alignment needs the
Whisper ISO code. Kept in one place so the table has a single home.
"""

from enum import StrEnum
from typing import Optional


class NarrationLanguage(StrEnum):
    AUTO = ""
    ENGLISH = "en"
    HINDI = "hi"
    PUNJABI = "pa"
    ITALIAN = "it"
    GERMAN = "de"
    SPANISH = "es"
    FRENCH = "fr"
    PORTUGUESE = "pt"
    HINGLISH = "hinglish"
    PUNGLISH = "punglish"


# Supported narration languages: key -> (prompt phrasing, Whisper ISO code for
# caption alignment). "" = auto (match the title). Hinglish/Punglish are code-mixed
# styles, so the phrasing describes them; Whisper aligns on the base language.
LANGUAGES: dict = {
    NarrationLanguage.ENGLISH.value: ("English", "en"),
    NarrationLanguage.HINDI.value: ("Hindi", "hi"),
    NarrationLanguage.PUNJABI.value: ("Punjabi", "pa"),
    NarrationLanguage.ITALIAN.value: ("Italian", "it"),
    NarrationLanguage.GERMAN.value: ("German", "de"),
    NarrationLanguage.SPANISH.value: ("Spanish", "es"),
    NarrationLanguage.FRENCH.value: ("French", "fr"),
    NarrationLanguage.PORTUGUESE.value: ("Portuguese", "pt"),
    NarrationLanguage.HINGLISH.value: (
        "Hinglish, a natural, modern mix of Hindi and English in Roman/Latin"
        " script, as spoken by young urban Indians",
        "hi",
    ),
    NarrationLanguage.PUNGLISH.value: (
        "Punglish, a natural, modern mix of Punjabi and English in Roman/Latin script",
        "pa",
    ),
}


def whisper_code(language: str) -> Optional[str]:
    """Whisper ISO code for a narration language key (None = auto-detect)."""
    entry = LANGUAGES.get((language or "").strip())
    return entry[1] if entry else None


def language_name(language: str) -> str:
    """The phrasing a language rule names: the language's own name ("English"), or the
    fallback phrase when none is set."""
    entry = LANGUAGES.get((language or "").strip())
    return entry[0] if entry else "the same language as the title"
