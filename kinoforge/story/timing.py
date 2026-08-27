"""Narration-driven timing used before and after voiceover exists."""

import re

from kinoforge.story.audio_modes import uses_native_audio

_WORD_RE = re.compile(r"\b[\w’']+\b", re.UNICODE)
_CJK_RE = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯]")
_PAUSE_RE = re.compile(r"[.!?。！？;:]")

# Reading-speed assumptions, not duration limits. Real TTS duration replaces them as
# soon as voiceover exists.
LATIN_WORDS_PER_SECOND = 2.2
CJK_CHARACTERS_PER_SECOND = 4.0
PAUSE_SECONDS = 0.18
MAX_ESTIMATED_PAUSE_SECONDS = 4.0
MIN_SCENE_SECONDS = 1.0


def narration_seconds(text: object) -> float:
    """Estimate speech length without imposing a target or maximum."""
    value = str(text or "")
    cjk_chars = len(_CJK_RE.findall(value))
    latin_words = len(_WORD_RE.findall(_CJK_RE.sub(" ", value)))
    pauses = len(_PAUSE_RE.findall(value))
    estimate = (
        latin_words / LATIN_WORDS_PER_SECOND
        + cjk_chars / CJK_CHARACTERS_PER_SECOND
        + min(pauses * PAUSE_SECONDS, MAX_ESTIMATED_PAUSE_SECONDS)
    )
    return round(max(MIN_SCENE_SECONDS, estimate), 3)


def scene_seconds(scene: dict) -> float:
    """Measured voiceover duration when available, otherwise narration estimate."""
    if uses_native_audio(scene):
        try:
            clip_duration = float(scene.get("video_seconds") or 0)
        except (TypeError, ValueError):
            clip_duration = 0
        if clip_duration > 0:
            return round(clip_duration, 3)
    try:
        measured = float(scene.get("audio_seconds") or 0)
    except (TypeError, ValueError):
        measured = 0
    return round(measured, 3) if measured > 0 else narration_seconds(scene.get("narration"))
