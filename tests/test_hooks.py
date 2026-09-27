"""HookDetector: opening-second hook classification and scoring."""

from __future__ import annotations

from kinoforge.segments.clips.signals.hooks import HookDetector


def _seg(text, start=0.0, end=3.0):
    return [{"start": start, "end": end, "text": text}]


def _hook(text):
    return HookDetector().analyze(_seg(text), 0.0, 3.0)


def test_no_text_in_opening_returns_none_hook():
    signal = HookDetector().analyze([], 0.0, 3.0)
    assert signal.hook_type == "none"
    assert signal.strength == 0.0


def test_question_mark_is_strongest():
    signal = _hook("Really?")
    assert signal.hook_type == "question"
    assert signal.strength == 10.0
    assert signal.confidence == 1.0


def test_question_word_start_lower_confidence():
    signal = _hook("why do cats purr")
    assert signal.hook_type == "question"
    assert signal.confidence == 0.85


def test_surprising_word_detected():
    assert _hook("actually it works").hook_type == "surprising"


def test_numeric_percentage_is_data():
    assert _hook("50% of people agree").hook_type == "data"


def test_cta_phrase_detected():
    assert _hook("check this out folks").hook_type == "cta"


def test_emotional_trigger_detected():
    assert _hook("i love this idea").hook_type == "emotional"


def test_urgency_language_detected():
    assert _hook("buy before it ends").hook_type == "urgency"


def test_vague_start_applies_penalty():
    signal = _hook("so actually this matters")
    assert signal.hook_type == "surprising"
    assert abs(signal.strength - 9.0 * 0.8) < 1e-9  # strong (9.0) * vague penalty (0.8)
    assert any("vague start" in r for r in signal.reasons)


def test_no_pattern_returns_none_with_preview():
    signal = _hook("the quiet afternoon passed slowly")
    assert signal.hook_type == "none"
    assert signal.text  # keeps a preview of the opening text


def test_opening_text_joins_overlapping_segments():
    transcript = [
        {"start": 0, "end": 1.5, "text": "hello"},
        {"start": 1.5, "end": 3, "text": "there"},
        {"start": 9, "end": 10, "text": "later"},
    ]
    assert HookDetector._opening_text(transcript, 0.0, 3.0) == "hello there"
