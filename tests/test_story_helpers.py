"""Pure story helpers: dash stripping, plan normalizing, and operation utilities."""

from __future__ import annotations

from kinoforge.segments.story.agent import strip_dashes
from kinoforge.segments.story.operations import StoryOperations
from kinoforge.segments.story.run_agent import (
    _normalize,
    _not_cancelled,
    _stage_temperature,
)


# --- strip_dashes ---------------------------------------------------------

def test_strip_dashes_replaces_em_en_bar_with_comma():
    assert strip_dashes("a — b") == "a, b"
    assert strip_dashes("a—b") == "a, b"
    assert strip_dashes("x") == "x"


def test_strip_dashes_leaves_plain_hyphen():
    assert strip_dashes("a - b") == "a - b"


# --- run_agent helpers ----------------------------------------------------

def test_not_cancelled_default():
    assert _not_cancelled() is False


def test_stage_temperature_reads_value_or_defaults():
    assert _stage_temperature({"temperature": 0.3}) == 0.3
    assert _stage_temperature({}) == 0.85
    assert _stage_temperature({"temperature": "nan-ish"}) == 0.85


def test_normalize_drops_promptless_scenes_and_coerces_motion():
    raw = {
        "logline": "A hero — reborn",
        "style": "noir",
        "characters": [{"name": "A", "description": "d"}, {"foo": 1}],
        "scenes": [
            {"prompt": "p", "narration": "n", "motion": 1},
            {"narration": "no prompt here"},
        ],
    }
    out = _normalize(raw)
    assert out["logline"] == "A hero, reborn"  # dash stripped
    assert out["characters"] == [{"name": "A", "description": "d"}]  # empty char dropped
    assert len(out["scenes"]) == 1
    assert out["scenes"][0]["motion"] is True


# --- StoryOperations static helpers ---------------------------------------

def test_clean_strips_dashes_and_whitespace():
    assert StoryOperations._clean("  a — b ") == "a, b"


def test_story_summary_lists_characters_or_none():
    summary = StoryOperations._story_summary(
        {"logline": "L", "style": "S", "characters": [{"name": "A", "description": "d"}]}
    )
    assert "LOGLINE: L" in summary
    assert "A: d" in summary
    assert "CHARACTERS: none" in StoryOperations._story_summary({})


def test_clean_suggestion_extracts_trailing_quote_and_strips_label():
    assert StoryOperations._clean_suggestion("title: Hello", False) == "Hello"
    long = 'Here is my suggestion for you today: "The Real Title"'
    assert StoryOperations._clean_suggestion(long, False) == "The Real Title"


def test_clean_suggestion_single_line_picks_first_substantial_line():
    assert StoryOperations._clean_suggestion("\n\nFirst line\nsecond", True) == "First line"


def test_plausible_suggestion_rules():
    assert StoryOperations._plausible_suggestion("A Great Title", "title") is True
    assert StoryOperations._plausible_suggestion("no", "title") is False
    assert StoryOperations._plausible_suggestion("Name: Bob", "title") is False
    assert StoryOperations._plausible_suggestion("I cannot help with that", "title") is False
    assert StoryOperations._plausible_suggestion("short", "description") is False
    assert StoryOperations._plausible_suggestion("a long enough description here", "description") is True
