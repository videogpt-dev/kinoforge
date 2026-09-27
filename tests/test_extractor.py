"""Transcript-boundary snapping and small extractor utilities."""

from __future__ import annotations

from kinoforge.segments.clips.moments.extractor import TranscriptText

detect_language = TranscriptText.detect_language
format_time = TranscriptText.format_time
get_text_between_times = TranscriptText.text_between
snap_to_transcript = TranscriptText.snap


def _segs():
    return [
        {"start": 0, "end": 5, "text": "a"},
        {"start": 5, "end": 10, "text": "b"},
        {"start": 10, "end": 15, "text": "c"},
    ]


# --- snap_to_transcript ---------------------------------------------------

def test_snap_empty_transcript_returns_input():
    assert snap_to_transcript([], 3.0, 7.0) == (3.0, 7.0, "")


def test_snap_lands_on_segment_edges():
    assert snap_to_transcript(_segs(), 6.0, 9.0) == (5.0, 10.0, "b")


def test_snap_extends_short_pick_to_min_length():
    start, end, text = snap_to_transcript(_segs(), 6.0, 7.0, min_len=8.0)
    assert (start, end) == (5.0, 15.0)
    assert text == "b c"


def test_snap_trims_long_pick_to_max_length():
    start, end, text = snap_to_transcript(_segs(), 0.0, 15.0, max_len=6.0)
    assert (start, end) == (0.0, 5.0)
    assert text == "a"


# --- get_text_between_times -----------------------------------------------

def test_text_between_times_only_fully_contained():
    text = get_text_between_times(_segs(), 4.0, 11.0)
    assert text == "b"  # only [5,10] fully inside [4,11]


# --- format_time ----------------------------------------------------------

def test_format_time_mm_ss():
    assert format_time(5) == "00:05"
    assert format_time(65) == "01:05"
    assert format_time(600) == "10:00"


# --- detect_language ------------------------------------------------------

def test_detect_language_by_script():
    assert detect_language("नमस्ते दुनिया") == "hindi"
    assert detect_language("你好世界") == "chinese"
    assert detect_language("مرحبا بالعالم") == "arabic"
    assert detect_language("hello world") == "english"


def test_detect_language_uses_dominant_script_not_stray_char():
    # A mostly-Hindi clip with one stray Arabic glyph must stay hindi, not flip to arabic.
    assert detect_language("नमस्ते दोस्तों आज हम बात करेंगे ؛ विषय पर") == "hindi"
    # Empty / punctuation-only text is english, never a non-Latin script.
    assert detect_language("") == "english"
    assert detect_language("... --- 123") == "english"
