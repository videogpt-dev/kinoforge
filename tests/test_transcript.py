"""Transcript snapping, candidate windows, language guess."""

from __future__ import annotations

from kinoforge.segments.clips.moments.transcript import TranscriptText

detect_language = TranscriptText.detect_language
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


# --- windows --------------------------------------------------------------

def _speech(n):
    return [{"start": i * 5.0, "end": i * 5.0 + 5, "text": "one two three four five"}
            for i in range(n)]


def test_windows_are_non_overlapping_and_sized():
    windows = TranscriptText.windows(_speech(12), 10, 20)
    assert [(w["start"], w["end"]) for w in windows][:2] == [(0.0, 10.0), (10.0, 20.0)]
    assert all(10 <= w["end"] - w["start"] <= 20 for w in windows)


def test_windows_skip_short_text():
    short = [{"start": 0, "end": 12, "text": "hi"}]
    assert TranscriptText.windows(short, 10, 20) == []


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
