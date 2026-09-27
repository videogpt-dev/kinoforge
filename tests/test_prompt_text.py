"""Deterministic media-prompt helpers: sentence trim, labeled join/fit, cast mentions."""

from __future__ import annotations

from kinoforge.segments.story.media.prompt_text import LabeledPrompt, PromptText

leading_sentences = PromptText.leading_sentences
mentioned_characters = PromptText.mentioned_characters


def join_labeled(parts):
    return LabeledPrompt(parts).join()


def fit_labeled(parts, max_chars):
    return LabeledPrompt(parts).fit(max_chars)


# --- leading_sentences ----------------------------------------------------

def test_leading_sentences_keeps_first_n():
    assert leading_sentences("A cat. A dog. A bird.", 2) == "A cat. A dog."


def test_leading_sentences_empty():
    assert leading_sentences("", 3) == ""
    assert leading_sentences(None, 3) == ""


# --- join_labeled ---------------------------------------------------------

def test_join_labeled_skips_empty_and_terminates():
    out = join_labeled([("Style", "warm"), ("", ""), ("Scene", "a dog")])
    assert out == "Style: warm. Scene: a dog."


def test_join_labeled_unlabeled_section():
    assert join_labeled([("", "just text")]) == "just text."


# --- fit_labeled ----------------------------------------------------------

def test_fit_labeled_keeps_everything_under_budget():
    assert fit_labeled([("", "hello world foo")], 100) == "hello world foo."


def test_fit_labeled_clips_at_word_boundary():
    out = fit_labeled([("", "one two three four five")], 12)
    assert out == "one two."
    assert len(out) <= 12


# --- mentioned_characters -------------------------------------------------

def test_mentions_by_first_name():
    chars = [{"name": "Alice"}, {"name": "Bob"}]
    assert mentioned_characters(chars, "alice walks in") == [{"name": "Alice"}]


def test_mentions_by_role_after_dash():
    chars = [{"name": "Bob — the villain"}]
    assert mentioned_characters(chars, "the villain arrives") == [{"name": "Bob — the villain"}]


def test_short_aliases_below_three_chars_ignored():
    assert mentioned_characters([{"name": "Jo"}], "jo waves") == []


def test_unnamed_characters_skipped():
    assert mentioned_characters([{"name": "  "}], "anything") == []
