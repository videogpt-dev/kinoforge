"""Standalone-clip rejection rules and the filter that applies them (MomentFilter)."""

from __future__ import annotations

from kinoforge.observ import bind, build_logger, reset
from kinoforge.segments.clips.moments.filter import MomentFilter

EN = MomentFilter("english")


# --- window_text ----------------------------------------------------------

def test_window_collects_overlapping_segments():
    transcript = [
        {"start": 0, "end": 2, "text": "hello"},
        {"start": 2, "end": 4, "text": "world"},
        {"start": 10, "end": 12, "text": "later"},
    ]
    assert MomentFilter.window_text(transcript, 1, 3) == "hello world"


def test_window_empty_when_no_transcript_or_bad_range():
    assert MomentFilter.window_text([], 0, 5) == ""
    assert MomentFilter.window_text([{"start": 0, "end": 2, "text": "x"}], 5, 5) == ""


# --- has_clear_topic ------------------------------------------------------

def test_clear_topic_accepts_question_number_pattern_and_substance():
    assert EN.has_clear_topic("Is this real?") is True
    assert EN.has_clear_topic("5 ways to save") is True
    assert EN.has_clear_topic("why does the sky glow") is True
    assert EN.has_clear_topic("the clever dog solved puzzles") is True


def test_clear_topic_rejects_short_or_all_filler():
    assert EN.has_clear_topic("hi") is False
    assert EN.has_clear_topic("and the a is") is False


# --- starts_mid_thought ---------------------------------------------------

def test_starts_mid_thought_catches_obvious_openers():
    assert EN.starts_mid_thought("so i was thinking") is True
    assert EN.starts_mid_thought("because of that reason") is True
    assert EN.starts_mid_thought("Welcome everyone today") is False


# --- has_unclear_pronouns -------------------------------------------------

def test_unclear_pronoun_early_is_flagged():
    assert EN.has_unclear_pronouns("this changes everything") is True


def test_pronoun_with_earlier_referent_is_ok():
    assert EN.has_unclear_pronouns("the dog ran because it tripped") is False


# --- is_explanation_without_question --------------------------------------

def test_bare_explanation_is_flagged():
    assert EN.is_explanation_without_question("because it rained all day") is True


def test_question_or_problem_keyword_is_not_bare():
    assert EN.is_explanation_without_question("why did it rain?") is False
    assert EN.is_explanation_without_question("the secret to focus") is False


# --- requires_context / podcast / branding --------------------------------

def test_requires_context_detects_backreference():
    assert EN.requires_context("remember when we met") is True
    assert EN.requires_context("the sky is blue") is False


def test_podcast_dependency_detected():
    assert EN.has_podcast_dependency("my guest is here") is True
    assert EN.has_podcast_dependency("here is a fact") is False


def test_branding_before_insight_detected():
    assert EN.has_branding_before_insight("subscribe to my channel now") is True
    assert EN.has_branding_before_insight("the truth about money") is False


# --- MomentFilter.run integration -----------------------------------------

def test_filter_empty_returns_empty():
    assert MomentFilter.run([], []) == []


def test_filter_keeps_standalone_rejects_dependent():
    token = bind(build_logger(segment="clips"))
    try:
        transcript = [
            {"start": 0, "end": 2, "text": "5 ways to save money and build wealth today"},
            {"start": 10, "end": 12, "text": "so i was saying"},
        ]
        good = {"start": 0, "text": "5 ways to save money and build wealth today"}
        bad = {"start": 10, "text": "so i was saying"}
        kept = MomentFilter.run([good, bad], transcript)
    finally:
        reset(token)
    assert kept == [good]
