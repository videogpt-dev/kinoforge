"""Deterministic story quality gate: blocking vs advisory findings."""

from __future__ import annotations

from kinoforge.segments.story.quality import StoryQualityGate


def validate_story(story, ctx):
    return StoryQualityGate(story, ctx).evaluate()


def _clean_story():
    return {
        "logline": "A dog learns to code.",
        "style": "warm cinematic",
        "scenes": [{"narration": "Once upon a time.", "prompt": "a dog at a laptop"}],
    }


def test_clean_story_has_no_findings():
    blocking, advisory = validate_story(_clean_story(), {"scene_count": 1})
    assert blocking == []
    assert advisory == []


def test_silent_scene_is_blocking():
    story = {"logline": "x", "style": "y", "scenes": [{"narration": "  "}, {"narration": "ok"}]}
    blocking, _ = validate_story(story, {})
    assert blocking == ["scenes 1 have no narration"]


def test_missing_logline_and_style_are_advisory():
    story = {"scenes": [{"narration": "hi", "prompt": "a field"}]}
    _, advisory = validate_story(story, {})
    assert "logline is missing" in advisory
    assert "visual style is missing" in advisory


def test_scene_count_mismatch_is_advisory():
    story = _clean_story()
    _, advisory = validate_story(story, {"scene_count": 3})
    assert "expected 3 scenes, got 1" in advisory


def test_visible_text_prompt_is_advisory():
    story = _clean_story()
    story["scenes"][0]["prompt"] = 'a poster with a logo'
    _, advisory = validate_story(story, {"scene_count": 1})
    assert any("may render on-screen text" in a for a in advisory)


def test_quoted_text_prompt_is_advisory():
    story = _clean_story()
    story["scenes"][0]["prompt"] = 'a sign saying "OPEN"'
    _, advisory = validate_story(story, {"scene_count": 1})
    assert any("may render on-screen text" in a for a in advisory)


def test_require_motion_without_any_motion_is_advisory():
    story = _clean_story()
    _, advisory = validate_story(story, {"scene_count": 1, "require_motion": True})
    assert "Video Mode set but no scene is marked for motion" in advisory


def test_require_motion_satisfied_when_a_scene_has_motion():
    story = _clean_story()
    story["scenes"][0]["motion"] = True
    _, advisory = validate_story(story, {"scene_count": 1, "require_motion": True})
    assert all("motion" not in a for a in advisory)
