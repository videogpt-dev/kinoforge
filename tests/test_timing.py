"""Narration-driven timing: reading-speed estimate and measured-duration precedence."""

from __future__ import annotations

from kinoforge.segments.story.timing import NarrationTimer

MIN_SCENE_SECONDS = NarrationTimer.MIN_SCENE_SECONDS
narration_seconds = NarrationTimer.narration_seconds
scene_seconds = NarrationTimer.scene_seconds


def test_narration_floors_at_min_scene_seconds():
    assert narration_seconds("hi") == MIN_SCENE_SECONDS
    assert narration_seconds("") == MIN_SCENE_SECONDS
    assert narration_seconds(None) == MIN_SCENE_SECONDS


def test_narration_counts_latin_words_at_rate():
    # 5 words / 2.2 words per second.
    assert narration_seconds("one two three four five") == round(5 / 2.2, 3)


def test_narration_counts_cjk_characters_at_rate():
    # 20 CJK chars / 4.0 chars per second = 5.0s.
    assert narration_seconds("阿" * 20) == 5.0


def test_narration_pause_contribution_is_capped():
    # Pure pauses: 100 * 0.18 = 18s, capped to 4.0s.
    assert narration_seconds("." * 100) == 4.0


def test_scene_seconds_prefers_native_video_duration():
    scene = {"audio_mode": "native", "video_seconds": 12.34, "audio_seconds": 99}
    assert scene_seconds(scene) == 12.34


def test_scene_seconds_native_without_video_falls_through_to_audio():
    scene = {"audio_mode": "native", "video_seconds": 0, "audio_seconds": 7.5}
    assert scene_seconds(scene) == 7.5


def test_scene_seconds_uses_measured_audio_when_present():
    assert scene_seconds({"audio_seconds": 6.25, "narration": "ignored"}) == 6.25


def test_scene_seconds_falls_back_to_narration_estimate():
    scene = {"audio_seconds": 0, "narration": "one two three four five"}
    assert scene_seconds(scene) == round(5 / 2.2, 3)


def test_scene_seconds_ignores_non_numeric_measured_values():
    scene = {"audio_seconds": "oops", "narration": "hi"}
    assert scene_seconds(scene) == MIN_SCENE_SECONDS
