"""Regression tests for Story segment bugs found in review: each failed before its fix."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kinoforge.segments.story.media.ceiling import PromptCeiling
from kinoforge.segments.story.media.images import ImagePainter
from kinoforge.segments.story.media.prompt_text import PromptText
from kinoforge.segments.story.operations import StoryOperations
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.run_agent import RunAgent, StoryShape
from kinoforge.service.runtimes.text_ports import InfrelayTextPorts

_STAGE = {"format": "story_json", "role": "writer"}
_CTX = {"title": "T", "description": "", "output_format": "vertical", "language_rule": ""}


def _ports(reply: str) -> StoryPorts:
    def complete(purpose, system, user, *, route, temperature, max_tokens):
        return reply, {"input_tokens": 1, "output_tokens": 1, "model": "m"}

    return StoryPorts(
        complete=complete, complete_json=None, parse_json=json.loads,
        prompt=lambda key, **_: f"<{key}>", contract=lambda _n: "C",
        language_rule=lambda *_: "", story_budget=lambda _o=None: 100,
    )


def _story(count: int) -> str:
    scenes = [{"prompt": f"shot {i}", "narration": f"line {i}"} for i in range(count)]
    return json.dumps({"logline": "l", "style": "s", "characters": [], "scenes": scenes})


def test_extra_scenes_are_trimmed_to_the_requested_count():
    ctx = {**_CTX, "scene_count": 8}
    result = RunAgent(_ports(_story(40)), {"stages": [_STAGE]}, ctx).run()
    assert result["ok"] is True
    assert len(result["story"]["scenes"]) == 8
    assert result["story"]["scenes"][-1]["prompt"] == "shot 7"
    assert any("40" in warning for warning in result["warnings"])


def test_fewer_scenes_are_kept_with_a_warning():
    ctx = {**_CTX, "scene_count": 8}
    result = RunAgent(_ports(_story(5)), {"stages": [_STAGE]}, ctx).run()
    assert len(result["story"]["scenes"]) == 5
    assert "expected 8 scenes, got 5" in result["warnings"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("false", False), ("False", False), ("no", False),
     ("0", False), ("", False), (None, False), ("true", True), ("yes", True), (1, True),
     (0, False)],
)
def test_motion_flag_parses_text_booleans(raw, expected):
    story = StoryShape.normalize({"scenes": [{"prompt": "p", "narration": "n", "motion": raw}]})
    assert story["scenes"][0]["motion"] is expected


def test_empty_job_id_is_not_a_crash(monkeypatch):
    from kinoforge.api.app import app
    from kinoforge.service.runtimes import StoryRuntime

    seen = {}

    def write(self, request, is_cancelled=None):
        seen["cancelled"] = is_cancelled()
        return {"result": {"ok": True}, "meter_events": [], "logs": []}

    monkeypatch.setattr(StoryRuntime, "from_env", classmethod(lambda cls: cls(None)))
    monkeypatch.setattr(StoryRuntime, "write", write)
    body = {
        "job_id": "", "context": {"title": "T"}, "agent": {"stages": [_STAGE]},
        "definitions": {
            "schema_version": 1, "id": "b", "version": "1", "engine": {"minimum": "0.1.0"},
        },
    }
    response = TestClient(app).post("/v1/segments/story/write", json=body)
    assert response.status_code == 200, response.text
    assert seen["cancelled"] is False


def test_music_preset_with_a_stray_dollar_still_renders():
    from string import Template

    from kinoforge.service.runtimes.media import MediaUtils

    assert MediaUtils.music_prompt("Upbeat $mood bed, budget $5 synths", "calm") == (
        "Upbeat calm bed, budget $5 synths"
    )
    assert Template("x $mood").substitute({"mood": "a"}) == "x a"


def _painter() -> ImagePainter:
    return ImagePainter(
        generate_image=None, image_scene=lambda *a, **k: "",
        image_character=lambda who, style, key=None: who,
        image_negative=lambda: "", ceiling=PromptCeiling(),
    )


def test_character_without_a_name_does_not_crash_the_sheet():
    story = {"characters": [{"description": "a brass robot"}, {"name": "Ana", "description": "pilot"}]}
    prompt = _painter().character_prompt(story, {})
    assert "a brass robot" in prompt and "Ana: pilot" in prompt


@pytest.mark.parametrize(
    ("name", "text", "mentioned"),
    [
        ("Max", "zoom to maximum resolution", False),
        ("Max", "Max opens the door", True),
        ("Ann", "planning the heist", False),
        ("Ann", "Ann's hands shake", True),
        ("The Narrator", "the wind rises over the hill", False),
        ("The Narrator", "the Narrator steps forward", True),
        ("Dr. Elena Vos", "Vos checks the readings", True),
        ("Captain Reyes", "the captain shouts", True),
    ],
)
def test_character_mentions_use_whole_words(name, text, mentioned):
    found = PromptText.mentioned_characters([{"name": name}], text)
    assert bool(found) is mentioned


def test_rewrite_uses_the_same_voice_as_writing():
    captured = {}

    def complete_json(purpose, system, user, *, route, temperature, max_tokens):
        captured["system"] = system
        return {"prompt": "p", "narration": "n"}

    ports = StoryPorts(
        complete=None, complete_json=complete_json, parse_json=json.loads,
        prompt=lambda key, **_: f"<{key}>", contract=lambda _n: "",
        language_rule=lambda *_: "", story_budget=lambda _o=None: 0,
    )
    legacy_clone = {"stages": [{"format": "story_json", "system": "old per-stage persona"}]}
    record = {"story": {"scenes": [{"prompt": "a", "narration": "b"}]}}
    StoryOperations(ports).rewrite_scene(record, 0, legacy_clone)
    assert captured["system"] == "<story_system>"
    StoryOperations(ports).rewrite_scene(record, 0, {"persona": "the agent voice"})
    assert captured["system"] == "the agent voice"


@pytest.mark.parametrize(
    ("ceiling", "stage", "expected"),
    [(0, 4000, 4000), (0, None, 0), (8000, 4000, 4000), (8000, 20000, 8000), (8000, None, 8000)],
)
def test_story_budget_without_a_ceiling_keeps_the_stage_budget(ceiling, stage, expected):
    ports = InfrelayTextPorts(None, None, budgets={"story": ceiling}, label="story")
    assert ports.story_budget(stage) == expected


def test_a_gateway_failure_is_a_502_with_its_reason(monkeypatch):
    from kinoforge.api.app import app
    from kinoforge.inference.infrelay import InfrelayError
    from kinoforge.service.runtimes.media import MediaRuntime

    def render_voice(self, request):
        raise InfrelayError("infrelay audio/edge failed (429): busy")

    monkeypatch.setattr(MediaRuntime, "from_env", classmethod(lambda cls: object.__new__(cls)))
    monkeypatch.setattr(MediaRuntime, "render_voice", render_voice)
    body = {"job_id": "j", "text": "hi", "pick": {"provider": "edge", "model": ""}}
    response = TestClient(app).post("/v1/segments/story/media/voice", json=body)
    assert response.status_code == 502
    assert "busy" in response.json()["detail"]
