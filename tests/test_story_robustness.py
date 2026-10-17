"""Story runner robustness: retries, temperatures, usage on failure, and request checks."""

from __future__ import annotations

import json

import pytest

from kinoforge.contract import MeterAction
from kinoforge.segments.story.operations import StoryOperations, Suggestion
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.run_agent import JSON_TEMPERATURE, TEXT_TEMPERATURE, RunAgent

_CTX = {"title": "T", "description": "", "output_format": "v", "language_rule": "",
        "scene_count": 1}
_STORY = json.dumps({"logline": "l", "style": "s", "characters": [],
                     "scenes": [{"prompt": "p", "narration": "n"}]})


class _Fake:
    def __init__(self, replies, usage=None):
        self.replies = list(replies)
        self.temperatures: list[float] = []
        self.prompts: list[str] = []
        self.meters: list[tuple] = []
        self.usage = {"input_tokens": 2, "output_tokens": 3, "model": "m"} if usage is None else usage

    def complete(self, purpose, system, user, *, route, temperature, max_tokens):
        self.temperatures.append(temperature)
        self.prompts.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, dict(self.usage)

    def ports(self) -> StoryPorts:
        return StoryPorts(
            complete=self.complete, complete_json=None, parse_json=json.loads,
            prompt=lambda key, **_: f"<{key}>", contract=lambda _n: "C",
            language_rule=lambda *_: "", story_budget=lambda _o=None: 100,
            meter=lambda action, qty, variant="": self.meters.append((action, qty)),
        )


def _run(stages, fake):
    return RunAgent(fake.ports(), {"stages": stages}, dict(_CTX)).run()


def test_agent_fee_is_charged_once_per_stage_even_when_json_is_retried():
    fake = _Fake(["not json", _STORY])
    result = _run([{"format": "story_json"}], fake)
    assert result["ok"] is True
    assert fake.meters == [(MeterAction.AGENT_RUN, 1)]
    assert "not valid JSON" in fake.prompts[1]


def test_json_stage_defaults_cooler_and_retry_never_runs_hotter():
    fake = _Fake(["nope", _STORY])
    _run([{"format": "story_json"}], fake)
    assert fake.temperatures == [JSON_TEMPERATURE, JSON_TEMPERATURE]
    hot = _Fake(["nope", _STORY])
    _run([{"format": "story_json", "temperature": 1.1}], hot)
    assert hot.temperatures == [1.1, JSON_TEMPERATURE]
    text = _Fake(["notes", _STORY])
    _run([{"role": "research"}, {"format": "story_json", "temperature": 0.9}], text)
    assert text.temperatures == [TEXT_TEMPERATURE, 0.9]


def test_a_failing_stage_still_reports_the_tokens_already_spent():
    fake = _Fake(["notes", RuntimeError("provider down")])
    result = _run([{"role": "research"}, {"format": "story_json"}], fake)
    assert result["ok"] is False and result["detail"] == "provider down"
    assert result["usage"]["input_tokens"] == 2 and result["usage"]["output_tokens"] == 3


def test_missing_token_counts_do_not_crash_the_run():
    fake = _Fake([_STORY], usage={})
    result = _run([{"format": "story_json"}], fake)
    assert result["ok"] is True
    assert result["usage"]["input_tokens"] == 0


def test_suggestion_kinds_come_from_one_enum():
    assert Suggestion.of("title") is Suggestion.TITLE
    assert Suggestion.of("headline") is None
    ports = StoryPorts(complete=None, complete_json=None, parse_json=json.loads,
                       prompt=lambda key, **_: key, contract=lambda _n: "",
                       language_rule=lambda *_: "", story_budget=lambda _o=None: 0)
    assert StoryOperations(ports).suggest_field("headline")["error"] == "unknown suggestion kind"


@pytest.mark.parametrize("index", [None, "1", True])
def test_scene_rewrite_requires_a_real_index(index):
    from kinoforge.schemas import StoryOperation
    from kinoforge.service.runtimes.story import StoryRuntime

    class _Request:
        operation = StoryOperation.REWRITE_SCENE
        payload = {"record": {"story": {"scenes": [{"prompt": "p"}]}}, "index": index}
        agent: dict = {}
        config: dict = {}

    result = StoryRuntime._dispatch(StoryOperations(None), _Request(), None)
    assert result == {"ok": False, "error": "a scene index is required"}


def test_image_render_response_has_no_dead_prompt_field():
    from kinoforge.schemas import ImageRenderResponse

    assert "prompt" not in ImageRenderResponse.model_fields
