"""RunAgent.run: every outcome path (characterization of the story stage runner)."""

from __future__ import annotations

import json
from typing import List

from kinoforge.contract import ModelRef
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.run_agent import RunAgent

_CTX = {
    "title": "Heist", "description": "", "scene_count": 1, "output_format": "vertical",
    "language_rule": "RULE",
}
_STORY = {"logline": "l", "style": "s", "characters": [],
          "scenes": [{"prompt": "a vault", "narration": "they wait"}]}


class _Ports:
    def __init__(self, replies: List) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete(self, purpose, system, user, *, route, temperature, max_tokens):
        self.calls += 1
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return text, {"input_tokens": 3, "output_tokens": 5, "model": route.model or "m",
                      "finish_reason": finish}

    def ports(self) -> StoryPorts:
        return StoryPorts(
            complete=self.complete, complete_json=None, parse_json=json.loads,
            prompt=lambda key, **_: f"<{key}>", contract=lambda _n: "CONTRACT",
            language_rule=lambda *_: "", story_budget=lambda _o=None: 100,
        )


def _run(stages, replies, ctx=_CTX, cancelled=lambda: False, route=None):
    fake = _Ports(replies)
    result = RunAgent(fake.ports(), {"stages": stages}, dict(ctx), route=route,
                      is_cancelled=cancelled).run()
    return result, fake


_JSON_STAGE = {"format": "story_json", "role": "writer"}


def test_happy_path_returns_story_usage_and_warnings():
    result, _ = _run([_JSON_STAGE], [json.dumps(_STORY)])
    assert result["ok"] is True
    assert result["story"]["scenes"][0]["narration"] == "they wait"
    assert result["usage"]["input_tokens"] == 3 and result["usage"]["models"] == ["m"]
    assert result["warnings"] == []


def test_plain_stages_then_last_output_parsed_as_story():
    result, fake = _run([{"role": "research"}, {"role": "writer"}], ["notes", json.dumps(_STORY)])
    assert result["ok"] is True and fake.calls == 2


def test_validation_errors():
    assert _run([], [])[0] == {"ok": False, "error": "agent has no stages"}
    assert _run([_JSON_STAGE], [], ctx={**_CTX, "title": ""})[0] == {
        "ok": False, "error": "title is required"}


def test_stage_failure_names_the_stage_model():
    result, _ = _run([{"role": "writer", "model": "x1", "provider": "op"}], [RuntimeError("429")])
    assert result["ok"] is False
    assert "Story 'writer' stage (model 'x1' on op) failed: 429" in result["error"]


def test_cut_off_json_blames_the_budget():
    result, _ = _run([_JSON_STAGE], [('{"scenes": [', "length")])
    assert "cut off at 5 tokens" in result["error"] and "usage" in result


def test_invalid_json_suggests_retry():
    result, _ = _run([_JSON_STAGE], ["not json"])
    assert "did not return valid JSON" in result["error"]


def test_no_scenes_and_unparseable_final_output():
    empty = json.dumps({**_STORY, "scenes": []})
    assert _run([_JSON_STAGE], [empty])[0]["error"] == "story had no usable scenes"
    assert _run([{"role": "w"}], ["prose"])[0]["error"] == "agent produced no valid story JSON"


def test_quality_gate_blocks_silent_scenes():
    silent = json.dumps({**_STORY, "scenes": [{"prompt": "p", "narration": ""}]})
    result, _ = _run([_JSON_STAGE], [silent])
    assert result["error"].startswith("Story can't be filmed: ")


def test_cancellation_before_and_after_a_stage():
    before, fake = _run([_JSON_STAGE], [json.dumps(_STORY)], cancelled=lambda: True)
    assert before == {"ok": False, "cancelled": True, "error": "story execution cancelled"}
    assert fake.calls == 0
    ticks = iter([False, True])
    after, _ = _run([_JSON_STAGE], [json.dumps(_STORY)], cancelled=lambda: next(ticks))
    assert after["cancelled"] is True and "usage" in after


def test_stage_route_override_reaches_usage():
    result, _ = _run([{**_JSON_STAGE, "model": "big"}], [json.dumps(_STORY)],
                     route=ModelRef("op", "small"))
    assert result["usage"]["models"] == ["big"]
