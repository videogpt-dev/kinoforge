"""Story and series runtimes (text stages) over a faked Infrelay."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx")

from kinoforge.schemas import (  # noqa: E402
    SeriesPlanRequest,
    StoryOperationRequest,
    StoryWriteRequest,
)
from kinoforge.service.runtimes.series import SeriesRuntime  # noqa: E402
from kinoforge.service.runtimes.story import StoryRuntime  # noqa: E402
from tests.integration.runtime_support import SETTINGS, bundle, fake  # noqa: E402

USAGE = {"model": "m1", "input_tokens": 40, "output_tokens": 120, "finish_reason": "stop"}
STORY_JSON = json.dumps({
    "logline": "A keeper faces one last night.",
    "style": "moody coastal noir",
    "characters": [{"name": "Ana", "description": "the weathered keeper"}],
    "scenes": [
        {"prompt": "The lamp sweeps the dark water", "narration": "The light still turns."},
        {"prompt": "Ana climbs the iron stair", "narration": "One last climb."},
    ],
})


# --- story ----------------------------------------------------------------

def _storybundle() -> dict:
    return bundle(
        {"type": "agent", "key": "story_system", "body": "You are a screenwriter."},
        {"type": "agent", "key": "story_idea", "body": (
            "Write $scene_count scenes for $title as a $format. $description "
            "Series $series_name position $series_position history $series_history"
        )},
        {"type": "fragment", "key": "prompts.fragments.contract",
         "body": "Return strict JSON with $scene_count scenes."},
        {"type": "fragment", "key": "prompts.fragments.language",
         "body": "Write $subject in $language."},
    )


def _write_request(stages=None) -> StoryWriteRequest:
    return StoryWriteRequest(
        job_id="job-1", project_id="proj-1", owner="tenant-1",
        context={
            "title": "The Lighthouse", "description": "one last night", "scene_count": 2,
            "series_name": "Coastal Noir", "series_position": 2,
            "series_episodes": [{"title": "Pilot"}],
        },
        agent={"persona": "", "stages": stages or [{"role": "writer", "format": "story_json"}]},
        pick={"provider": "openrouter", "model": "m1"},
        config={"budgets": {"story": 3000}}, definitions=_storybundle(),
    )


def test_story_write_returns_plan_and_meter(monkeypatch):
    seen = {}

    def text(self, ref, system, user, *, temperature, max_tokens):
        seen.update(ref=str(ref), system=system, user=user)
        return STORY_JSON, dict(USAGE)

    fake(monkeypatch, "text", text)
    out = StoryRuntime(SETTINGS).write(_write_request())

    assert seen["ref"] == "openrouter/m1"
    assert seen["system"] == "You are a screenwriter."
    assert "strict JSON" in seen["user"]
    assert "Write logline and all narration in" in seen["user"]
    assert "Series Coastal Noir position 2" in seen["user"]
    assert '"title": "Pilot"' in seen["user"]
    result = out["result"]
    assert result["ok"] is True
    assert len(result["story"]["scenes"]) == 2
    assert result["usage"]["model"] == "m1"
    assert [e["action"] for e in out["meter_events"]] == ["agent_run"]


def test_story_write_reports_cutoff(monkeypatch):
    cut = {**USAGE, "output_tokens": 3000, "finish_reason": "length"}
    fake(monkeypatch, "text", lambda self, *a, **k: ("{", cut))
    out = StoryRuntime(SETTINGS).write(_write_request())

    assert out["result"]["ok"] is False
    assert "cut off" in out["result"]["error"]


def test_story_write_honors_cancellation_between_stages(monkeypatch):
    calls = {"n": 0}

    def text(self, *args, **kwargs):
        calls["n"] += 1
        return "research", dict(USAGE)

    fake(monkeypatch, "text", text)
    stages = [{"role": "research", "format": "text"}, {"role": "writer", "format": "story_json"}]
    out = StoryRuntime(SETTINGS).write(_write_request(stages), is_cancelled=lambda: calls["n"] > 0)

    assert out["result"]["cancelled"] is True
    assert calls["n"] == 1


def test_story_rewrites_scene_from_frozen_definition(monkeypatch):
    seen = {}

    def text(self, ref, system, user, *, temperature, max_tokens):
        seen.update(system=system, user=user, max_tokens=max_tokens)
        return json.dumps({"prompt": "New lighthouse shot", "narration": "A final warning."}), USAGE

    fake(monkeypatch, "text", text)
    request = StoryOperationRequest(
        job_id="job-rewrite", project_id="proj-1", owner="tenant-1", operation="rewrite_scene",
        payload={
            "record": {
                "input": {"language": "en"},
                "story": {
                    "logline": "Storm warning", "style": "noir", "characters": [],
                    "scenes": [{"prompt": "Old shot", "narration": "Old line"}],
                },
            },
            "index": 0, "instructions": "Increase tension",
        },
        agent={"persona": "Tense writer", "stages": []},
        pick={"provider": "openrouter", "model": "m1"},
        config={"budgets": {"rewrite": 500}},
        definitions=bundle(
            {"type": "agent", "key": "story_system", "body": "Fallback writer"},
            {"type": "agent", "key": "story_rewrite_scene", "body": (
                "$summary\nScene $scene_num/$scene_total\n$cur_prompt\n$cur_narration\n"
                "$language_rule"
            )},
            {"type": "fragment", "key": "prompts.fragments.language",
             "body": "Write $subject in $language."},
        ),
    )
    out = StoryRuntime(SETTINGS).operate(request)

    assert out["result"] == {
        "ok": True, "scene": {"prompt": "New lighthouse shot", "narration": "A final warning."},
    }
    assert seen["system"] == "Tense writer"
    assert "Increase tension" in seen["user"]
    assert seen["max_tokens"] == 500


# --- series ---------------------------------------------------------------

def _plan_request() -> SeriesPlanRequest:
    return SeriesPlanRequest(
        job_id="series-1-plan", project_id="series-1", owner="tenant-1",
        series={
            "name": "Coastal Noir", "premise": "A lighthouse keeper solves crimes",
            "style": "salt-stained noir", "aspect_ratio": "16:9",
            "cast": [{"name": "Ana", "look": "a weathered keeper"}],
            "episodes": [{"title": "The bell", "description": "A warning rings."}],
        },
        count=3, pick={"provider": "openrouter", "model": "free"},
        config={"budgets": {"story": 4000}},
        definitions=bundle(
            {"type": "agent", "key": "story_system", "body": "You are a showrunner."},
            {"type": "agent", "key": "episode_plan", "body": (
                "Plan $count episodes for $name ($format). Style: $style. "
                "Premise: $premise. Cast: $cast. History: $history."
            )},
        ),
    )


def test_series_plan_builds_prompt_and_returns_episodes(monkeypatch):
    seen = {}
    episodes = [{"title": "Fog", "description": "A body in the surf."}]

    def text(self, ref, system, user, *, temperature, max_tokens):
        seen.update(ref=ref, user=user, max_tokens=max_tokens)
        return json.dumps({"episodes": episodes}), {}

    fake(monkeypatch, "text", text)
    out = SeriesRuntime(SETTINGS).plan(_plan_request())

    assert seen["ref"].provider == "openrouter"
    assert seen["max_tokens"] == 4000
    for part in ("Coastal Noir", "widescreen 16:9", "Ana: a weathered keeper",
                 "salt-stained noir", '"title": "The bell"'):
        assert part in seen["user"]
    assert out["result"] == {"ok": True, "episodes": episodes}
    assert out["meter_events"] == [{"action": "agent_run", "qty": 1.0, "variant": ""}]


def test_series_plan_reports_empty_ideas(monkeypatch):
    fake(monkeypatch, "text", lambda self, *a, **k: (json.dumps({"episodes": []}), {}))
    out = SeriesRuntime(SETTINGS).plan(_plan_request())

    assert out["result"]["ok"] is False
    assert "no episode ideas" in out["result"]["error"]
    assert out["meter_events"] == [{"action": "agent_run", "qty": 1.0, "variant": ""}]
