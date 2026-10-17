"""AiMomentFinder: time-split transcript parts, reply parsing, and failing loudly."""

from __future__ import annotations

from typing import List

import pytest

from kinoforge.contract import ModelRef
from kinoforge.segments.clips.moments.finders import AiMomentFinder, MomentSpec
from kinoforge.segments.clips.moments.finders.ai import MomentFinderError

_SEGS = [{"start": float(i), "end": float(i + 1), "text": f"line {i}"} for i in range(40)]
_SPEC = MomentSpec(min_len=5, max_len=30, count=3)
_PICK = '[{"start": 2.0, "end": 12.0, "score": 70}]'


def _finder(reply: str = _PICK, context_window: int = 1_000_000,
            prompts: List[dict] | None = None, calls: List[dict] | None = None) -> AiMomentFinder:
    def render(name, **params):
        if prompts is not None and params:
            prompts.append(params)
        return name

    def complete(route, prompt, **kwargs):
        if calls is not None:
            calls.append({"route": route, "prompt": prompt, **kwargs})
        return reply, ModelRef("op", "m")

    return AiMomentFinder(
        ModelRef("op", "m"), complete=complete, render=render, context_window=context_window,
    )


def test_rules_go_in_the_system_prompt_and_transcript_in_the_user_prompt():
    calls: list = []
    _finder(calls=calls).find(_SEGS, _SPEC)
    assert [(c["system"], c["prompt"]) for c in calls] == [
        ("prompts.agents.moment_discovery_system", "prompts.agents.moment_discovery"),
    ]


def test_small_transcript_is_sent_whole_in_one_part():
    prompts: list = []
    _finder(prompts=prompts).find(_SEGS, _SPEC)
    assert len(prompts) == 1
    assert prompts[0]["part"] == "This is the whole transcript."
    assert prompts[0]["transcript"].splitlines()[-1] == "[39.0] line 39"
    assert prompts[0]["count"] == "3"


def test_long_transcript_is_split_by_time_with_overlap_and_no_tail_lost():
    segs = [{"start": float(i), "end": float(i + 1), "text": "x" * 200} for i in range(60)]
    prompts: list = []
    _finder(context_window=1, prompts=prompts).find(segs, MomentSpec(5, 3, 1))
    assert len(prompts) > 1
    assert "part 1 of" in prompts[0]["part"] and "0:00-" in prompts[0]["part"]
    assert prompts[-1]["transcript"].splitlines()[-1].startswith("[59.0]")
    first = prompts[0]["transcript"].splitlines()
    second = prompts[1]["transcript"].splitlines()
    assert second[0] in first


def test_parses_fenced_json_and_snaps_to_segments():
    reply = 'sure:\n```json\n[{"start": 2.4, "end": 12.2, "score": 80, "title": "t"}]\n```'
    moments = _finder(reply).find(_SEGS, _SPEC)
    assert [(m["start"], m["end"], m["score"]) for m in moments] == [(2.0, 13.0, 80)]
    assert moments[0]["ai_scored"] is True


def test_reads_the_json_after_a_reasoning_preamble():
    reply = ('Let me list lines:\n[0.0] So, you have an idea\n[4.9] and more\n'
             'Final: [{"start": 2.0, "end": 12.0, "score": 70}]')
    moments = _finder(reply).find(_SEGS, _SPEC)
    assert [(m["start"], m["score"]) for m in moments] == [(2.0, 70)]


def test_returns_at_most_count_moments():
    reply = str([{"start": s, "end": s + 6, "score": 90 - s} for s in (0, 10, 20, 30)])
    moments = _finder(reply.replace("'", '"')).find(_SEGS, MomentSpec(5, 30, 2))
    assert len(moments) == 2


def test_no_picks_fails_the_run_instead_of_guessing():
    with pytest.raises(MomentFinderError, match="picked no usable moments"):
        _finder("no json here").find(_SEGS, _SPEC)


def test_too_few_moments_ask_the_routes_next_model_and_keep_both_answers():
    replies = {"a": '[{"start": 0, "end": 10, "score": 90}]',
               "b": '[{"start": 20, "end": 30, "score": 80}]'}
    asked: list = []

    def complete(route, prompt, **kwargs):
        asked.append(route.exclude)
        model = "b" if route.exclude else "a"
        return replies[model], ModelRef("op", model)

    finder = AiMomentFinder(ModelRef(), complete=complete, render=lambda name, **_: name)
    moments = finder.find(_SEGS, MomentSpec(min_len=5, max_len=10, count=2))

    assert [m["start"] for m in moments] == [0.0, 20.0]
    assert asked == [(), (ModelRef("op", "a"),)]


def test_a_gateway_that_answers_with_the_same_model_again_is_not_asked_forever():
    calls: list = []
    with pytest.raises(MomentFinderError):
        _finder("[]", calls=calls).find(_SEGS, _SPEC)
    assert len(calls) == 2


def test_a_failed_model_call_fails_the_run_with_its_reason():
    def complete(route, prompt, **kwargs):
        raise RuntimeError("no answer within 120s")

    finder = AiMomentFinder(ModelRef("op", "m"), complete=complete, render=lambda name, **_: name)
    with pytest.raises(MomentFinderError, match="did not answer"):
        finder.find(_SEGS, _SPEC)
