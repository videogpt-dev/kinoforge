"""Value objects introduced by the smell cleanup: ModelRef, PromptCeiling, Moment."""

from __future__ import annotations

import pytest

from kinoforge.contract import ModelRef
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.story.media.ceiling import PromptCeiling, limit_from_error

# --- ModelRef -------------------------------------------------------------

def test_model_ref_from_mapping_strips_and_tolerates_missing():
    assert ModelRef.from_mapping({"provider": " op ", "model": " m "}) == ModelRef("op", "m")
    assert ModelRef.from_mapping(None) == ModelRef()
    assert ModelRef.from_mapping({"provider": "op"}) == ModelRef("op", "")


def test_model_ref_overrides_are_independent():
    base = ModelRef("op", "m")
    assert base.with_overrides(model="m2") == ModelRef("op", "m2")
    assert base.with_overrides(provider="other") == ModelRef("other", "m")
    assert base.with_overrides("  ", "") == base  # blanks keep the base


def test_model_ref_str():
    assert str(ModelRef("op", "m")) == "op/m"
    assert str(ModelRef("op")) == "op/default"


# --- PromptCeiling --------------------------------------------------------

def _rejects_over(declared: int, calls: list):
    def attempt(limit: int) -> str:
        calls.append(limit)
        if len(calls) == 1:
            raise RuntimeError(f"input.prompt must be less than or equal to {declared} chars")
        return "ok"
    return attempt


def test_ceiling_learns_a_smaller_limit_and_retries_once():
    calls: list = []
    ceiling = PromptCeiling(1000)
    assert ceiling.run(_rejects_over(300, calls)) == "ok"
    assert calls == [1000, 300]
    assert ceiling.learned == 300


def test_ceiling_reraises_when_declared_limit_is_not_smaller():
    ceiling = PromptCeiling(200)
    with pytest.raises(RuntimeError):
        ceiling.run(_rejects_over(300, []))
    assert ceiling.learned == 0


def test_ceiling_reraises_unrelated_errors():
    def attempt(limit: int) -> str:
        raise ValueError("network down")
    with pytest.raises(ValueError):
        PromptCeiling(100).run(attempt)


def test_limit_from_error_parses_or_zero():
    assert limit_from_error(RuntimeError("input.prompt: maximum length 512")) == 512
    assert limit_from_error(RuntimeError("something else")) == 0


# --- Moment ---------------------------------------------------------------

def test_moment_new_sets_consistent_span_and_keeps_extras():
    data = Moment.new(2, 7, text="hi", title="t")
    assert data == {"text": "hi", "title": "t", "start": 2.0, "end": 7.0, "duration": 5.0}


def test_moment_clamp_keeps_duration_in_sync():
    data = Moment.new(10, 100)
    Moment(data).clamp_to(30)
    assert (data["start"], data["end"], data["duration"]) == (10.0, 40.0, 30.0)
    Moment(data).clamp_to(0)  # 0 = no limit
    assert data["end"] == 40.0


def test_moment_overlap_ratio_uses_the_shorter_span():
    a = Moment(Moment.new(0, 10))
    b = Moment(Moment.new(5, 25))
    assert a.overlap_ratio(b) == pytest.approx(0.5)
    assert a.overlap_ratio(Moment(Moment.new(20, 30))) == 0.0


def test_moment_score_scale_detects_0_10_vs_0_100():
    assert Moment.score_scale([{"score": 8}, {"score": 3}]) == 10.0
    assert Moment.score_scale([{"score": 80}]) == 100.0
    assert Moment.score_scale([]) == 10.0


def test_moment_tolerates_missing_fields():
    view = Moment({})
    assert (view.start, view.end, view.span, view.score) == (0.0, 0.0, 0.0, 0.0)

