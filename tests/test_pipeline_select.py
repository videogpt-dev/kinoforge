"""Moment selection stages: preset find, discovered ranking, and limit application."""

from __future__ import annotations

from kinoforge.segments.clips.pipeline.select import (
    apply_limits,
    find_moments,
    rank_moments,
)

from tests.support import make_ctx, make_run, texts


# --- find_moments (preset branch, no provider) ----------------------------

def test_find_moments_uses_preset_and_skips_discovery():
    ctx, result, _ = make_ctx()
    preset = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 9.0}]
    out = find_moments(make_run({"preset_moments": preset}, sc=ctx), None)
    assert out is not None
    moments, discovered = out
    assert discovered is True
    assert len(moments) == 2
    assert moments is not preset  # copied, not aliased
    assert result.data["moments"] == moments
    assert any("preset moments" in t for t in texts(ctx.logger))


def test_find_moments_stops_when_cancelled():
    ctx, _, _ = make_ctx(is_cancelled=lambda: True)
    out = find_moments(make_run({"preset_moments": [{"start": 0, "end": 5}]}, sc=ctx), None)
    assert out is None


# --- rank_moments (discovered path is a pure score sort) ------------------

def test_rank_moments_discovered_sorts_by_score_desc():
    ctx, _, _ = make_ctx()
    moments = [{"score": 10}, {"score": 90}, {"score": 50}]
    ranked = rank_moments(make_run(sc=ctx), None, moments, discovered=True)
    assert [m["score"] for m in ranked] == [90, 50, 10]


def test_rank_moments_discovered_treats_missing_score_as_zero():
    ctx, _, _ = make_ctx()
    ranked = rank_moments(make_run(sc=ctx), None, [{"score": 5}, {}], discovered=True)
    assert [m.get("score") for m in ranked] == [5, None]


# --- apply_limits ---------------------------------------------------------

def test_apply_limits_threshold_on_0_100_scale_keeps_high_scorers():
    ctx, _, _ = make_ctx()
    ranked = [{"score": 80, "start": 0, "end": 5}, {"score": 20, "start": 6, "end": 11}]
    out = apply_limits(ctx, ranked, {"min_interest_score": 0.5})
    assert [m["score"] for m in out] == [80]


def test_apply_limits_threshold_detects_0_10_scale():
    ctx, _, _ = make_ctx()
    ranked = [{"score": 8, "start": 0, "end": 5}, {"score": 2, "start": 6, "end": 11}]
    out = apply_limits(ctx, ranked, {"min_interest_score": 0.5})
    assert [m["score"] for m in out] == [8]


def test_apply_limits_keeps_top_when_none_meet_threshold():
    ctx, _, _ = make_ctx()
    ranked = [{"score": 5, "start": 0, "end": 5}]  # 5/10 scale = 0.5, below 0.9
    out = apply_limits(ctx, ranked, {"min_interest_score": 0.9})
    assert out == ranked
    assert any("using top results" in t for t in texts(ctx.logger))


def test_apply_limits_backfills_when_fewer_pass_than_clip_count():
    # Only 1 clears the threshold but 5 are requested: keep the full ordered list so the
    # runner's clip_count slice is not starved to 1 (the "Kept 1/10" bug).
    ctx, _, _ = make_ctx()
    ranked = [{"score": 80, "start": 0, "end": 5}, {"score": 20, "start": 6, "end": 11}]
    out = apply_limits(ctx, ranked, {"min_interest_score": 0.5, "clip_count": 5})
    assert [m["score"] for m in out] == [80, 20]
    assert any("requested count is met" in t for t in texts(ctx.logger))


def test_apply_limits_trims_over_max_length():
    ctx, _, _ = make_ctx()
    ranked = [{"start": 0.0, "end": 30.0}]
    out = apply_limits(ctx, ranked, {"max_length": 10})
    assert out[0]["end"] == 10.0
    assert out[0]["duration"] == 10


def test_apply_limits_drops_under_min_length_when_some_qualify():
    ctx, _, _ = make_ctx()
    ranked = [{"start": 0.0, "end": 2.0}, {"start": 0.0, "end": 8.0}]
    out = apply_limits(ctx, ranked, {"min_length": 5})
    assert out == [{"start": 0.0, "end": 8.0}]


def test_apply_limits_keeps_all_when_all_under_min_length():
    ctx, _, _ = make_ctx()
    ranked = [{"start": 0.0, "end": 2.0}, {"start": 0.0, "end": 3.0}]
    out = apply_limits(ctx, ranked, {"min_length": 5})
    assert out == ranked
    assert any("keeping them anyway" in t for t in texts(ctx.logger))
