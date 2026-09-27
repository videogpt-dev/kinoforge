"""Moment analysis coercion: neutral fallbacks and normalizing a parsed model object."""

from __future__ import annotations

from kinoforge.segments.clips.moments.analysis import MomentAnalysis

fallback_analyses = MomentAnalysis.fallback
normalize_analysis = MomentAnalysis.normalize


def test_fallback_analyses_keeps_boundaries_at_middling_score():
    batch = [{"start": 1.0, "end": 4.0}, {"start": 5.0, "end": 9.0}]
    out = fallback_analyses(batch)
    assert [a["start"] for a in out] == [1.0, 5.0]
    assert all(a["worthy"] and a["score"] == 60.0 and a["reason"] == "fallback" for a in out)


def test_normalize_clamps_window_and_keeps_score():
    moment = {"start": 0.0, "end": 20.0}
    obj = {"score": 88, "worthy": True, "start": -3, "end": 100, "reason": "r", "hook": "h"}
    out = normalize_analysis(obj, moment, window=(0.0, 30.0))
    assert out["start"] == 0.0 and out["end"] == 30.0  # clamped to window
    assert out["score"] == 88


def test_normalize_bad_score_defaults_to_60():
    out = normalize_analysis({"score": "abc"}, {"start": 0, "end": 5}, window=(0.0, 5.0))
    assert out["score"] == 60.0


def test_normalize_score_is_clamped_0_100():
    assert normalize_analysis({"score": 250}, {"start": 0, "end": 5}, (0, 5))["score"] == 100
    assert normalize_analysis({"score": -9}, {"start": 0, "end": 5}, (0, 5))["score"] == 0


def test_normalize_string_worthy_is_coerced():
    assert normalize_analysis({"worthy": "yes"}, {"start": 0, "end": 5}, (0, 5))["worthy"] is True
    assert normalize_analysis({"worthy": "nope"}, {"start": 0, "end": 5}, (0, 5))["worthy"] is False


def test_normalize_keeps_original_boundaries_when_window_span_too_small():
    # ne - ns < 2.0 after clamping -> fall back to the moment's own boundaries.
    moment = {"start": 10.0, "end": 40.0}
    out = normalize_analysis({"start": 1, "end": 1.5}, moment, window=(0.0, 50.0))
    assert out["start"] == 10.0 and out["end"] == 40.0


def test_normalize_truncates_reason_and_hook():
    out = normalize_analysis(
        {"reason": "x" * 500, "hook": "y" * 500}, {"start": 0, "end": 5}, (0, 5)
    )
    assert len(out["reason"]) == 200
    assert len(out["hook"]) == 120
