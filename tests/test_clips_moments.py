"""Moments stage: preset find, discovered ranking, limits, and clip-id assignment."""

from __future__ import annotations

import pytest

from kinoforge.segments.clips.moments.finders import MomentFinder
from kinoforge.segments.clips.stages.moments import Moments
from tests.support import make_run, texts

_BASE = {"min_length": 0, "max_length": 0, "clip_count": 10, "output_dir": "/out", "slug": "s"}


class _Finder(MomentFinder):
    name = "stub"

    def find(self, transcript, spec):
        return [{"start": 1, "end": 12, "score": 90, "text": "moment"}]


@pytest.fixture(autouse=True)
def _no_probe(monkeypatch):
    monkeypatch.setattr(
        "kinoforge.segments.clips.render.ffmpeg.Ffmpeg.probe",
        lambda _p: {"width": 640, "height": 360, "fps": 25.0, "duration": 30.0},
    )


def _limit(config, ranked):
    run = make_run(config)
    return Moments(_Finder())._limit(run, ranked), run


# --- find + rank ------------------------------------------------------------

def test_preset_moments_skip_discovery_and_sort_by_score():
    preset = [{"start": 0.0, "end": 5.0, "score": 10}, {"start": 6.0, "end": 9.0, "score": 90}]
    run = make_run({**_BASE, "preset_moments": preset})
    assert Moments(_Finder())(run) is True
    assert [m["score"] for m in run.data["used_moments"]] == [90, 10]
    assert run.data["moments"] is not preset  # copied, not aliased
    assert any("preset moments" in t for t in texts(run.logger))
    assert run.stages == ["moments:running", "moments:done"]


def test_preset_missing_score_sorts_last():
    run = make_run({**_BASE, "preset_moments": [{"start": 0, "end": 1}, {"start": 2, "end": 3,
                                                                          "score": 5}]})
    Moments(_Finder())(run)
    assert [m.get("score") for m in run.data["used_moments"]] == [5, None]


def test_finder_moments_are_used():
    run = make_run(_BASE)
    assert Moments(_Finder())(run) is True
    assert run.record["moments"][0]["text"] == "moment"
    assert run.clip_ids == ["clip_01"]


def test_stops_when_cancelled():
    run = make_run({**_BASE, "preset_moments": [{"start": 0, "end": 5}]},
                   is_cancelled=lambda: True)
    assert Moments(_Finder())(run) is False
    assert run.status == "cancelled"


def test_clip_ids_skip_zero_length_and_merge_existing_record():
    preset = [{"start": 0, "end": 5, "score": 3}, {"start": 5, "end": 5, "score": 2},
              {"start": 6, "end": 9, "score": 1}]
    existing = {"moments": [{"start": 20.0, "end": 25.0, "text": "old", "score": 1, "id": 1}]}
    run = make_run({**_BASE, "preset_moments": preset}, record=existing)
    Moments(_Finder())(run)
    assert run.clip_ids == ["clip_01", None, "clip_03"]
    assert [m["start"] for m in run.record["moments"]] == [20.0, 0.0, 5.0, 6.0]


# --- limits -----------------------------------------------------------------

def test_threshold_on_0_100_scale_keeps_high_scorers():
    out, _ = _limit({"min_interest_score": 0.5},
                    [{"score": 80, "start": 0, "end": 5}, {"score": 20, "start": 6, "end": 11}])
    assert [m["score"] for m in out] == [80]


def test_threshold_detects_0_10_scale():
    out, _ = _limit({"min_interest_score": 0.5},
                    [{"score": 8, "start": 0, "end": 5}, {"score": 2, "start": 6, "end": 11}])
    assert [m["score"] for m in out] == [8]


def test_keeps_top_when_none_meet_threshold():
    ranked = [{"score": 5, "start": 0, "end": 5}]  # 5/10 scale = 0.5, below 0.9
    out, run = _limit({"min_interest_score": 0.9}, ranked)
    assert out == ranked
    assert any("using top results" in t for t in texts(run.logger))


def test_backfills_when_fewer_pass_than_clip_count():
    # Only 1 clears the threshold but 5 are requested: keep the full ordered list so the
    # clip_count slice is not starved to 1 (the "Kept 1/10" bug).
    out, run = _limit({"min_interest_score": 0.5, "clip_count": 5},
                      [{"score": 80, "start": 0, "end": 5}, {"score": 20, "start": 6, "end": 11}])
    assert [m["score"] for m in out] == [80, 20]
    assert any("requested count is met" in t for t in texts(run.logger))


def test_trims_over_max_length():
    out, _ = _limit({"max_length": 10}, [{"start": 0.0, "end": 30.0}])
    assert out[0]["end"] == 10.0
    assert out[0]["duration"] == 10


def test_drops_under_min_length_when_some_qualify():
    out, _ = _limit({"min_length": 5}, [{"start": 0.0, "end": 2.0}, {"start": 0.0, "end": 8.0}])
    assert out == [{"start": 0.0, "end": 8.0}]


def test_keeps_all_when_all_under_min_length():
    ranked = [{"start": 0.0, "end": 2.0}, {"start": 0.0, "end": 3.0}]
    out, run = _limit({"min_length": 5}, ranked)
    assert out == ranked
    assert any("keeping them anyway" in t for t in texts(run.logger))
