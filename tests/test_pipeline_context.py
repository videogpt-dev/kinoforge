"""Clips pipeline shared state: StageCtx bookkeeping and the guard stages."""

from __future__ import annotations

from pathlib import Path

from kinoforge.segments.clips.pipeline.context import (
    ClipStage,
    StageStatus,
    already_processed,
    duration_ok,
)

from tests.support import levels, make_ctx


# --- StageCtx -------------------------------------------------------------

def test_stage_reports_and_records():
    ctx, result, reported = make_ctx()
    ctx.stage(ClipStage.MOMENTS, StageStatus.RUNNING)
    assert reported == [(ClipStage.MOMENTS, StageStatus.RUNNING)]
    assert result.stages == ["moments:running"]


def test_fail_sets_status_error_and_appends():
    ctx, result, _ = make_ctx()
    ctx.fail("bad thing")
    assert result.status == "failed"
    assert result.error == "bad thing"
    assert result.data["errors"] == ["bad thing"]


def test_stopped_false_when_not_cancelled():
    ctx, result, reported = make_ctx(is_cancelled=lambda: False)
    assert ctx.stopped(ClipStage.MOMENTS) is False
    assert reported == []
    assert result.status == "ok"


def test_stopped_true_marks_cancelled_and_logs():
    ctx, result, reported = make_ctx(is_cancelled=lambda: True)
    assert ctx.stopped(ClipStage.CLIPS) is True
    assert result.status == "cancelled"
    assert result.error == ""
    assert result.data["cancelled"] is True
    assert reported == [(ClipStage.CLIPS, StageStatus.CANCELLED)]
    assert levels(ctx.logger) == ["warning"]


def test_stopped_without_stage_skips_report():
    ctx, result, reported = make_ctx(is_cancelled=lambda: True)
    assert ctx.stopped() is True
    assert reported == []
    assert result.data["cancelled"] is True


# --- already_processed ----------------------------------------------------

class _Store:
    def __init__(self, clips):
        self._clips = clips

    def list_clips(self, job_id):
        return self._clips


def _cfg(**over):
    base = {"skip_already_processed": True, "is_regenerate": False, "force": False}
    base.update(over)
    return base


def test_already_processed_true_when_skip_and_clips_exist():
    assert already_processed(_Store([{"id": "c1"}]), "j1", "slug", _cfg()) is True


def test_already_processed_false_when_no_clips():
    assert already_processed(_Store([]), "j1", "slug", _cfg()) is False


def test_already_processed_false_when_regenerate_or_force_or_disabled():
    store = _Store([{"id": "c1"}])
    assert already_processed(store, "j1", "slug", _cfg(is_regenerate=True)) is False
    assert already_processed(store, "j1", "slug", _cfg(force=True)) is False
    assert already_processed(store, "j1", "slug", _cfg(skip_already_processed=False)) is False


def test_already_processed_false_when_no_slug():
    assert already_processed(_Store([{"id": "c1"}]), "j1", "", _cfg()) is False


# --- duration_ok ----------------------------------------------------------

def test_duration_ok_true_when_no_error(monkeypatch):
    import kinoforge.segments.clips.pipeline.context as mod

    monkeypatch.setattr(mod, "get_video_metadata", lambda _p: {"duration": 42})
    ctx, result, _ = make_ctx()
    assert duration_ok(ctx, lambda _d: "", Path("v.mp4")) is True
    assert result.status == "ok"


def test_duration_ok_false_fails_result_and_logs(monkeypatch):
    import kinoforge.segments.clips.pipeline.context as mod

    monkeypatch.setattr(mod, "get_video_metadata", lambda _p: {"duration": 1})
    ctx, result, _ = make_ctx()
    seen = {}

    def check(duration):
        seen["duration"] = duration
        return "too short"

    assert duration_ok(ctx, check, Path("v.mp4")) is False
    assert seen["duration"] == 1
    assert result.status == "failed"
    assert result.error == "too short"
    assert levels(ctx.logger) == ["warning"]
