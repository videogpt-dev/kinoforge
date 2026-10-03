"""ClipWorker bookkeeping: stage records, failure, cancellation."""

from __future__ import annotations

from kinoforge.segments.clips.worker import ClipStage
from kinoforge.segments.common import StageStatus
from tests.support import levels, make_worker


def test_stage_records_status():
    run = make_worker()
    run.stage(ClipStage.MOMENTS, StageStatus.RUNNING)
    assert run.stages == ["moments:running"]


def test_fail_sets_status_error_and_appends():
    run = make_worker()
    run.fail("bad thing")
    assert run.status == "failed"
    assert run.error == "bad thing"
    assert run.data["errors"] == ["bad thing"]


def test_stopped_false_when_not_cancelled():
    run = make_worker()
    assert run.stopped(ClipStage.MOMENTS) is False
    assert run.stages == []
    assert run.status == "ok"


def test_stopped_true_marks_cancelled_and_logs():
    run = make_worker(is_cancelled=lambda: True)
    assert run.stopped(ClipStage.CLIPS) is True
    assert run.status == "cancelled"
    assert run.error == ""
    assert run.data["cancelled"] is True
    assert run.stages == ["clips:cancelled"]
    assert levels(run.logger)[-1] == "warning"


def test_stopped_without_stage_records_nothing():
    run = make_worker(is_cancelled=lambda: True)
    assert run.stopped() is True
    assert run.stages == []
    assert run.data["cancelled"] is True


def test_data_starts_with_the_response_keys():
    run = make_worker(video=None, audio=None)
    assert run.data == {"video_path": "", "audio_path": "", "clips": [], "moments": [],
                        "transcript": None, "errors": []}
