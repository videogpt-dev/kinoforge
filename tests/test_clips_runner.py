"""ClipsRunner control flow without ffmpeg: analyze-only, duration gate, cancellation."""

from __future__ import annotations

from kinoforge.contract import Context, Job, JobKind, Runner
from kinoforge.segments.clips.runner import ClipsRunner
from kinoforge.service.executions import ExecutionStore
from tests.support import capturing_logger

_TRANSCRIPT = [{"start": 0, "end": 15, "text": "hello"}]


class _Provider:
    name = "test"

    def discover_moments(self, transcript, min_length, max_length, clip_count):
        return [{"start": 1, "end": 12, "score": 90, "text": "moment"}]


def _save_project(video_path, config, slug, moments, transcript, store):
    store.save_record(slug, {"moments": moments, "transcript": transcript})


def _runner(transcribed, *, duration_error=None, cancelled=False, stages=None) -> ClipsRunner:
    return ClipsRunner(
        logger=capturing_logger()[0],
        report_stage=lambda stage, status: (stages if stages is not None else []).append(
            (stage.value, status.value)),
        check_source_duration=lambda _seconds: duration_error,
        transcribe_video=lambda *args, **kwargs: transcribed.append(args),
        save_project=_save_project,
        moment_provider=lambda _config, _ctx: _Provider(),
        is_cancelled=lambda: cancelled,
    )


def _job(tmp_path, **options) -> Job:
    return Job(kind=JobKind.CLIPS, job_id="video",
               input={"video_path": str(tmp_path / "source.mp4")}, options=options)


def _store(tmp_path) -> ExecutionStore:
    return ExecutionStore(tmp_path / "ws", "video", tmp_path / "source.mp4")


def _duration(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(
        "kinoforge.segments.clips.pipeline.context.get_video_metadata",
        lambda _path: {"duration": seconds},
    )


def test_analyze_only_reuses_captions_and_skips_rendering(monkeypatch, tmp_path):
    _duration(monkeypatch, 30)
    transcribed: list = []
    stages: list = []
    store = _store(tmp_path)
    runner = _runner(transcribed, stages=stages)
    job = _job(tmp_path, slug="video", pretranscript=_TRANSCRIPT, min_length=5, max_length=30,
               clip_count=1, verbose=False, analyze_only=True)

    result = runner.run(job, Context(store=store))

    assert isinstance(runner, Runner)
    assert result.status == "ok"
    assert result.data["transcript"] == _TRANSCRIPT
    assert result.data["moments"][0]["score"] == 90
    assert result.stages == [
        "transcribe:running", "transcribe:skipped", "moments:running", "moments:done",
        "clips:skipped",
    ]
    assert stages[-1] == ("clips", "skipped")
    assert not transcribed
    assert store.load_record("video")["moments"][0]["text"] == "moment"


def test_duration_limit_stops_before_transcription(monkeypatch, tmp_path):
    _duration(monkeypatch, 120)
    transcribed: list = []

    result = _runner(transcribed, duration_error="source too long").run(
        _job(tmp_path), Context(store=_store(tmp_path)))

    assert result.status == "failed"
    assert result.error == "source too long"
    assert result.data["errors"] == ["source too long"]
    assert result.stages == []
    assert not transcribed


def test_cancelled_run_stops_before_work(tmp_path):
    transcribed: list = []

    result = _runner(transcribed, cancelled=True).run(
        _job(tmp_path), Context(store=_store(tmp_path)))

    assert result.status == "cancelled"
    assert result.data["cancelled"] is True
    assert not transcribed
