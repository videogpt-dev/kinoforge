"""ClipsPipeline control flow without ffmpeg: analyze-only, transcript reuse, guards,
cancellation."""

from __future__ import annotations

from pathlib import Path

import pytest

from kinoforge.segments.clips.pipeline import ClipsPipeline
from tests.support import make_run

_TRANSCRIPT = [{"start": 0, "end": 15, "text": "hello"}]
_CONFIG = {"slug": "video", "output_dir": "/out", "min_length": 5, "max_length": 30,
           "clip_count": 1, "analyze_only": True}


class _Engine:
    name = "test"

    def discover_moments(self, transcript, min_length, max_length, clip_count):
        return [{"start": 1, "end": 12, "score": 90, "text": "moment"}]


@pytest.fixture(autouse=True)
def _probe(monkeypatch):
    probe = {"width": 640, "height": 360, "fps": 25.0, "duration": 30.0}
    monkeypatch.setattr("kinoforge.segments.clips.project.get_video_metadata", lambda _p: probe)
    monkeypatch.setattr("kinoforge.segments.clips.pipeline.get_video_metadata", lambda _p: probe)


def _pipeline(transcribed: list) -> ClipsPipeline:
    def transcribe_video(*args, **kwargs):
        transcribed.append(args)
        return list(_TRANSCRIPT)

    return ClipsPipeline(transcribe_video=transcribe_video, moment_engine=_Engine())


def test_analyze_only_uses_captions_and_skips_rendering():
    transcribed: list = []
    run = make_run({**_CONFIG, "pretranscript": _TRANSCRIPT})
    _pipeline(transcribed).run(run)

    assert run.status == "ok"
    assert run.data["transcript"] == _TRANSCRIPT
    assert run.data["moments"][0]["score"] == 90
    assert run.stages == [
        "transcribe:running", "transcribe:skipped", "moments:running", "moments:done",
        "clips:skipped",
    ]
    assert not transcribed
    assert run.record["moments"][0]["text"] == "moment"


def test_transcript_from_state_is_reused():
    transcribed: list = []
    run = make_run(_CONFIG, transcript=list(_TRANSCRIPT))
    _pipeline(transcribed).run(run)
    assert not transcribed
    assert "transcribe:skipped" in run.stages


def test_force_transcribes_again_and_fills_state():
    transcribed: list = []
    run = make_run({**_CONFIG, "force": True}, transcript=[{"start": 0, "end": 1, "text": "x"}])
    _pipeline(transcribed).run(run)
    assert len(transcribed) == 1
    assert run.transcript == _TRANSCRIPT


def test_duration_limit_stops_before_transcription():
    transcribed: list = []
    run = make_run({**_CONFIG, "limits": {"source_max_seconds": 10}})
    _pipeline(transcribed).run(run)
    assert run.status == "failed"
    assert run.error.startswith("Video is")
    assert run.stages == []
    assert not transcribed


def test_source_under_the_limit_runs():
    run = make_run({**_CONFIG, "limits": {"source_max_seconds": 600}},
                   transcript=list(_TRANSCRIPT))
    _pipeline([]).run(run)
    assert run.status == "ok"


def test_already_processed_is_skipped():
    transcribed: list = []
    run = make_run({**_CONFIG, "skip_already_processed": True}, has_clips=True)
    _pipeline(transcribed).run(run)
    assert run.status == "skipped"
    assert not transcribed


def test_regenerate_ignores_existing_clips():
    transcribed: list = []
    run = make_run({**_CONFIG, "skip_already_processed": True, "is_regenerate": True},
                   has_clips=True)
    _pipeline(transcribed).run(run)
    assert run.status == "ok"


def test_cancelled_run_stops_before_work():
    transcribed: list = []
    run = make_run(_CONFIG, is_cancelled=lambda: True)
    _pipeline(transcribed).run(run)
    assert run.status == "cancelled"
    assert run.data["cancelled"] is True
    assert not transcribed


def test_audio_only_render_fails_loudly():
    run = make_run({**_CONFIG, "analyze_only": False, "pretranscript": _TRANSCRIPT},
                   video=None, audio=Path("a.mp3"))
    _pipeline([]).run(run)
    assert run.status == "failed"
    assert "audio-only" in run.error
