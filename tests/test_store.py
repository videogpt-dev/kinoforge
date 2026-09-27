"""In-memory ExecutionStore: job binding, record/transcript, clip appends."""

from __future__ import annotations

import pytest

from kinoforge.service.executions import ExecutionStore


def _store(tmp_path, **kw):
    return ExecutionStore(tmp_path / "work", "job1", tmp_path / "src.mp4", **kw)


def test_record_roundtrip(tmp_path):
    store = _store(tmp_path)
    assert store.load_record("job1") is None
    store.save_record("job1", {"id": "job1"})
    assert store.load_record("job1") == {"id": "job1"}
    assert store.record == {"id": "job1"}


def test_transcript_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.save_transcript("job1", [{"text": "hi"}])
    assert store.load_transcript("job1") == [{"text": "hi"}]
    assert store.transcript_path("job1").name == "transcript.json"


def test_wrong_job_id_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.load_record("other")


def test_source_and_paths(tmp_path):
    store = _store(tmp_path)
    assert store.source_path("job1") == tmp_path / "src.mp4"
    assert store.clips_path("job1") == tmp_path / "work" / "artifacts"
    assert store.clip_path("job1", "clip_01") == tmp_path / "work" / "artifacts" / "clip_01"


def test_list_clips_reports_preexisting(tmp_path):
    assert _store(tmp_path, has_clips=True).list_clips("job1") == [{"clip_id": "existing"}]
    assert _store(tmp_path).list_clips("job1") == []


def test_append_clips_skips_zero_length_moments(tmp_path):
    store = _store(tmp_path)
    moments = [{"start": 0, "end": 5}, {"start": 5, "end": 5}, {"start": 6, "end": 9}]
    created = store.append_clips("job1", {}, moments)
    assert created == ["clip_01", None, "clip_03"]
    assert store.list_clips("job1") == [{"clip_id": "clip_01"}, {"clip_id": "clip_03"}]
