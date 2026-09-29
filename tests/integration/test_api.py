"""API integration tests over the real FastAPI app: routing, path guards, validation.

Exercises the wiring end to end via TestClient without reaching any runtime/ffmpeg/infrelay
work (every case returns at a guard or request validation). Skipped when fastapi/httpx are
absent (dev box); runs under the container/venv deps."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from kinoforge.api.app import app  # noqa: E402


def _bundle() -> dict:
    return {"schema_version": 1, "id": "cat", "version": "1.0.0", "engine": {"minimum": "0.1.0"}}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# --- routing / discovery --------------------------------------------------

def test_health_is_public_and_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "service": "kinoforge", "version": "0.1.0"}


def test_segments_lists_three_available_segments(client):
    r = client.get("/v1/segments")
    assert r.status_code == 200
    segments = r.json()["segments"]
    assert {s["code_name"] for s in segments} == {"clips", "story", "series"}
    assert all(s["status"] in {"available", "planned"} for s in segments)


def test_unknown_route_is_404(client):
    assert client.get("/v1/does-not-exist").status_code == 404


# --- clips execute guards -------------------------------------------------

def test_execute_requires_audio_or_video(client):
    body = {
        "job_id": "j1", "project_id": "p1", "workspace": "work",
        "input": {}, "definitions": _bundle(),
    }
    r = client.post("/v1/segments/clips/execute", json=body)
    assert r.status_code == 400
    assert "audio_path or video_path" in r.json()["detail"]


def test_execute_rejects_non_clips_segment(client):
    body = {
        "job_id": "j1", "project_id": "p1", "workspace": "work",
        "input": {"video_path": "x.mp4"}, "definitions": _bundle(),
    }
    r = client.post("/v1/segments/story/execute", json=body)
    assert r.status_code == 404


# --- shared-root path guards ----------------------------------------------

def test_probe_rejects_path_outside_shared_root(client, tmp_path, monkeypatch):
    monkeypatch.setenv("KINOFORGE_SHARED_ROOT", str(tmp_path))
    r = client.post("/v1/segments/clips/media/probe", json={"path": "/etc/passwd"})
    assert r.status_code == 400
    assert "outside shared root" in r.json()["detail"]


def test_probe_missing_file_under_root_is_404(client, tmp_path, monkeypatch):
    monkeypatch.setenv("KINOFORGE_SHARED_ROOT", str(tmp_path))
    r = client.post(
        "/v1/segments/clips/media/probe", json={"path": str(tmp_path / "gone.mp4")}
    )
    assert r.status_code == 404


# --- request validation (pydantic model_validators) -----------------------

def test_render_base_rejects_inverted_window(client):
    body = {
        "source": "a.mp4", "output": "b.mp4", "in_sec": 2.0, "out_sec": 1.0,
        "src_width": 1920, "src_height": 1080,
    }
    assert client.post("/v1/segments/clips/media/render-base", json=body).status_code == 422


def test_render_base_rejects_crop_outside_frame(client):
    body = {
        "source": "a.mp4", "output": "b.mp4", "in_sec": 0.0, "out_sec": 5.0,
        "src_width": 1920, "src_height": 1080,
        "crop": {"x": 0.9, "y": 0.0, "w": 0.9, "h": 0.1},
    }
    assert client.post("/v1/segments/clips/media/render-base", json=body).status_code == 422


def test_clips_options_reject_inverted_length_window(client):
    body = {
        "job_id": "j1", "project_id": "p1", "workspace": "work",
        "input": {"video_path": "x.mp4"}, "definitions": _bundle(),
        "options": {"min_length": 60, "max_length": 10},
    }
    assert client.post("/v1/segments/clips/execute", json=body).status_code == 422


# --- execution lease + enum validation -----------------------------------

def test_second_run_of_an_active_job_is_409(client, monkeypatch, tmp_path):
    from kinoforge.service.executions import executions

    monkeypatch.setenv("KINOFORGE_SHARED_ROOT", str(tmp_path))
    (tmp_path / "v.mp4").write_bytes(b"x")
    body = {
        "job_id": "busy", "project_id": "p1", "workspace": str(tmp_path / "w"),
        "input": {"video_path": str(tmp_path / "v.mp4")}, "definitions": _bundle(),
    }
    with executions.running("busy"):
        r = client.post("/v1/segments/clips/execute", json=body)
    assert r.status_code == 409
    assert "already active" in r.json()["detail"]


def test_unknown_output_format_is_422(client):
    body = {
        "job_id": "j1", "project_id": "p1", "workspace": "w",
        "input": {"video_path": "v.mp4"}, "definitions": _bundle(),
        "options": {"formats": ["4:3"]},
    }
    assert client.post("/v1/segments/clips/execute", json=body).status_code == 422
