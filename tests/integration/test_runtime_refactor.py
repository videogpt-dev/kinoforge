"""Service/inference pieces after the smell cleanup (httpx-gated: they import the client)."""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("httpx")

from kinoforge.contract import Context, ImageSpec, JobKind, ModelRef, VideoSpec  # noqa: E402
from kinoforge.inference import InfrelayClient, InfrelayError  # noqa: E402
from kinoforge.inference import infrelay as infrelay_module  # noqa: E402
from kinoforge.observ import bind, build_logger, reset  # noqa: E402
from kinoforge.schemas import SegmentStatus  # noqa: E402
from kinoforge.segments.clips.moments.ai_engine import AiMomentEngine  # noqa: E402
from kinoforge.segments.clips.moments.offline_engine import OfflineMomentEngine  # noqa: E402
from kinoforge.segments.clips.transcription import TranscriptionError  # noqa: E402
from kinoforge.service.discovery import SegmentCatalog  # noqa: E402
from kinoforge.service.runtimes.clips import SourceDurationLimit  # noqa: E402
from kinoforge.service.runtimes.moment_engines import MomentEngines  # noqa: E402
from kinoforge.service.runtimes.transcription import GatewayTranscription  # noqa: E402


class _Response:
    def __init__(self, output, status=200):
        self.status_code = status
        self._body = {"request_id": "r1", "output": output}
        self.text = "boom"

    def json(self):
        return self._body


def _capture(monkeypatch, output):
    sent = []

    def post(url, headers, json, timeout):
        sent.append(json)
        return _Response(output)

    monkeypatch.setattr(infrelay_module.httpx, "post", post)
    return sent


# --- InfrelayClient -------------------------------------------------------

def test_image_payload_comes_from_the_spec(monkeypatch):
    sent = _capture(monkeypatch, {"type": "b64", "value": base64.b64encode(b"png").decode()})
    spec = ImageSpec(aspect_ratio="1:1", seed=7, mature=True, negative="blur",
                     reference=b"ref", enhance=True, enhance_style="cine")
    data = InfrelayClient("http://gw", tenant="t").image(ModelRef("fal", "flux"), "a cat", spec)
    assert data == b"png"
    body = sent[0]
    assert (body["kind"], body["provider"], body["model"], body["tenant_id"]) == (
        "image", "fal", "flux", "t")
    assert body["input"] == {
        "prompt": "a cat", "aspect_ratio": "1:1", "safe": False, "model": "flux", "seed": 7,
        "negative": "blur", "reference_b64": base64.b64encode(b"ref").decode(),
        "enhance": True, "enhance_style": "cine",
    }


def test_video_payload_carries_dialogue_and_music_off(monkeypatch):
    sent = _capture(monkeypatch, {"type": "b64", "value": base64.b64encode(b"mp4").decode()})
    spec = VideoSpec(seconds=5, resolution=720, aspect_ratio="9:16", dialogue="hi", music=False)
    InfrelayClient("http://gw").video(ModelRef("fal"), "walk", spec)
    payload = sent[0]["input"]
    assert (payload["dialogue"], payload["music"]) == ("hi", False)
    assert "model" not in payload  # no model -> provider default


def test_complete_is_text_without_the_usage(monkeypatch):
    _capture(monkeypatch, {"type": "text", "value": "hello", "meta": {"model": "m"}})
    client = InfrelayClient("http://gw")
    ref = ModelRef("op", "m")
    assert client.complete(ref, "p", max_tokens=5, temperature=0.1) == "hello"
    assert client.text(ref, "", "p", temperature=0.1, max_tokens=5) == ("hello", {"model": "m"})


def test_missing_url_fails_loudly():
    with pytest.raises(InfrelayError, match="INFRELAY_URL is required"):
        InfrelayClient("").complete(ModelRef("op"), "p", max_tokens=1, temperature=0)


# --- GatewayTranscription -------------------------------------------------

class _FailingClient:
    def transcribe(self, audio, ref, **kwargs):
        raise InfrelayError("gateway down")


def test_gateway_failure_surfaces_as_transcription_error():
    with pytest.raises(TranscriptionError, match="gateway down"):
        GatewayTranscription(_FailingClient())(b"audio", ModelRef("whisper", "base"))


# --- MomentEngines --------------------------------------------------------

_CONFIG = {"min_length": 20, "max_length": 60, "scoring": {}}


def _engine(config):
    return MomentEngines(InfrelayClient("http://gw"))(config, Context(store=None))


def test_ai_route_builds_the_ai_engine_named_by_route():
    engine = _engine({**_CONFIG, "moment_finder": "auto",
                      "moment_route": {"provider": "cloud", "model": "auto"}})
    assert isinstance(engine, AiMomentEngine)
    assert engine.name == "cloud/auto"


def test_offline_finder_builds_the_offline_engine_even_with_a_route():
    engine = _engine({**_CONFIG, "moment_finder": "offline",
                      "moment_route": {"provider": "cloud", "model": "auto"}})
    assert isinstance(engine, OfflineMomentEngine)


def test_ai_without_route_warns_loudly_and_falls_back():
    logger = build_logger(segment="clips")
    token = bind(logger)
    try:
        engine = _engine({**_CONFIG, "moment_finder": "ai"})
    finally:
        reset(token)
    assert isinstance(engine, OfflineMomentEngine)
    assert any(e["level"] == "warning" and "no moment_route" in e["text"] for e in logger.entries)


# --- SourceDurationLimit / catalog ---------------------------------------

def test_source_duration_limit():
    limit = SourceDurationLimit({"limits": {"source_max_seconds": 600}})
    assert limit(300) is None
    assert "over 10 minute limit" in limit(900)
    assert SourceDurationLimit({})(99999) is None


def test_catalog_lists_three_segments_with_paths_and_planned_assembly():
    segments = {s.code_name: s for s in SegmentCatalog.list().segments}
    assert set(segments) == {JobKind.CLIPS, JobKind.STORY, JobKind.SERIES}
    assert segments[JobKind.CLIPS].execute_path == "/v1/segments/clips/execute"
    story = {f.code_name: f.status for f in segments[JobKind.STORY].features}
    assert story["video-assembly"] is SegmentStatus.PLANNED
    assert story["story-writing"] is SegmentStatus.AVAILABLE
