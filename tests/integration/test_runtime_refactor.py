"""Service/inference pieces after the smell cleanup (httpx-gated: they import the client)."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from kinoforge.contract import AITask, ImageSpec, JobKind, ModelRef, VideoSpec  # noqa: E402
from kinoforge.definitions import DefinitionBundle  # noqa: E402
from kinoforge.inference import InfrelayClient, InfrelayError  # noqa: E402
from kinoforge.inference import infrelay as infrelay_module  # noqa: E402
from kinoforge.schemas import SegmentStatus  # noqa: E402
from kinoforge.segments.clips.moments.finders import (  # noqa: E402
    AiMomentFinder,
    MomentFinders,
    OfflineMomentFinder,
)
from kinoforge.segments.clips.transcription import TranscriptionError  # noqa: E402
from kinoforge.service.discovery import SegmentCatalog  # noqa: E402
from kinoforge.service.runtimes.transcription import GatewayTranscriber  # noqa: E402
from kinoforge.service.settings import ServiceSettings  # noqa: E402


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
        sent.append({**json, "url": url})
        return _Response(output)

    monkeypatch.setattr(infrelay_module.InfrelayClient, "_http", SimpleNamespace(post=post))
    return sent


# --- InfrelayClient -------------------------------------------------------

def test_image_payload_comes_from_the_spec(monkeypatch):
    sent = _capture(monkeypatch, {"type": "b64", "value": base64.b64encode(b"png").decode()})
    spec = ImageSpec(aspect_ratio="1:1", seed=7, mature=True, negative="blur",
                     reference=b"ref", enhance=True, enhance_style="cine")
    client = InfrelayClient("http://gw", tenant="t")
    data = client.image(AITask.STORY_IMAGE, ModelRef("fal", "flux"), "a cat", spec)
    assert data == b"png"
    body = sent[0]
    assert (body["url"], body["inference"], body["tenant_id"]) == (
        "http://gw/v1/inference/story.image", {"model": "flux", "provider": "fal"}, "t")
    assert body["input"] == {
        "prompt": "a cat", "aspect_ratio": "1:1", "safe": False, "seed": 7,
        "negative": "blur", "reference_b64": base64.b64encode(b"ref").decode(),
        "enhance": True, "enhance_style": "cine",
    }


def test_video_payload_carries_dialogue_and_music_off(monkeypatch):
    sent = _capture(monkeypatch, {"type": "b64", "value": base64.b64encode(b"mp4").decode()})
    spec = VideoSpec(seconds=5, resolution=720, aspect_ratio="9:16", dialogue="hi", music=False)
    InfrelayClient("http://gw").video(AITask.STORY_VIDEO, ModelRef("fal"), "walk", spec)
    payload = sent[0]["input"]
    assert (payload["dialogue"], payload["music"]) == ("hi", False)
    assert sent[0]["inference"] == {"model": "auto", "provider": "fal"}  # the provider's route


def test_a_pick_with_fallbacks_reaches_the_gateway_as_is(monkeypatch):
    sent = _capture(monkeypatch, {"type": "text", "value": "ok"})
    ref = ModelRef.from_mapping(
        {"model": "gpt-x", "fallback": [{"model": "auto"}, {"provider": "op", "model": "m"}]}
    )
    InfrelayClient("http://gw").complete(AITask.TEXT, ref, "p", max_tokens=5, temperature=0)
    assert sent[0]["inference"] == {
        "model": "gpt-x",
        "provider": "auto",
        "fallback": [{"model": "auto", "provider": "auto"}, {"model": "m", "provider": "op"}],
    }


def test_complete_is_text_without_the_usage(monkeypatch):
    _capture(monkeypatch, {"type": "text", "value": "hello", "meta": {"model": "m"}})
    client = InfrelayClient("http://gw")
    ref = ModelRef("op", "m")
    assert client.complete(AITask.TEXT, ref, "p", max_tokens=5, temperature=0.1) == "hello"
    assert client.text(AITask.TEXT, ref, "", "p", temperature=0.1, max_tokens=5) == (
        "hello", {"model": "m"})


def test_missing_url_fails_loudly():
    with pytest.raises(InfrelayError, match="INFRELAY_URL is required"):
        InfrelayClient("").complete(AITask.TEXT, ModelRef("op"), "p", max_tokens=1, temperature=0)


# --- GatewayTranscriber ---------------------------------------------------

class _FailingClient:
    def transcribe(self, task, audio, ref, **kwargs):
        raise InfrelayError("gateway down")


def test_gateway_failure_surfaces_as_transcription_error():
    with pytest.raises(TranscriptionError, match="gateway down"):
        transcriber = GatewayTranscriber(ServiceSettings(), "", {"model": "base"})
        transcriber._infrelay = _FailingClient()
        transcriber._gateway(AITask.TRANSCRIBE, b"audio", ModelRef("whisper", "base"))


# --- MomentFinders --------------------------------------------------------

_CONFIG = {"min_length": 20, "max_length": 60, "scoring": {}}


def _finder(config):
    return MomentFinders.pick(config, DefinitionBundle.empty(), InfrelayClient("http://gw").complete)


def test_ai_route_builds_the_ai_finder_named_by_route():
    finder = _finder({**_CONFIG, "moment_finder": "auto",
                      "moment_route": {"provider": "cloud", "model": "auto"}})
    assert isinstance(finder, AiMomentFinder)
    assert finder.name == "cloud/auto"


def test_offline_finder_builds_the_offline_finder_even_with_a_route():
    finder = _finder({**_CONFIG, "moment_finder": "offline",
                      "moment_route": {"provider": "cloud", "model": "auto"}})
    assert isinstance(finder, OfflineMomentFinder)


def test_ai_without_route_leaves_the_model_to_the_gateway():
    finder = _finder({**_CONFIG, "moment_finder": "ai"})
    assert isinstance(finder, AiMomentFinder)
    assert finder.name == "auto/auto"


# --- catalog ---------------------------------------------------------------

def test_catalog_lists_three_segments_with_paths_and_planned_assembly():
    segments = {s.code_name: s for s in SegmentCatalog.list().segments}
    assert set(segments) == {JobKind.CLIPS, JobKind.STORY, JobKind.SERIES}
    assert segments[JobKind.CLIPS].execute_path == "/v1/segments/clips/execute"
    story = {f.code_name: f.status for f in segments[JobKind.STORY].features}
    assert story["video-assembly"] is SegmentStatus.PLANNED
    assert story["story-writing"] is SegmentStatus.AVAILABLE


def test_a_clip_is_queued_with_a_callback_url_and_the_video_payload(monkeypatch):
    sent = []

    class Queued:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-1", "status": "queued"}

    def request(method, url, headers, json, timeout):
        sent.append((method, url, json))
        return Queued()

    monkeypatch.setattr(
        infrelay_module.InfrelayClient, "_http", SimpleNamespace(request=request)
    )
    spec = VideoSpec(seconds=10, resolution=720, aspect_ratio="9:16", dialogue="hi", music=False)
    client = InfrelayClient("http://gw", tenant="t")

    job_id = client.submit_video(
        AITask.STORY_VIDEO, ModelRef("beam", "h3"), "a kite", spec, "https://core/hook"
    )
    client.job(job_id)

    (method, url, body), (get, job_url, _) = sent
    assert (method, url, job_id) == ("POST", "http://gw/v1/inference/story.video", "job-1")
    assert (body["inference"], body["tenant_id"]) == ({"model": "h3", "provider": "beam"}, "t")
    assert body["callback_url"] == "https://core/hook"
    assert body["input"]["dialogue"] == "hi" and body["input"]["music"] is False
    assert (get, job_url) == ("GET", "http://gw/v1/jobs/job-1")


def test_a_call_that_runs_only_queued_is_queued_and_read_until_done(monkeypatch):
    from kinoforge.inference.charges import InferenceCharges

    calls = []
    transcript = {"type": "transcript", "value": "hi", "meta": {"segments": [{"text": "hi"}]}}
    replies = {
        "POST": [SimpleNamespace(status_code=409, text="runs only as a queued job",
                                 json=lambda: {})],
        "QUEUE": [SimpleNamespace(status_code=202, json=lambda: {"id": "job-9"})],
        "GET": [
            SimpleNamespace(status_code=429, headers={"Retry-After": "0"}),
            SimpleNamespace(status_code=200, headers={}, json=lambda: {"status": "running"}),
            SimpleNamespace(status_code=200, headers={}, json=lambda: {
                "status": "done", "task": "viralClip.transcribe",
                "result": {"provider": "whisper", "model": "turbo", "kind": "transcribe",
                           "output": transcript, "usage": {"credits": 24, "units": 3}},
            }),
        ],
    }

    def post(url, headers, json, timeout):
        calls.append(("POST", url, json))
        return replies["POST"].pop(0)

    def request(method, url, headers, json, timeout):
        calls.append((method, url, json))
        return replies["QUEUE" if method == "POST" else "GET"].pop(0)

    monkeypatch.setattr(infrelay_module.InfrelayClient, "_http",
                        SimpleNamespace(post=post, request=request))
    monkeypatch.setattr(infrelay_module.InfrelayClient, "POLL_FIRST", 0)
    monkeypatch.setattr(infrelay_module.time, "sleep", lambda seconds: None)

    with InferenceCharges.collect() as charges:
        segments, _meta = InfrelayClient("http://gw").transcribe(
            AITask.VIRAL_CLIP_TRANSCRIBE, b"audio", ModelRef("whisper", "turbo")
        )

    assert segments == [{"text": "hi"}]
    assert calls[1][2]["queue"] is True
    assert [c[0] for c in calls] == ["POST", "POST", "GET", "GET", "GET"]
    assert charges == [{"task": "viralClip.transcribe", "kind": "transcribe", "provider": "whisper",
                        "model": "turbo", "credits": 24, "byok": False, "units": 3}]


def test_every_answered_call_is_reported_with_its_price(monkeypatch):
    from kinoforge.inference.charges import InferenceCharges

    def post(url, headers, json, timeout):
        return SimpleNamespace(status_code=200, text="", json=lambda: {
            "provider": "videogpt", "model": "story-auto", "kind": "text",
            "output": {"type": "text", "value": "hello"}, "usage": {"credits": 2, "byok": False},
        })

    monkeypatch.setattr(infrelay_module.InfrelayClient, "_http", SimpleNamespace(post=post))

    @InferenceCharges.reported
    def run() -> dict:
        client = InfrelayClient("http://gw")
        client.complete(AITask.STORY_WRITE, ModelRef(), "p", max_tokens=5, temperature=0)
        client.complete(AITask.STORY_IDEAS, ModelRef(), "q", max_tokens=5, temperature=0)
        return {"ok": True}

    out = run()

    assert out["ok"] is True
    assert [(c["task"], c["credits"]) for c in out["inference"]] == [
        ("story.write", 2), ("story.ideas", 2)]
