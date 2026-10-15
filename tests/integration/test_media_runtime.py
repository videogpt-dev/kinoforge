"""Media runtime (image, clip, music, voice, prompt preview) over a faked Infrelay."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from kinoforge.schemas import (  # noqa: E402
    ClipJobRequest,
    ImageRenderRequest,
    MusicRenderRequest,
    StoryPromptPreviewRequest,
    VideoRenderRequest,
    VoiceRenderRequest,
)
from kinoforge.service.runtimes.media import MediaRuntime  # noqa: E402
from kinoforge.service.settings import ServiceSettings  # noqa: E402
from tests.integration.runtime_support import SETTINGS, bundle, fake, preset  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"
MP4 = b"\x00\x00\x00\x18ftypmp42fake-clip"
MP3 = b"ID3fake-audio-bytes"
PRESET_KEYS = {
    "scene": "prompts.presets.image.default",
    "portrait": "prompts.presets.image.portrait",
    "negative": "prompts.presets.image.negative",
}


def _image_bundle() -> dict:
    return bundle(
        preset(PRESET_KEYS["scene"], "$prompt in $style style", motion="slow push in, $style"),
        preset(PRESET_KEYS["portrait"], "Portrait of $who, $style"),
        preset(PRESET_KEYS["negative"], "blurry, low quality"),
    )


def _failing_once(calls: dict, message: str, data: bytes):
    def generate(self, task, ref, prompt, spec):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(message)
        return data
    return generate


# --- images ---------------------------------------------------------------

def _image_request(mode: str, **over) -> ImageRenderRequest:
    base = {
        "job_id": "job-1", "project_id": "proj-1", "owner": "tenant-1", "mode": mode,
        "story": {"style": "noir", "characters": [], "scenes": []},
        "scene": {"prompt": "a lighthouse"}, "rec": {"input": {}},
        "preset_keys": PRESET_KEYS, "pick": {"provider": "fal", "model": "flux"},
        "options": {"aspect_ratio": "9:16", "mature": False, "seed": 7, "apply_negative": True},
        "definitions": _image_bundle(),
    }
    return ImageRenderRequest(**{**base, **over})


def test_render_image_scene_builds_prompt_and_negative(monkeypatch):
    seen = {}

    def image(self, task, ref, prompt, spec):
        seen.update(ref=ref, prompt=prompt, spec=spec)
        return PNG

    fake(monkeypatch, "image", image)
    out = MediaRuntime(SETTINGS).render_image(_image_request("scene"))

    assert seen["prompt"] == "a lighthouse in noir style"
    assert str(seen["ref"]) == "fal/flux"
    assert (seen["spec"].negative, seen["spec"].reference, seen["spec"].seed) == (
        "blurry, low quality", None, 7)
    assert base64.b64decode(out["image_b64"]) == PNG
    assert out["observed_limit"] == 0


def test_render_image_character_uses_portrait_without_negative(monkeypatch):
    seen = {}

    def image(self, task, ref, prompt, spec):
        seen.update(prompt=prompt, negative=spec.negative)
        return PNG

    fake(monkeypatch, "image", image)
    story = {"style": "noir", "characters": [{"name": "Ana", "description": "a weathered keeper"}]}
    MediaRuntime(SETTINGS).render_image(_image_request("character", story=story))

    assert seen["prompt"].startswith("Portrait of Ana: a weathered keeper")
    assert not seen["negative"]


def test_render_image_learns_prompt_ceiling_on_retry(monkeypatch):
    calls = {"n": 0}
    message = "input.prompt must be less than or equal to 500 characters"
    fake(monkeypatch, "image", _failing_once(calls, message, PNG))

    out = MediaRuntime(SETTINGS).render_image(_image_request("scene", prompt_limit=4000))

    assert calls["n"] == 2
    assert out["observed_limit"] == 500
    assert base64.b64decode(out["image_b64"]) == PNG


def test_story_prompt_preview_runs_without_inference():
    request = StoryPromptPreviewRequest(
        owner="tenant-1", story={"style": "noir", "characters": [], "scenes": []},
        scene={"prompt": "a lighthouse", "shot": {"action": "lamp rotates"}},
        rec={"input": {}}, preset_keys=PRESET_KEYS, definitions=_image_bundle(),
    )
    out = MediaRuntime(ServiceSettings()).preview_prompts(request)

    assert out["image_prompt"] == "a lighthouse in noir style"
    assert out["negative_prompt"] == "blurry, low quality"
    assert "lamp rotates" in out["video_prompt"]


# --- video ----------------------------------------------------------------

def _video_request(**over) -> VideoRenderRequest:
    base = {
        "job_id": "job-1", "project_id": "proj-1", "owner": "tenant-1",
        "story": {"style": "noir", "characters": [{"name": "Ana", "description": "the keeper"}]},
        "scene": {
            "prompt": "the lighthouse",
            "shot": {"action": "lamp sweeps", "camera": "wide", "continuity": "night"},
        },
        "rec": {"input": {}}, "preset_keys": PRESET_KEYS,
        "pick": {"provider": "fal", "model": "seedance"},
        "options": {"seconds": 5, "resolution": 720, "aspect_ratio": "9:16", "mature": False},
        "definitions": _image_bundle(),
    }
    return VideoRenderRequest(**{**base, **over})


def test_render_clip_builds_prompt_with_shot_and_motion(monkeypatch):
    seen = {}

    def video(self, task, ref, prompt, spec):
        seen.update(prompt=prompt, spec=spec)
        return MP4

    fake(monkeypatch, "video", video)
    out = MediaRuntime(SETTINGS).render_clip(_video_request())

    assert "lamp sweeps" in seen["prompt"] and "wide" in seen["prompt"]
    assert "slow push in, noir" in seen["prompt"]
    assert (seen["spec"].seconds, seen["spec"].image) == (5, None)
    assert base64.b64decode(out["video_b64"]) == MP4


def test_render_clip_learns_prompt_ceiling_on_retry(monkeypatch):
    calls = {"n": 0}
    message = "input.prompt: String length must be less than or equal to 2000"
    fake(monkeypatch, "video", _failing_once(calls, message, MP4))

    out = MediaRuntime(SETTINGS).render_clip(_video_request(prompt_limit=6000))

    assert calls["n"] == 2
    assert out["observed_limit"] == 2000


def test_video_prompt_compacts_to_the_ceiling_and_keeps_the_shot(monkeypatch):
    prompts = []

    def video(self, task, ref, prompt, spec):
        prompts.append(prompt)
        return MP4

    fake(monkeypatch, "video", video)
    story = {
        "style": "noir",
        "characters": [
            {"name": "Ana", "description": "the keeper. " * 30},
            {"name": "Bruno", "description": "the smuggler. " * 30},
        ],
    }
    MediaRuntime(SETTINGS).render_clip(_video_request(story=story, prompt_limit=400))

    assert len(prompts[0]) <= 400
    assert "lamp sweeps" in prompts[0]
    assert "Bruno" not in prompts[0]


def test_a_queued_clip_is_submitted_with_its_prompt_and_read_back_when_done(monkeypatch):
    seen = {}

    def submit_video(self, task, ref, prompt, spec, callback_url):
        seen.update(ref=ref, prompt=prompt, spec=spec, callback=callback_url)
        return "job-9"

    fake(monkeypatch, "submit_video", submit_video)
    fake(
        monkeypatch,
        "job",
        lambda self, job_id: {
            "id": job_id,
            "status": "done",
            "result": {
                "output": {"type": "b64", "value": base64.b64encode(MP4).decode()},
                "usage": {"enhance": {"provider": "openrouter", "model": "m", "credits": 1}},
            },
        },
    )
    runtime = MediaRuntime(SETTINGS)

    request = ClipJobRequest(
        **_video_request().model_dump(), callback_url="https://core/internal/clip-jobs"
    )
    queued = runtime.submit_clip(request)
    done = runtime.clip_job("tenant-1", queued["job_id"])

    assert queued["job_id"] == "job-9"
    assert seen["callback"] == "https://core/internal/clip-jobs"
    assert "lamp sweeps" in seen["prompt"] and seen["spec"].seconds == 5
    assert (done["status"], base64.b64decode(done["video_b64"])) == ("done", MP4)
    assert done["enhance"] == [{"provider": "openrouter", "model": "m", "credits": 1}]


def test_a_failed_clip_job_reports_the_prompt_ceiling_the_provider_declared(monkeypatch):
    error = "input.prompt: String length must be less than or equal to 2000"
    fake(
        monkeypatch,
        "job",
        lambda self, job_id: {"id": job_id, "status": "error", "error": error},
    )

    failed = MediaRuntime(SETTINGS).clip_job("tenant-1", "job-9")

    assert (failed["status"], failed["observed_limit"]) == ("error", 2000)
    assert "video_b64" not in failed


# --- music and voice ------------------------------------------------------

def test_render_music_builds_mood_prompt_and_sizes_bed(monkeypatch):
    seen = {}

    def music(self, task, ref, prompt, *, seconds):
        seen.update(ref=ref, prompt=prompt, seconds=seconds)
        return MP3

    fake(monkeypatch, "music", music)
    request = MusicRenderRequest(
        job_id="job-1", project_id="proj-1", owner="tenant-1",
        story={
            "style": "moody coastal noir",
            "scenes": [{"prompt": "a", "narration": "one"}, {"prompt": "b", "narration": "two"}],
        },
        music_key="prompts.presets.music.default",
        pick={"provider": "fal", "model": "musicgen"},
        definitions=bundle(
            preset("prompts.presets.music.default", "instrumental score, mood: $mood")
        ),
    )
    out = MediaRuntime(SETTINGS).render_music(request)

    assert seen["prompt"] == "instrumental score, mood: moody coastal noir"
    assert seen["ref"].provider == "fal"
    assert 5 <= seen["seconds"] <= 30
    assert base64.b64decode(out["music_b64"]) == MP3


def test_render_voice_forwards_resolved_route(monkeypatch):
    seen = {}

    def speech(self, task, ref, text, *, voice, language):
        seen.update(ref=str(ref), text=text, voice=voice, language=language)
        return MP3

    fake(monkeypatch, "speech", speech)
    request = VoiceRenderRequest(
        job_id="voice-1", project_id="proj-1", owner="tenant-1", text="The light still turns.",
        pick={"provider": "edge", "model": "en-US-AriaNeural"},
        options={"voice": "en-US-AriaNeural", "language": "en"},
    )
    out = MediaRuntime(SETTINGS).render_voice(request)

    assert base64.b64decode(out["audio_b64"]) == MP3
    assert seen == {
        "ref": "edge/en-US-AriaNeural", "text": "The light still turns.",
        "voice": "en-US-AriaNeural", "language": "en",
    }


def test_render_reports_each_prompt_rewrite_infrelay_ran(monkeypatch):
    import httpx

    from kinoforge.inference import infrelay as infrelay_module

    rewrites = iter([{"provider": "openrouter", "model": "m", "credits": 2}, None])

    def post(url, headers, json, timeout):
        usage = {"credits": 3, "enhance": next(rewrites)}
        output = {"type": "b64", "value": base64.b64encode(PNG).decode()}
        return httpx.Response(200, json={"output": output, "usage": usage})

    monkeypatch.setattr(infrelay_module.InfrelayClient, "_http", SimpleNamespace(post=post))
    runtime = MediaRuntime(SETTINGS)
    options = {"aspect_ratio": "9:16", "seed": 7, "enhance": True}
    first = runtime.render_image(_image_request("scene", options=options))
    cached = runtime.render_image(_image_request("scene", options=options))
    assert first["enhance"] == [{"provider": "openrouter", "model": "m", "credits": 2}]
    assert cached["enhance"] == []


def _answering(monkeypatch, sent: list, provider: str, model: str) -> None:
    import httpx

    from kinoforge.inference import infrelay as infrelay_module

    def post(url, headers, json, timeout):
        sent.append((url, json))
        output = {"type": "b64", "value": base64.b64encode(PNG).decode()}
        return httpx.Response(
            200, json={"provider": provider, "model": model, "output": output, "usage": {}}
        )

    monkeypatch.setattr(infrelay_module.InfrelayClient, "_http", SimpleNamespace(post=post))


def test_an_auto_pick_lets_infrelay_route_it_and_reports_the_step_that_answered(monkeypatch):
    sent: list = []
    _answering(monkeypatch, sent, "pollinations", "z-image-turbo")

    out = MediaRuntime(SETTINGS).render_image(_image_request("scene", pick={}))

    [(url, body)] = sent
    assert url == "http://infrelay/v1/inference/story.image"
    assert body["inference"] == {"model": "auto", "provider": "auto"}
    assert "model" not in body["input"]  # each route step names its own
    assert out["served"] == {"provider": "pollinations", "model": "z-image-turbo"}


def test_a_character_sheet_is_the_cast_task(monkeypatch):
    sent: list = []
    _answering(monkeypatch, sent, "fal", "flux")
    story = {"style": "noir", "characters": [{"name": "Ana", "description": "a keeper"}]}

    MediaRuntime(SETTINGS).render_image(_image_request("character", story=story))

    assert sent[0][0] == "http://infrelay/v1/inference/story.cast"


def test_a_pinned_pick_runs_only_that_model(monkeypatch):
    sent: list = []
    _answering(monkeypatch, sent, "fal", "flux")

    out = MediaRuntime(SETTINGS).render_image(_image_request("scene"))

    [(url, body)] = sent
    assert url == "http://infrelay/v1/inference/story.image"
    assert body["inference"] == {"model": "flux", "provider": "fal"}
    assert out["served"] == {"provider": "fal", "model": "flux"}


def test_an_auto_music_pick_reports_what_served_it(monkeypatch):
    sent: list = []
    _answering(monkeypatch, sent, "replicate", "riffusion")
    request = MusicRenderRequest(
        job_id="job-1", project_id="proj-1", owner="tenant-1",
        story={"style": "noir", "scenes": [{"prompt": "a", "narration": "one"}]},
        music_key="prompts.presets.music.default",
        pick={"model": "auto"},
        definitions=bundle(preset("prompts.presets.music.default", "score, mood: $mood")),
    )

    out = MediaRuntime(SETTINGS).render_music(request)

    assert sent[0][0] == "http://infrelay/v1/inference/story.music"
    assert out["served"] == {"provider": "replicate", "model": "riffusion"}


def test_a_scene_with_a_reference_is_told_what_the_reference_is_for(monkeypatch):
    seen = {}

    def image(self, task, ref, prompt, spec):
        seen["prompt"] = prompt
        return PNG

    fake(monkeypatch, "image", image)
    runtime = MediaRuntime(SETTINGS)
    runtime.render_image(_image_request("scene"))
    assert "reference image" not in seen["prompt"]
    sheet = base64.b64encode(b"sheet").decode()
    runtime.render_image(_image_request("scene", reference_b64=sheet))
    assert seen["prompt"].startswith("a lighthouse. Match the reference image")
