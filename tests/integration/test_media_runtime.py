"""Media runtime (image, clip, music, voice, prompt preview) over a faked Infrelay."""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("httpx")

from kinoforge.schemas import (  # noqa: E402
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
    def generate(self, ref, prompt, spec):
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

    def image(self, ref, prompt, spec):
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

    def image(self, ref, prompt, spec):
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

    def video(self, ref, prompt, spec):
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

    def video(self, ref, prompt, spec):
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


# --- music and voice ------------------------------------------------------

def test_render_music_builds_mood_prompt_and_sizes_bed(monkeypatch):
    seen = {}

    def music(self, ref, prompt, *, seconds):
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

    def speech(self, ref, text, *, voice, language):
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
