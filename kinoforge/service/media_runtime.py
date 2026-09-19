"""Cloud-called image kernel over REST.

Renders one scene image or a character reference sheet: builds the prompt from the frozen
image presets, calls the gateway, and learns one provider prompt-length ceiling. Stateless
by design, the caller owns the project store, reference-sheet composition, placement,
progress and billing; the engine only turns a resolved request into image bytes.
"""

from __future__ import annotations

import base64
import os
import re
from functools import partial
from string import Template
from typing import Dict, Optional

from kinoforge.definitions import DefinitionBundle
from kinoforge.definitions.models import DefinitionKind
from kinoforge.service.infrelay import InfrelayClient
from kinoforge.service.media_presets import BundleImagePresets
from kinoforge.service.models import (
    ImageRenderRequest,
    MusicRenderRequest,
    StoryPromptPreviewRequest,
    VideoRenderRequest,
    VoiceRenderRequest)
from kinoforge.story.media.images import ImagePainter
from kinoforge.story.media.music import MusicComposer
from kinoforge.story.media.videos import ClipMaker

_VALIDATION_LIMIT = re.compile(
    r"input\.prompt.*?(?:less than or equal to|maximum|max(?:imum)? length)\D+(?P<limit>\d+)",
    re.IGNORECASE | re.DOTALL)


def _limit_from_error(error: Exception) -> int:
    match = _VALIDATION_LIMIT.search(str(error))
    if not match:
        return 0
    try:
        value = int(match.group("limit"))
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


class MediaRuntime:
    def __init__(self, infrelay_url: str, infrelay_token: str) -> None:
        self._infrelay_url = infrelay_url
        self._infrelay_token = infrelay_token

    def render_image(self, request: ImageRenderRequest) -> Dict[str, object]:
        infrelay = InfrelayClient(self._infrelay_url, self._infrelay_token, request.owner)
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0")
        presets = BundleImagePresets(
            bundle,
            scene_key=request.preset_keys.scene,
            portrait_key=request.preset_keys.portrait,
            negative_key=request.preset_keys.negative)
        observed: Dict[str, int] = {"limit": 0}
        painter = ImagePainter(
            generate_image=infrelay.image,
            image_scene=presets.image_scene,
            image_character=presets.image_character,
            image_negative=presets.image_negative,
            prompt_limit_configured=lambda _p, _k, _m: request.prompt_limit,
            prompt_limit_from_error=_limit_from_error,
            prompt_limit_remember=lambda _p, _k, _m, limit: observed.__setitem__("limit", limit))

        reference: Optional[bytes] = (
            base64.b64decode(request.reference_b64) if request.reference_b64 else None
        )
        cfg = {
            "provider": str(request.pick.get("provider") or ""),
            "model": str(request.pick.get("model") or ""),
        }
        opts = request.options
        story = dict(request.story)
        rec = dict(request.rec)

        if request.mode == "character":
            data = painter.generate_for(
                lambda limit: painter.character_prompt(story, rec, limit) or "",
                cfg,
                reference=reference,
                aspect_ratio=opts.aspect_ratio,
                mature=opts.mature,
                seed=opts.seed)
        else:
            scene = dict(request.scene)
            negative = painter.negative() if opts.apply_negative else ""
            data = painter.generate_for(
                partial(painter.scene_prompt, scene, story, rec),
                cfg,
                reference=reference,
                aspect_ratio=opts.aspect_ratio,
                mature=opts.mature,
                seed=opts.seed,
                negative=negative)

        return {
            "image_b64": base64.b64encode(data).decode(),
            "prompt": "",
            "observed_limit": observed["limit"],
            "logs": [],
        }

    def render_clip(self, request: VideoRenderRequest) -> Dict[str, object]:
        infrelay = InfrelayClient(self._infrelay_url, self._infrelay_token, request.owner)
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0")
        presets = BundleImagePresets(
            bundle,
            scene_key=request.preset_keys.scene,
            portrait_key=request.preset_keys.portrait,
            negative_key=request.preset_keys.negative)
        observed: Dict[str, int] = {"limit": 0}
        maker = ClipMaker(
            generate_video=infrelay.video,
            prompt_limit_configured=lambda _p, _k, _m: request.prompt_limit,
            prompt_limit_from_error=_limit_from_error,
            prompt_limit_remember=lambda _p, _k, _m, limit: observed.__setitem__("limit", limit))

        story = dict(request.story)
        scene = dict(request.scene)
        rec = dict(request.rec)
        cfg = {
            "provider": str(request.pick.get("provider") or ""),
            "model": str(request.pick.get("model") or ""),
        }
        image: Optional[bytes] = (
            base64.b64decode(request.image_b64) if request.image_b64 else None
        )
        # The cinematography line is a property of the picked image preset, applied film-wide.
        motion = presets.image_motion(
            scene.get("prompt") or "",
            story.get("style") or "",
            key=maker.preset_key(rec, scene) or None)
        opts = request.options
        data = maker.generate_clip(
            scene,
            story,
            motion,
            cfg,
            image=image,
            seconds=opts.seconds,
            resolution=opts.resolution,
            aspect_ratio=opts.aspect_ratio,
            mature=opts.mature)
        return {
            "video_b64": base64.b64encode(data).decode(),
            "observed_limit": observed["limit"],
            "logs": [],
        }

    def render_music(self, request: MusicRenderRequest) -> Dict[str, object]:
        infrelay = InfrelayClient(self._infrelay_url, self._infrelay_token, request.owner)
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0")
        body = bundle.require(DefinitionKind.PRESET, request.music_key).body

        def music_preset(mood: str) -> str:
            return Template(body).substitute({"mood": mood}).strip()

        composer = MusicComposer(
            # The gateway sizes nothing per-project; drop the billing tag the in-process
            # provider carried (the caller bills around this call).
            generate_music=lambda prompt, provider, model, *, seconds, project="": infrelay.music(
                prompt, provider, model, seconds=seconds
            ),
            music_preset=music_preset)
        cfg = {
            "provider": str(request.pick.get("provider") or ""),
            "model": str(request.pick.get("model") or ""),
        }
        data = composer.compose(dict(request.story), cfg)
        return {"music_b64": base64.b64encode(data).decode(), "logs": []}

    def render_voice(self, request: VoiceRenderRequest) -> Dict[str, object]:
        infrelay = InfrelayClient(self._infrelay_url, self._infrelay_token, request.owner)
        data = infrelay.speech(
            request.text,
            str(request.pick.get("provider") or ""),
            str(request.pick.get("model") or ""),
            voice=request.options.voice,
            language=request.options.language)
        return {"audio_b64": base64.b64encode(data).decode(), "logs": []}

    def preview_prompts(self, request: StoryPromptPreviewRequest) -> Dict[str, str]:
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0")
        presets = BundleImagePresets(
            bundle,
            scene_key=request.preset_keys.scene,
            portrait_key=request.preset_keys.portrait,
            negative_key=request.preset_keys.negative)
        painter = ImagePainter(
            generate_image=lambda *_args, **_kwargs: b"",
            image_scene=presets.image_scene,
            image_character=presets.image_character,
            image_negative=presets.image_negative,
            prompt_limit_configured=lambda *_args: 0,
            prompt_limit_from_error=lambda _error: 0,
            prompt_limit_remember=lambda *_args: None)
        maker = ClipMaker(
            generate_video=lambda *_args, **_kwargs: b"",
            prompt_limit_configured=lambda *_args: 0,
            prompt_limit_from_error=lambda _error: 0,
            prompt_limit_remember=lambda *_args: None)
        story = dict(request.story)
        scene = dict(request.scene)
        rec = dict(request.rec)
        image_prompt = painter.scene_prompt(scene, story, rec, request.image_limit)
        motion = presets.image_motion(
            scene.get("prompt") or "",
            story.get("style") or "",
            key=maker.preset_key(rec, scene) or None)
        return {
            "image_prompt": image_prompt,
            "negative_prompt": painter.negative(),
            "video_prompt": maker.video_prompt(
                scene,
                scene.get("shot") or {},
                story,
                motion,
                request.video_limit),
        }


def media_runtime() -> MediaRuntime:
    return MediaRuntime(
        os.getenv("INFRELAY_URL") or "",
        os.getenv("INFRELAY_SERVICE_TOKEN") or "")
