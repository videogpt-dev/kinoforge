"""Story media renders over REST: one scene image or character sheet, one scene clip, the music
bed, one voiceover, and a prompt preview. Stateless: the caller owns the project store,
reference-sheet composition, placement, progress and billing."""

from __future__ import annotations

import base64
from dataclasses import replace
from functools import partial
from string import Template
from typing import Any, Dict, Optional

from kinoforge.contract import ImageSpec, ModelRef, VideoSpec
from kinoforge.definitions import DefinitionBundle
from kinoforge.definitions.models import DefinitionKind
from kinoforge.observ import KinoLogger, build_logger, logged
from kinoforge.schemas import (
    ImageRenderRequest,
    MusicRenderRequest,
    StoryPromptPreviewRequest,
    VideoRenderRequest,
    VoiceRenderRequest,
)
from kinoforge.segments.story.media.ceiling import PromptCeiling
from kinoforge.segments.story.media.images import ImagePainter
from kinoforge.segments.story.media.music import MusicComposer
from kinoforge.segments.story.media.videos import ClipMaker
from kinoforge.service.runtimes.presets import BundleImagePresets
from kinoforge.service.settings import ServiceSettings


def _decode(value: Optional[str]) -> Optional[bytes]:
    return base64.b64decode(value) if value else None


def _no_media(*_args: Any) -> bytes:
    return b""


class MediaRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    @classmethod
    def from_env(cls) -> "MediaRuntime":
        return cls(ServiceSettings.from_env())

    @staticmethod
    def _logger(request: Any, segment: str) -> KinoLogger:
        return build_logger(
            job_id=request.project_id, segment=segment,
            idempotency_key=request.idempotency_key, level=request.log_level,
        )

    @staticmethod
    def _bundle(request: Any) -> DefinitionBundle:
        return DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True), engine_version="0.1.0"
        )

    @classmethod
    def _presets(cls, request: Any) -> BundleImagePresets:
        keys = request.preset_keys
        return BundleImagePresets(
            cls._bundle(request), scene_key=keys.scene, portrait_key=keys.portrait,
            negative_key=keys.negative,
        )

    @staticmethod
    def _painter(
        presets: BundleImagePresets, generate: Any, ceiling: PromptCeiling
    ) -> ImagePainter:
        return ImagePainter(
            generate_image=generate, image_scene=presets.image_scene,
            image_character=presets.image_character, image_negative=presets.image_negative,
            ceiling=ceiling,
        )

    @staticmethod
    def _motion(presets: BundleImagePresets, maker: ClipMaker, scene: Dict, story: Dict,
                rec: Dict) -> str:
        """The picked image preset's cinematography line, applied film-wide."""
        return presets.image_motion(
            scene.get("prompt") or "", story.get("style") or "",
            key=maker.preset_key(rec, scene) or None,
        )

    @logged
    def render_image(self, request: ImageRenderRequest) -> Dict[str, object]:
        logger = self._logger(request, "image")
        logger.info(f"image render start ({request.mode})", mode=request.mode)
        ceiling = PromptCeiling(request.prompt_limit)
        painter = self._painter(
            self._presets(request), self._settings.infrelay(request.owner).image, ceiling
        )
        opts = request.options
        spec = ImageSpec(
            aspect_ratio=opts.aspect_ratio, seed=opts.seed, mature=opts.mature,
            reference=_decode(request.reference_b64), enhance=opts.enhance,
            enhance_style=opts.enhance_style,
        )
        story, rec = dict(request.story), dict(request.rec)
        if request.mode == "character":
            prompt = partial(painter.character_prompt, story, rec)
        else:
            prompt = partial(painter.scene_prompt, dict(request.scene), story, rec)
            if opts.apply_negative:
                spec = replace(spec, negative=painter.negative())
        data = painter.generate_for(prompt, ModelRef.from_mapping(request.pick), spec)
        logger.success("image render done", bytes=len(data), observed_limit=ceiling.learned)
        return {
            "image_b64": base64.b64encode(data).decode(),
            "prompt": "",
            "observed_limit": ceiling.learned,
            "logs": logger.entries,
        }

    @logged
    def render_clip(self, request: VideoRenderRequest) -> Dict[str, object]:
        logger = self._logger(request, "video")
        logger.info("video render start")
        ceiling = PromptCeiling(request.prompt_limit)
        maker = ClipMaker(
            generate_video=self._settings.infrelay(request.owner).video, ceiling=ceiling
        )
        story, scene, rec = dict(request.story), dict(request.scene), dict(request.rec)
        opts = request.options
        spec = VideoSpec(
            seconds=opts.seconds, resolution=opts.resolution, aspect_ratio=opts.aspect_ratio,
            mature=opts.mature, image=_decode(request.image_b64), enhance=opts.enhance,
            enhance_style=opts.enhance_style,
        )
        motion = self._motion(self._presets(request), maker, scene, story, rec)
        data = maker.generate_clip(scene, story, motion, ModelRef.from_mapping(request.pick), spec)
        logger.success("video render done", bytes=len(data), observed_limit=ceiling.learned)
        return {
            "video_b64": base64.b64encode(data).decode(),
            "observed_limit": ceiling.learned,
            "logs": logger.entries,
        }

    @logged
    def render_music(self, request: MusicRenderRequest) -> Dict[str, object]:
        logger = self._logger(request, "music")
        logger.info("music render start")
        body = self._bundle(request).require(DefinitionKind.PRESET, request.music_key).body
        composer = MusicComposer(
            generate_music=self._settings.infrelay(request.owner).music,
            music_preset=lambda mood: Template(body).substitute({"mood": mood}).strip(),
        )
        data = composer.compose(dict(request.story), ModelRef.from_mapping(request.pick))
        logger.success("music render done", bytes=len(data))
        return {"music_b64": base64.b64encode(data).decode(), "logs": logger.entries}

    @logged
    def render_voice(self, request: VoiceRenderRequest) -> Dict[str, object]:
        logger = self._logger(request, "voice")
        logger.info("voice render start")
        data = self._settings.infrelay(request.owner).speech(
            ModelRef.from_mapping(request.pick), request.text,
            voice=request.options.voice, language=request.options.language,
        )
        logger.success("voice render done", bytes=len(data))
        return {"audio_b64": base64.b64encode(data).decode(), "logs": logger.entries}

    def preview_prompts(self, request: StoryPromptPreviewRequest) -> Dict[str, str]:
        presets = self._presets(request)
        painter = self._painter(presets, _no_media, PromptCeiling())
        maker = ClipMaker(generate_video=_no_media, ceiling=PromptCeiling())
        story, scene, rec = dict(request.story), dict(request.scene), dict(request.rec)
        return {
            "image_prompt": painter.scene_prompt(scene, story, rec, request.image_limit),
            "negative_prompt": painter.negative(),
            "video_prompt": maker.video_prompt(
                scene, scene.get("shot") or {}, story,
                self._motion(presets, maker, scene, story, rec), request.video_limit,
            ),
        }
