"""Story media renders over REST: one scene image or character sheet, one scene clip, the music
bed, one voiceover, and a prompt preview. Stateless: the caller owns the project store,
reference-sheet composition, placement, progress and billing."""

from __future__ import annotations

import base64
from dataclasses import replace
from functools import partial
from string import Template
from typing import Any

from kinoforge.contract import AITask, ImageSpec, ModelRef, VideoSpec
from kinoforge.definitions import DefinitionBundle
from kinoforge.definitions.models import DefinitionKind
from kinoforge.inference.charges import InferenceCharges
from kinoforge.observ import KinoLogger, build_logger, logged
from kinoforge.schemas import (
    ClipJobRequest,
    ImageRenderRequest,
    MusicRenderRequest,
    StoryPromptPreviewRequest,
    VideoRenderRequest,
    VoiceRenderRequest,
)
from kinoforge.segments.story.media.ceiling import PromptCeiling, limit_from_error
from kinoforge.segments.story.media.images import ImagePainter
from kinoforge.segments.story.media.music import MusicComposer
from kinoforge.segments.story.media.videos import ClipMaker
from kinoforge.service.runtimes.presets import BundleImagePresets
from kinoforge.service.settings import ServiceSettings


class MediaUtils:
    @staticmethod
    def decode(value: str | None) -> bytes | None:
        return base64.b64decode(value) if value else None

    @staticmethod
    def no_media(*_args: Any) -> bytes:
        return b""

    @staticmethod
    def music_prompt(body: str, mood: str) -> str:
        """The music preset with its mood filled in; any other `$` is left as written."""
        return Template(body).safe_substitute({"mood": mood}).strip()


class MediaRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    @classmethod
    def from_env(cls) -> MediaRuntime:
        return cls(ServiceSettings.from_env())

    @staticmethod
    def _logger(request: Any, segment: str) -> KinoLogger:
        return build_logger(
            job_id=request.project_id,
            segment=segment,
            idempotency_key=request.idempotency_key,
            level=request.log_level,
        )

    @staticmethod
    def _bundle(request: Any) -> DefinitionBundle:
        return DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0"
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
    def _motion(presets: BundleImagePresets, maker: ClipMaker, scene: dict, story: dict,
                rec: dict) -> str:
        """The picked image preset's cinematography line, applied film-wide."""
        return presets.image_motion(
            scene.get("prompt") or "", story.get("style") or "",
            key=maker.preset_key(rec, scene) or None,
        )

    @InferenceCharges.reported
    @logged
    def render_image(self, request: ImageRenderRequest) -> dict[str, object]:
        logger = self._logger(request, "image")
        logger.info(f"image render start ({request.mode})", mode=request.mode)
        ceiling = PromptCeiling(request.prompt_limit)
        infrelay = self._settings.infrelay(request.owner)
        task = AITask.STORY_CAST if request.mode == "character" else AITask.STORY_IMAGE
        painter = self._painter(self._presets(request), partial(infrelay.image, task), ceiling)
        opts = request.options
        spec = ImageSpec(
            aspect_ratio=opts.aspect_ratio, seed=opts.seed, mature=opts.mature,
            reference=MediaUtils.decode(request.reference_b64), enhance=opts.enhance,
            enhance_style=opts.enhance_style,
        )
        story, rec = dict(request.story), dict(request.rec)
        if request.mode == "character":
            prompt = partial(painter.character_prompt, story, rec)
        else:
            prompt = partial(
                painter.scene_prompt, dict(request.scene), story, rec,
                referenced=spec.reference is not None,
            )
            if opts.apply_negative:
                spec = replace(spec, negative=painter.negative())
        data = painter.generate_for(prompt, ModelRef.from_mapping(request.pick), spec)
        logger.success("image render done", bytes=len(data), observed_limit=ceiling.learned)
        return {
            "image_b64": base64.b64encode(data).decode(),
            "observed_limit": ceiling.learned,
            "enhance": infrelay.rewrites,
            "served": infrelay.served,
            "logs": logger.entries,
        }

    @InferenceCharges.reported
    @logged
    def render_clip(self, request: VideoRenderRequest) -> dict[str, object]:
        logger = self._logger(request, "video")
        logger.info("video render start")
        ceiling = PromptCeiling(request.prompt_limit)
        infrelay = self._settings.infrelay(request.owner)
        maker = ClipMaker(
            generate_video=partial(infrelay.video, AITask.STORY_VIDEO), ceiling=ceiling
        )
        story, scene, rec = dict(request.story), dict(request.scene), dict(request.rec)
        opts = request.options
        spec = VideoSpec(
            seconds=opts.seconds, resolution=opts.resolution, aspect_ratio=opts.aspect_ratio,
            mature=opts.mature, image=MediaUtils.decode(request.image_b64), enhance=opts.enhance,
            enhance_style=opts.enhance_style,
        )
        motion = self._motion(self._presets(request), maker, scene, story, rec)
        data = maker.generate_clip(scene, story, motion, ModelRef.from_mapping(request.pick), spec)
        logger.success("video render done", bytes=len(data), observed_limit=ceiling.learned)
        return {
            "video_b64": base64.b64encode(data).decode(),
            "observed_limit": ceiling.learned,
            "enhance": infrelay.rewrites,
            "served": infrelay.served,
            "logs": logger.entries,
        }

    @logged
    def submit_clip(self, request: ClipJobRequest) -> dict[str, object]:
        """Queues the scene clip on the gateway and returns its job id at once; the caller
        reads it with clip_job. The prompt is built at the configured ceiling; a smaller one
        the provider declares comes back on the job, for the caller to resubmit with."""
        logger = self._logger(request, "video")
        infrelay = self._settings.infrelay(request.owner)
        maker = ClipMaker(generate_video=MediaUtils.no_media, ceiling=PromptCeiling())
        story, scene, rec = dict(request.story), dict(request.scene), dict(request.rec)
        opts = request.options
        spec = VideoSpec(
            seconds=opts.seconds, resolution=opts.resolution, aspect_ratio=opts.aspect_ratio,
            mature=opts.mature, image=MediaUtils.decode(request.image_b64), enhance=opts.enhance,
            enhance_style=opts.enhance_style,
        )
        motion = self._motion(self._presets(request), maker, scene, story, rec)
        prompt, spec = maker.clip_call(scene, story, motion, spec, request.prompt_limit)
        job_id = infrelay.submit_video(
            AITask.STORY_VIDEO, ModelRef.from_mapping(request.pick), prompt, spec,
            request.callback_url,
        )
        logger.info("video job queued", job=job_id)
        return {"job_id": job_id, "logs": logger.entries}

    @InferenceCharges.reported
    def clip_job(self, owner: str, job_id: str) -> dict[str, object]:
        """A queued clip now, with its bytes once it is done."""
        infrelay = self._settings.infrelay(owner)
        job = infrelay.job(job_id)
        out: dict[str, object] = {
            "job_id": job_id,
            "status": job.get("status"),
            "queue_position": job.get("queue_position"),
        }
        if job.get("status") == "done":
            out["video_b64"] = base64.b64encode(infrelay.job_media(job)).decode()
            out["enhance"] = infrelay.rewrites
        elif job.get("error"):
            out["error"] = str(job["error"])
            out["observed_limit"] = limit_from_error(RuntimeError(job["error"]))
        return out

    def cancel_clip_job(self, owner: str, job_id: str) -> dict[str, object]:
        job = self._settings.infrelay(owner).cancel_job(job_id)
        return {"job_id": job_id, "status": job.get("status"), "error": job.get("error") or ""}

    @InferenceCharges.reported
    @logged
    def render_music(self, request: MusicRenderRequest) -> dict[str, object]:
        logger = self._logger(request, "music")
        logger.info("music render start")
        body = self._bundle(request).require(DefinitionKind.PRESET, request.music_key).body
        infrelay = self._settings.infrelay(request.owner)
        composer = MusicComposer(
            generate_music=partial(infrelay.music, AITask.STORY_MUSIC),
            music_preset=partial(MediaUtils.music_prompt, body),
        )
        data = composer.compose(dict(request.story), ModelRef.from_mapping(request.pick))
        logger.success("music render done", bytes=len(data))
        return {
            "music_b64": base64.b64encode(data).decode(),
            "served": infrelay.served,
            "logs": logger.entries,
        }

    @InferenceCharges.reported
    @logged
    def render_voice(self, request: VoiceRenderRequest) -> dict[str, object]:
        logger = self._logger(request, "voice")
        logger.info("voice render start")
        infrelay = self._settings.infrelay(request.owner)
        data = infrelay.speech(
            AITask.STORY_VOICE, ModelRef.from_mapping(request.pick), request.text,
            voice=request.options.voice, language=request.options.language,
        )
        logger.success("voice render done", bytes=len(data))
        return {
            "audio_b64": base64.b64encode(data).decode(),
            "served": infrelay.served,
            "logs": logger.entries,
        }

    def preview_prompts(self, request: StoryPromptPreviewRequest) -> dict[str, str]:
        presets = self._presets(request)
        painter = self._painter(presets, MediaUtils.no_media, PromptCeiling())
        maker = ClipMaker(generate_video=MediaUtils.no_media, ceiling=PromptCeiling())
        story, scene, rec = dict(request.story), dict(request.scene), dict(request.rec)
        return {
            "image_prompt": painter.scene_prompt(scene, story, rec, request.image_limit),
            "negative_prompt": painter.negative(),
            "video_prompt": maker.video_prompt(
                scene, scene.get("shot") or {}, story,
                self._motion(presets, maker, scene, story, rec), request.video_limit,
            ),
        }
