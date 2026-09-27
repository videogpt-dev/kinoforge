from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from kinoforge.contract import Context, Job, JobKind, Result
from kinoforge.segments.clips import pipeline as stages
from kinoforge.segments.clips.pipeline import (
    ClipStage,
    PipelineLogger,
    SaveProject,
    StageCtx,
    StageReporter,
    StageStatus,
    TranscribeVideo,
)

DurationChecker = Callable[[float], Optional[str]]
MomentProviderFactory = Callable[[Dict[str, Any], Context], object]
CancellationChecker = Callable[[], bool]


def _not_cancelled() -> bool:
    return False


class ClipsRunner:
    """Wires the injected host services to the clip pipeline and drives its stages in order.
    Stage logic lives in kinoforge.segments.clips.pipeline; this class only orchestrates."""

    kind = JobKind.CLIPS

    def __init__(
        self,
        logger: PipelineLogger,
        report_stage: StageReporter,
        check_source_duration: DurationChecker,
        transcribe_video: TranscribeVideo,
        save_project: SaveProject,
        moment_provider: MomentProviderFactory,
        is_cancelled: Optional[CancellationChecker] = None,
    ) -> None:
        self._logger = logger
        self._report_stage = report_stage
        self._check_source_duration = check_source_duration
        self._transcribe_video = transcribe_video
        self._save_project = save_project
        self._moment_provider = moment_provider
        self._is_cancelled = is_cancelled or _not_cancelled

    def run(self, job: Job, ctx: Context) -> Result:
        if job.kind != self.kind:
            raise ValueError(f"ClipsRunner cannot run {job.kind}")
        raw_video_path = job.input.get("video_path")
        raw_audio_path = job.input.get("audio_path")
        if not raw_video_path and not raw_audio_path:
            raise ValueError("clips job requires input.video_path or input.audio_path")
        video_path = Path(raw_video_path) if raw_video_path else None
        audio_path = Path(raw_audio_path) if raw_audio_path else None

        config = {**ctx.config, **job.options}
        return self._process(video_path, audio_path, job.job_id, config, ctx)

    def _process(
        self,
        video_path: Optional[Path],
        audio_path: Optional[Path],
        job_id: str,
        config: Dict[str, Any],
        ctx: Context,
    ) -> Result:
        # The find stages read the audio when the host sent it, else the video; rendering uses
        # video_path directly and is guarded in pipeline.render_clips.
        media_path = audio_path or video_path
        if media_path is None:
            raise ValueError("clips job requires input.video_path or input.audio_path")
        result = Result(
            data={
                "video_path": str(video_path) if video_path else "",
                "audio_path": str(audio_path) if audio_path else "",
                "clips": [],
                "moments": [],
                "transcript": None,
                "errors": [],
            }
        )
        sc = StageCtx(result, self._logger, self._report_stage, self._is_cancelled)
        video_out = ctx.store.workdir(job_id)
        slug = config.get("slug", "")
        provider = self._moment_provider(config, ctx)

        try:
            if sc.stopped():
                return result
            self._logger.info(f"Starting: {media_path.name}")

            if stages.already_processed(ctx.store, job_id, slug, config):
                self._logger.info("Skipping: already has clips (skip_already_processed)")
                result.status = "skipped"
                return result

            if not stages.duration_ok(sc, self._check_source_duration, media_path):
                return result

            transcript = stages.transcribe(
                sc, self._transcribe_video, media_path, video_out, job_id, config, ctx
            )
            if transcript is None:
                return result

            found = stages.find_moments(sc, provider, transcript, media_path, config)
            if found is None:
                return result
            moments, discovered = found
            if not moments:
                sc.stage(ClipStage.MOMENTS, StageStatus.DONE)
                self._logger.warning("No moments extracted from video")
                return result

            ranked = stages.rank_moments(sc, provider, moments, discovered, transcript)
            ranked = stages.apply_limits(sc, ranked, config)
            used_moments = ranked[: config["clip_count"]]
            result.data["used_moments"] = used_moments

            created_ids = stages.persist_moments(
                sc, self._save_project, video_path or media_path, slug, used_moments,
                transcript, job_id, config, ctx,
            )
            sc.stage(ClipStage.MOMENTS, StageStatus.DONE)

            if config.get("analyze_only"):
                sc.stage(ClipStage.CLIPS, StageStatus.SKIPPED)
                self._logger.success("Analyze-only: skipping clip extraction (open in editor)")
                return result

            if sc.stopped():
                return result

            stages.render_clips(
                sc, video_path, used_moments, transcript, created_ids, video_out, job_id, config, ctx
            )
            return result

        except Exception as exc:
            self._logger.error(f"Unexpected error: {exc}")
            sc.fail(f"Unexpected error: {exc!s}")
            return result
