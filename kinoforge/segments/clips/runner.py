from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kinoforge.contract import Context, Job, JobKind, Result
from kinoforge.segments.clips import pipeline as stages
from kinoforge.segments.clips.pipeline import (
    ClipRun,
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


def _path(value: Any) -> Optional[Path]:
    return Path(value) if value else None


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
        video_path = _path(job.input.get("video_path"))
        audio_path = _path(job.input.get("audio_path"))
        if video_path is None and audio_path is None:
            raise ValueError("clips job requires input.video_path or input.audio_path")
        result = Result(data={
            "video_path": str(video_path or ""), "audio_path": str(audio_path or ""),
            "clips": [], "moments": [], "transcript": None, "errors": [],
        })
        run = ClipRun(
            sc=StageCtx(result, self._logger, self._report_stage, self._is_cancelled),
            ctx=ctx, job_id=job.job_id, config={**ctx.config, **job.options},
            video_path=video_path, audio_path=audio_path, workdir=ctx.store.workdir(job.job_id),
        )
        try:
            self._stages(run)
        except Exception as exc:
            self._logger.error(f"Unexpected error: {exc}")
            run.sc.fail(f"Unexpected error: {exc!s}")
        return result

    def _stages(self, run: ClipRun) -> None:
        sc = run.sc
        if sc.stopped():
            return
        self._logger.info(f"Starting: {run.media_path.name}")
        if stages.already_processed(run):
            self._logger.info("Skipping: already has clips (skip_already_processed)")
            sc.result.status = "skipped"
            return
        if not stages.duration_ok(sc, self._check_source_duration, run.media_path):
            return
        if stages.transcribe(run, self._transcribe_video) is None:
            return
        used = self._select(run)
        if used is None:
            return
        created_ids = stages.persist_moments(run, self._save_project, used)
        sc.stage(ClipStage.MOMENTS, StageStatus.DONE)
        if run.config.get("analyze_only"):
            sc.stage(ClipStage.CLIPS, StageStatus.SKIPPED)
            self._logger.success("Analyze-only: skipping clip extraction (open in editor)")
            return
        if not sc.stopped():
            stages.render_clips(run, used, created_ids)

    def _select(self, run: ClipRun) -> Optional[List[Dict[str, Any]]]:
        """Find, rank and limit moments; the ones to use, or None when the run ends here."""
        provider = self._moment_provider(run.config, run.ctx)
        found = stages.find_moments(run, provider)
        if found is None:
            return None
        moments, discovered = found
        if not moments:
            run.sc.stage(ClipStage.MOMENTS, StageStatus.DONE)
            self._logger.warning("No moments extracted from video")
            return None
        ranked = stages.rank_moments(run, provider, moments, discovered)
        used = stages.apply_limits(run.sc, ranked, run.config)[: run.config["clip_count"]]
        run.sc.result.data["used_moments"] = used
        return used
