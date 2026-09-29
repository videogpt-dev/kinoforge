from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from kinoforge.contract import Context, Result
from kinoforge.segments.clips.render.probe import get_video_metadata


class ClipStage(StrEnum):
    TRANSCRIBE = "transcribe"
    MOMENTS = "moments"
    CLIPS = "clips"


class StageStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineLogger(Protocol):
    def info(self, text: str) -> None: ...
    def success(self, text: str) -> None: ...
    def warning(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...


StageReporter = Callable[[ClipStage, StageStatus], None]
CancellationChecker = Callable[[], bool]
TranscribeVideo = Callable[..., List[Dict[str, Any]]]
SaveProject = Callable[..., None]


class StageCtx:
    """Cross-cutting state the pipeline stages share: the Result being built, the logger, stage
    reporting, and cancellation. Keeps the stage functions free of runner plumbing."""

    def __init__(
        self,
        result: Result,
        logger: PipelineLogger,
        report_stage: StageReporter,
        is_cancelled: CancellationChecker,
    ) -> None:
        self.result = result
        self.logger = logger
        self._report = report_stage
        self._is_cancelled = is_cancelled

    def stage(self, stage: ClipStage, status: StageStatus) -> None:
        self._report(stage, status)
        self.result.stages.append(f"{stage.value}:{status.value}")

    def fail(self, message: str) -> None:
        self.result.status = "failed"
        self.result.error = message
        self.result.data["errors"].append(message)

    def stopped(self, stage: Optional[ClipStage] = None) -> bool:
        if not self._is_cancelled():
            return False
        if stage is not None:
            self.stage(stage, StageStatus.CANCELLED)
        self.result.status = "cancelled"
        self.result.error = ""
        self.result.data["cancelled"] = True
        self.logger.warning("Execution cancelled")
        return True


@dataclass
class ClipRun:
    """One clips execution: the stage context plus what every stage reads (job, config,
    inputs, working dir) and the transcript once produced."""

    sc: StageCtx
    ctx: Context
    job_id: str
    config: Dict[str, Any]
    video_path: Optional[Path]
    audio_path: Optional[Path]
    workdir: Path
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def media_path(self) -> Path:
        """What the find stages read: the audio when the host sent it, else the video."""
        path = self.audio_path or self.video_path
        if path is None:
            raise ValueError("clips job requires input.video_path or input.audio_path")
        return path

    @property
    def slug(self) -> str:
        return str(self.config.get("slug") or "")


def already_processed(run: ClipRun) -> bool:
    config = run.config
    return bool(
        config.get("skip_already_processed")
        and not config.get("is_regenerate")
        and not config.get("force")
        and run.slug
        and run.ctx.store.list_clips(run.job_id)
    )


def duration_ok(sc: StageCtx, check_source_duration, media_path: Path) -> bool:
    duration_error = check_source_duration(get_video_metadata(media_path).get("duration") or 0)
    if duration_error:
        sc.logger.warning(duration_error)
        sc.fail(duration_error)
        return False
    return True
