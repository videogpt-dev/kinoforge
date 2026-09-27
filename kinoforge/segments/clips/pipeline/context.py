from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from kinoforge.contract import Result
from kinoforge.segments.clips.render.formatter import get_video_metadata


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


def already_processed(store, job_id: str, slug: str, config: Dict[str, Any]) -> bool:
    return bool(
        config.get("skip_already_processed")
        and not config.get("is_regenerate")
        and not config.get("force")
        and slug
        and store.list_clips(job_id)
    )


def duration_ok(sc: StageCtx, check_source_duration, media_path: Path) -> bool:
    duration_error = check_source_duration(get_video_metadata(media_path).get("duration") or 0)
    if duration_error:
        sc.logger.warning(duration_error)
        sc.fail(duration_error)
        return False
    return True
