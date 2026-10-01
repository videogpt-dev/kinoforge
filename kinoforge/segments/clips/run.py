from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kinoforge.contract import Meter
from kinoforge.observ import KinoLogger

Segments = List[Dict[str, Any]]


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


@dataclass
class ClipRun:
    """Everything one clips execution reads and writes: inputs, merged config, the state a
    previous run left (record, transcript, has_clips), and the result being built."""

    job_id: str
    config: Dict[str, Any]
    workdir: Path
    logger: KinoLogger
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    meter: Optional[Meter] = None
    is_cancelled: Optional[Callable[[], bool]] = None
    record: Optional[Dict[str, Any]] = None
    transcript: Segments = field(default_factory=list)
    has_clips: bool = False
    clip_ids: List[Optional[str]] = field(default_factory=list)
    status: str = "ok"
    error: str = ""
    stages: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = {
            "video_path": str(self.video_path or ""),
            "audio_path": str(self.audio_path or ""),
            "clips": [], "moments": [], "transcript": None, "errors": [],
            **self.data,
        }

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

    def stage(self, stage: ClipStage, status: StageStatus) -> None:
        self.logger.debug(f"stage {stage.value}:{status.value}", stage=stage.value,
                          status=status.value)
        self.stages.append(f"{stage.value}:{status.value}")

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.error = message
        self.data["errors"].append(message)

    def stopped(self, stage: Optional[ClipStage] = None) -> bool:
        """True (and marks the run cancelled) once the caller asked to cancel."""
        if self.is_cancelled is None or not self.is_cancelled():
            return False
        if stage is not None:
            self.stage(stage, StageStatus.CANCELLED)
        self.status = "cancelled"
        self.error = ""
        self.data["cancelled"] = True
        self.logger.warning("Execution cancelled")
        return True
