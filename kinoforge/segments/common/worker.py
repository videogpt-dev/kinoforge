from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional

from kinoforge.observ import KinoLogger


class StageStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(kw_only=True)
class Worker:
    """State one segment execution carries through its pipeline: logger, cancellation,
    the stage log, status, and the result being built in `data`."""

    logger: KinoLogger
    is_cancelled: Optional[Callable[[], bool]] = None
    status: str = "ok"
    error: str = ""
    stages: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = {"errors": [], **self.data}

    def stage(self, stage: StrEnum, status: StageStatus) -> None:
        self.logger.debug(f"stage {stage.value}:{status.value}", stage=stage.value,
                          status=status.value)
        self.stages.append(f"{stage.value}:{status.value}")

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.error = message
        self.data["errors"].append(message)

    def stopped(self, stage: Optional[StrEnum] = None) -> bool:
        """True (and marks the worker cancelled) once the caller asked to cancel."""
        if self.is_cancelled is None or not self.is_cancelled():
            return False
        if stage is not None:
            self.stage(stage, StageStatus.CANCELLED)
        self.status = "cancelled"
        self.error = ""
        self.data["cancelled"] = True
        self.logger.warning("Execution cancelled")
        return True
