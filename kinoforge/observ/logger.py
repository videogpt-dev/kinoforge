from __future__ import annotations

import logging
from typing import Any, List, Optional

# Custom level between INFO and WARNING: a stage produced a good result.
SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

# Custom level below DEBUG for fine tracing; console/file only, never captured into `logs`.
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LEVELS = {
    "trace": TRACE,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "success": SUCCESS,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class KinoLogger:
    """Thin facade over a standard `logging.Logger`. The stdlib logger owns emission,
    formatting, handlers and levels; this adds only what stdlib lacks for this service:

    - a `.success` level, and per-call decision `fields` (shipped as the `kino` LogRecord attr),
    - correlation (`job_id`/`segment`/`stage`/`idempotency_key`) attached to every record,
    - an in-memory capture of {level, text} entries returned to caller as the job's `logs`.
    """

    def __init__(
        self,
        base: logging.Logger,
        *,
        job_id: str = "",
        segment: str = "",
        stage: str = "",
        idempotency_key: str = "",
        entries: Optional[List[dict]] = None,
        level: int = 0,
    ) -> None:
        self._log = base
        self.job_id = job_id
        self.segment = segment
        self.stage = stage
        self.idempotency_key = idempotency_key
        self._entries = entries if entries is not None else []
        self._level = level  # console/file threshold; capture (.entries) is independent

    @property
    def entries(self) -> List[dict]:
        return self._entries

    def at_stage(self, stage: str) -> "KinoLogger":
        """A child bound to a stage, sharing this logger's stdlib logger, capture, and level."""
        return KinoLogger(
            self._log, job_id=self.job_id, segment=self.segment, stage=stage,
            idempotency_key=self.idempotency_key, entries=self._entries, level=self._level,
        )

    def event(self, level: str, message: str, *, capture: bool = True, **fields: Any) -> None:
        levelno = _LEVELS.get(level, logging.INFO)
        if capture:
            self._entries.append({"level": level, "text": message})
        if levelno >= self._level and self._log.isEnabledFor(levelno):
            self._log.log(levelno, message, extra={"kino": {
                "job_id": self.job_id, "segment": self.segment, "stage": self.stage,
                "idempotency_key": self.idempotency_key, "fields": fields,
            }})

    def trace(self, message: str, **fields: Any) -> None:
        self.event("trace", message, capture=False, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self.event("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self.event("info", message, **fields)

    def success(self, message: str, **fields: Any) -> None:
        self.event("success", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self.event("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.event("error", message, **fields)
