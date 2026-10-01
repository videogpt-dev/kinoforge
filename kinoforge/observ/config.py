from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from kinoforge.observ.logger import TRACE, KinoLogger, _LEVELS

_ROOT = "kinoforge"
_MARK = {"ERROR": "x", "WARNING": "!", "SUCCESS": "+", "INFO": " ", "DEBUG": ".", "TRACE": "-"}
_configured = False


def resolve_level(name: object, fallback: int) -> int:
    """Level name or int to its number; unknown/None -> fallback."""
    if isinstance(name, int):
        return name
    if not name:
        return fallback
    return _LEVELS.get(str(name).strip().lower(), fallback)


def default_level() -> int:
    """Default per-request console level: KINOFORGE_LOG_LEVEL, else error."""
    return resolve_level(os.getenv("KINOFORGE_LOG_LEVEL"), _LEVELS["error"])


class _TextFormatter(logging.Formatter):
    """Human console line: level mark, [segment:stage], message, then decision fields."""

    def format(self, record: logging.LogRecord) -> str:
        kino = getattr(record, "kino", {}) or {}
        mark = _MARK.get(record.levelname, " ")
        segment, stage = kino.get("segment", ""), kino.get("stage", "")
        loc = f"[{segment}:{stage}] " if stage else (f"[{segment}] " if segment else "")
        fields = kino.get("fields") or {}
        extra = f" {fields}" if fields else ""
        return f"{mark} {loc}{record.getMessage()}{extra}"


class _JsonFormatter(logging.Formatter):
    """One JSON object per record: level, message, correlation, decision fields, timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        kino = getattr(record, "kino", {}) or {}
        out = {
            "ts": record.created,
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }
        for key in ("job_id", "segment", "stage", "idempotency_key"):
            if kino.get(key):
                out[key] = kino[key]
        if kino.get("fields"):
            out["fields"] = kino["fields"]
        return json.dumps(out, ensure_ascii=False)


def configure(force: bool = False) -> logging.Logger:
    """Set up the `kinoforge` logger tree at the TRACE floor (each KinoLogger gates its own
    level). Env: KINOFORGE_LOG_LEVEL, KINOFORGE_LOG_JSON, KINOFORGE_LOG_FILE."""
    global _configured
    root = logging.getLogger(_ROOT)
    if _configured and not force:
        return root

    # floor; let the KinoLogger decides what emits
    root.setLevel(TRACE)
    root.propagate = False
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_JsonFormatter() if os.getenv("KINOFORGE_LOG_JSON") == "1"
                         else _TextFormatter())
    root.addHandler(console)

    log_file = os.getenv("KINOFORGE_LOG_FILE")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Add posthog/splunk/OTLP by attaching their logging.Handler here.
        file_handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=3)
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    _configured = True
    return root


def logger_for(segment: str = "") -> logging.Logger:
    configure()
    return logging.getLogger(f"{_ROOT}.{segment or 'core'}")


def build_logger(
    *,
    job_id: str = "",
    segment: str = "",
    idempotency_key: str = "",
    level: object = None
) -> KinoLogger:
    """Per-request KinoLogger with a fresh capture buffer. `level` (name/int/None) sets console
    verbosity, None -> env default; `.entries` capture is unaffected."""
    return KinoLogger(
        logger_for(segment), job_id=job_id, segment=segment, idempotency_key=idempotency_key,
        entries=[], level=resolve_level(level, default_level()),
    )
