"""Structured logging for kinoforge, built on Python's standard `logging`.

Build a per-request logger with build_logger(job_id=, segment=), bind() it so any method can
log the "why" via observ.active() without threading a logger, and it emits through the
`kinoforge` logging tree (console + optional rotating JSON file, per env; posthog/splunk drop
in as a standard logging.Handler in config.configure). The logger's .entries are returned to
Caller for the job log. The @logged decorator auto-logs a method's input and output.

Env: KINOFORGE_LOG_LEVEL (default INFO), KINOFORGE_LOG_JSON (1 = JSON console),
KINOFORGE_LOG_FILE (path = also append rotating JSON there)."""

from kinoforge.observ.autolog import logged
from kinoforge.observ.config import build_logger, configure, logger_for
from kinoforge.observ.logger import SUCCESS, TRACE, KinoLogger
from kinoforge.observ.logger_context import LoggerContext, with_context
from kinoforge.observ.shape import shape

# Terse aliases so pervasive callers stay `active()` / `bind()` rather than the qualified form.
bind = LoggerContext.bind
reset = LoggerContext.reset
current = LoggerContext.current
active = LoggerContext.active

__all__ = [
    "build_logger",
    "configure",
    "logger_for",
    "LoggerContext",
    "bind",
    "reset",
    "current",
    "active",
    "with_context",
    "shape",
    "logged",
    "KinoLogger",
    "SUCCESS",
    "TRACE",
]
