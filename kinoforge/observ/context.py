from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from kinoforge.observ.logger import Logger

_T = TypeVar("_T")


def with_context(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Wrap `fn` in a snapshot of the current context (call on the main thread) so a pool worker
    inherits the bound logger. Each call captures a fresh copy."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


class LoggerContext:
    """Ambient (contextvar-bound) logger access. Lets any function log the "why" without
    threading a logger through every call, and stays correct across threads and async tasks.
    Owns the bound-logger ContextVar and the lazily built env fallback."""

    _current: contextvars.ContextVar[Optional["Logger"]] = contextvars.ContextVar(
        "kinoforge_logger", default=None
    )
    _fallback: Optional["Logger"] = None

    @staticmethod
    def bind(logger: "Logger") -> contextvars.Token:
        """Make `logger` the ambient logger for this request. Returns a token to reset() with."""
        return LoggerContext._current.set(logger)

    @staticmethod
    def reset(token: contextvars.Token) -> None:
        LoggerContext._current.reset(token)

    @staticmethod
    def current() -> Optional["Logger"]:
        """The ambient logger, or None when nothing is bound (e.g. a direct unit-test call)."""
        return LoggerContext._current.get()

    @staticmethod
    def active() -> "Logger":
        """The bound logger if there is one, else a lazily built fallback (env sinks, no run
        correlation). Lets library code log unconditionally without threading a logger or
        None-checks; inside a request the runner has bound the correlated logger, so events
        land in its lifeline."""
        log = LoggerContext._current.get()
        if log is not None:
            return log
        if LoggerContext._fallback is None:
            from kinoforge.observ.config import build_logger

            LoggerContext._fallback = build_logger()
        return LoggerContext._fallback
