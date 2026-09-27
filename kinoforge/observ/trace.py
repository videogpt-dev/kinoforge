from __future__ import annotations

import functools
import time
from typing import Any, Callable

from kinoforge.observ.context import LoggerContext

_MAX_REPR = 200


def _short(value: Any) -> str:
    """A compact, safe repr so auto-logging never dumps a whole payload into a line."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unreprable {type(value).__name__}>"
    return text if len(text) <= _MAX_REPR else text[:_MAX_REPR] + "…"


def logged(fn: Callable | None = None, *, level: str = "debug"):
    """Auto-log a method's input and output to the ambient logger, no logger.log() calls needed.

    On call it logs `enter <name>` with the args, on return `exit <name>` with the result and
    elapsed ms, and re-raises with an `error` on exception. Fires only when a logger is bound
    (observ.bind); reprs are computed only then, so an unbound call is nearly free. Default
    level is debug, so it stays quiet unless KINOFORGE_LOG_LEVEL=DEBUG. Use `@logged` or
    `@logged(level="info")`.
    """

    def wrap(func: Callable) -> Callable:
        name = func.__qualname__

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            log = LoggerContext.current()
            if log is None:
                return func(*args, **kwargs)
            emit = getattr(log, level)
            # Drop a bound `self`/`cls` from the logged args; it is noise, not input.
            shown = args[1:] if args and hasattr(args[0], func.__name__) else args
            emit(f"enter {name}", args=[_short(a) for a in shown],
                 kwargs={k: _short(v) for k, v in kwargs.items()})
            start = time.perf_counter()
            try:
                out = func(*args, **kwargs)
            except Exception as exc:
                log.error(f"raised {name}: {exc!s}", ms=round((time.perf_counter() - start) * 1000))
                raise
            emit(f"exit {name}", ms=round((time.perf_counter() - start) * 1000), result=_short(out))
            return out

        return inner

    return wrap(fn) if callable(fn) else wrap
