"""Shared test helpers: a capturing logger and a ClipRun for stage tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

from kinoforge.observ import KinoLogger, build_logger
from kinoforge.segments.clips.run import ClipRun


def capturing_logger() -> Tuple[KinoLogger, KinoLogger]:
    """A per-request logger whose .entries capture the run (console stays quiet in tests)."""
    logger = build_logger(segment="clips")
    return logger, logger


def levels(logger: KinoLogger) -> List[str]:
    return [entry["level"] for entry in logger.entries]


def texts(logger: KinoLogger) -> List[str]:
    return [entry["text"] for entry in logger.entries]


def make_run(
    config=None, *, video=Path("v.mp4"), audio=None, workdir=Path("work"),
    is_cancelled: Callable[[], bool] = lambda: False, **state,
) -> ClipRun:
    return ClipRun(
        job_id="j1", config=dict(config or {}), workdir=workdir, logger=capturing_logger()[0],
        video_path=video, audio_path=audio, is_cancelled=is_cancelled, **state,
    )
