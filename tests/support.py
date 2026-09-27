"""Shared test helpers: a capturing logger and a StageCtx over a fresh Result."""

from __future__ import annotations

from typing import Callable, List, Tuple

from kinoforge.contract import Result
from kinoforge.observ import Logger, build_logger
from kinoforge.segments.clips.pipeline.context import (
    ClipStage,
    StageCtx,
    StageStatus,
)


def capturing_logger() -> Tuple[Logger, Logger]:
    """A per-request logger whose .entries capture the run (console stays quiet in tests)."""
    logger = build_logger(segment="clips")
    return logger, logger


def make_ctx(is_cancelled: Callable[[], bool] = lambda: False) -> Tuple[StageCtx, Result, List[tuple]]:
    """A StageCtx over a fresh Result, with a capturing logger and a recording report_stage.
    Returns (ctx, result, reported) where `reported` collects (stage, status) tuples."""
    result = Result(data={"errors": []})
    logger, _ = capturing_logger()
    reported: List[Tuple[ClipStage, StageStatus]] = []
    ctx = StageCtx(
        result,
        logger,
        report_stage=lambda stage, status: reported.append((stage, status)),
        is_cancelled=is_cancelled,
    )
    return ctx, result, reported


def levels(logger: Logger) -> List[str]:
    return [entry["level"] for entry in logger.entries]


def texts(logger: Logger) -> List[str]:
    return [entry["text"] for entry in logger.entries]
