"""Clip pipeline stages. ClipsRunner wires the host services and drives these in order; each
stage function takes a StageCtx (shared result/logger/stage-reporting/cancellation) plus its deps."""

from kinoforge.segments.clips.pipeline.context import (
    CancellationChecker,
    ClipStage,
    PipelineLogger,
    SaveProject,
    StageCtx,
    StageReporter,
    StageStatus,
    TranscribeVideo,
    already_processed,
    duration_ok,
)
from kinoforge.segments.clips.pipeline.render import render_clips
from kinoforge.segments.clips.pipeline.select import (
    apply_limits,
    find_moments,
    persist_moments,
    rank_moments,
)
from kinoforge.segments.clips.pipeline.transcribe import transcribe

__all__ = [
    "CancellationChecker",
    "ClipStage",
    "PipelineLogger",
    "SaveProject",
    "StageCtx",
    "StageReporter",
    "StageStatus",
    "TranscribeVideo",
    "already_processed",
    "duration_ok",
    "transcribe",
    "find_moments",
    "rank_moments",
    "apply_limits",
    "persist_moments",
    "render_clips",
]
