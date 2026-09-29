"""Clip pipeline stages. ClipsRunner wires the host services and drives these in order; each
stage takes the ClipRun (shared run state + stage context) plus the service it needs."""

from kinoforge.segments.clips.pipeline.context import (
    CancellationChecker,
    ClipRun,
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
    "ClipRun",
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
