"""Clips segment: full pipeline execution, cancellation, and the media/transcription
operations the caller drives against shared-volume files."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from kinoforge.contract import JobKind
from kinoforge.segments.clips.render.base_render import render_base_clip
from kinoforge.segments.clips.render.formatter import (
    apply_format_with_aspect_ratio,
    get_video_metadata,
)
from kinoforge.segments.clips.transcription import Transcriber, TranscriptionError, align_words
from kinoforge.service.executions import ExecutionConflict, executions
from kinoforge.inference import InfrelayClient, InfrelayError
from kinoforge.schemas import (
    AlignmentResponse,
    AlignRequest,
    BaseRenderRequest,
    CancellationResponse,
    ClipsExecutionRequest,
    ClipsExecutionResponse,
    FormatRequest,
    MediaMetadataResponse,
    MediaOperationResponse,
    PathRequest,
)
from kinoforge.api.paths import require_available, require_clips, shared_path, shared_root
from kinoforge.service.runtimes import ClipsRuntime

router = APIRouter()


@router.post(
    "/v1/segments/{code_name}/execute",
    tags=["Clips"],
    summary="Execute clips pipeline",
    description=(
        "Runs transcription, moment discovery, scoring, base rendering, and output formatting. "
    ),
    response_model=ClipsExecutionResponse,
)
def execute_segment(code_name: JobKind, request: ClipsExecutionRequest) -> dict:
    require_clips(code_name)
    audio_raw = str(request.input.get("audio_path") or "")
    video_raw = str(request.input.get("video_path") or "")
    if not audio_raw and not video_raw:
        raise HTTPException(status_code=400, detail="input needs audio_path or video_path")
    if video_raw:
        request.input["video_path"] = str(shared_path(video_raw, must_exist=True))
    if audio_raw:
        request.input["audio_path"] = str(shared_path(audio_raw, must_exist=True))
    request.workspace = str(shared_path(request.workspace))
    try:
        control = executions.begin(request.job_id)
    except ExecutionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        return ClipsRuntime.from_env().execute(request, is_cancelled=control.is_cancelled)
    finally:
        executions.finish(request.job_id)


@router.post(
    "/v1/segments/{code_name}/executions/{execution_id}/cancel",
    tags=["Clips"],
    summary="Cancel active segment execution",
    description=(
        "Requests cooperative cancellation. Caller remains owner of durable job state and "
        "marks its job cancelled independently."
    ),
    response_model=CancellationResponse,
)
def cancel_execution(code_name: JobKind, execution_id: str) -> dict:
    require_available(code_name)
    return {"accepted": executions.cancel(execution_id), "execution_id": execution_id}


@router.post(
    "/v1/segments/{code_name}/media/probe",
    tags=["Media"],
    summary="Probe media metadata",
    response_model=MediaMetadataResponse,
)
def probe_media(code_name: JobKind, request: PathRequest) -> dict:
    require_clips(code_name)
    return get_video_metadata(shared_path(request.path, must_exist=True))


@router.post(
    "/v1/segments/{code_name}/media/render-base",
    tags=["Media"],
    summary="Render base clip",
    response_model=MediaOperationResponse,
)
def render_base(code_name: JobKind, request: BaseRenderRequest) -> dict:
    require_clips(code_name)
    source = shared_path(request.source, must_exist=True)
    output = shared_path(request.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    control = None
    if request.execution_id:
        try:
            control = executions.begin(request.execution_id)
        except ExecutionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        ok = render_base_clip(
            source,
            output,
            request.in_sec,
            request.out_sec,
            request.crop.model_dump() if request.crop else None,
            request.src_width,
            request.src_height,
            request.quality,
            is_cancelled=control.is_cancelled if control else None,
        )
    finally:
        if request.execution_id:
            executions.finish(request.execution_id)
    return {"ok": ok, "path": str(output) if ok else ""}


@router.post(
    "/v1/segments/{code_name}/media/format",
    tags=["Media"],
    summary="Format media aspect ratio",
    response_model=MediaOperationResponse,
)
def format_media(code_name: JobKind, request: FormatRequest) -> dict:
    require_clips(code_name)
    source = shared_path(request.source, must_exist=True)
    output = shared_path(request.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = get_video_metadata(source)
    ok = apply_format_with_aspect_ratio(
        source, output, request.aspect_ratio, info, moment={}, fill=request.fill
    )
    return {"ok": ok, "path": str(output) if ok else ""}


@router.post(
    "/v1/segments/{code_name}/transcription/align",
    tags=["Transcription"],
    summary="Align narration transcript",
    description=(
        "Transcribes narration through Infrelay, then maps canonical input text onto recognized "
        "word timing for accurate captions."
    ),
    response_model=AlignmentResponse,
)
def align_transcription(code_name: JobKind, request: AlignRequest) -> dict:
    require_clips(code_name)
    source = shared_path(request.path, must_exist=True)
    infrelay = InfrelayClient(
        os.getenv("INFRELAY_URL") or "", os.getenv("INFRELAY_SERVICE_TOKEN") or "", request.owner
    )

    def transcribe(*args, **kwargs):
        try:
            return infrelay.transcribe(*args, **kwargs)
        except InfrelayError as exc:
            raise TranscriptionError(str(exc)) from exc

    engine = Transcriber(
        transcription=dict(request.transcription),
        cache_dir=Path(os.getenv("KINOFORGE_CACHE_DIR") or shared_root() / ".kinoforge-cache"),
        transcribe_bytes=transcribe,
    )
    segments = engine.transcribe_words(source, request.language or None)
    return {
        "segments": segments,
        "aligned": align_words(request.text, segments, request.duration),
    }
