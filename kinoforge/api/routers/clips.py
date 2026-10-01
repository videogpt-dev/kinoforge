"""Clips segment: full pipeline execution, cancellation, and the media/transcription
operations the caller drives against shared-volume files."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from kinoforge.api.paths import require_available, require_clips, shared_path
from kinoforge.contract import JobKind
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
from kinoforge.segments.clips.render.base_render import BaseCut
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg
from kinoforge.segments.clips.render.formatter import ClipFormatter
from kinoforge.segments.clips.render.media import FillStyle, MasterQuality
from kinoforge.segments.clips.transcription import WordAligner
from kinoforge.service.executions import executions
from kinoforge.service.runtimes import ClipsRuntime
from kinoforge.service.runtimes.transcription import GatewayTranscriber
from kinoforge.service.settings import ServiceSettings

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
    with executions.running(request.job_id) as control:
        return ClipsRuntime.from_env().execute(request, is_cancelled=control.is_cancelled)


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
    return Ffmpeg.probe(shared_path(request.path, must_exist=True))


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
    cut = BaseCut(
        source, output, request.in_sec, request.out_sec, (request.src_width, request.src_height),
        crop=request.crop.model_dump() if request.crop else None,
        quality=MasterQuality(request.quality),
    )
    with executions.running(request.execution_id) as control:
        ok = cut.render(is_cancelled=control.is_cancelled if control else None)
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
    ok = ClipFormatter(fill=FillStyle(request.fill)).format(
        source, output, request.aspect_ratio, Ffmpeg.probe(source)["aspect_ratio"]
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
    engine = GatewayTranscriber(ServiceSettings.from_env(), request.owner, request.transcription)
    segments = engine.transcribe_words(source, request.language or None)
    return {
        "segments": segments,
        "aligned": WordAligner.align(request.text, segments, request.duration),
    }
