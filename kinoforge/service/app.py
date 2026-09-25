import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from kinoforge.clips.render.base_render import render_base_clip
from kinoforge.clips.render.formatter import (
    apply_format_with_aspect_ratio,
    get_video_metadata,
)
from kinoforge.clips.transcription import Transcriber, TranscriptionError, align_words
from kinoforge.contract import JobKind
from kinoforge.service.auth import require_service
from kinoforge.service.executions import ExecutionConflict, executions
from kinoforge.service.infrelay import InfrelayClient, InfrelayError
from kinoforge.service.media_runtime import media_runtime
from kinoforge.service.models import (
    AlignmentResponse,
    AlignRequest,
    BaseRenderRequest,
    CancellationResponse,
    ClipsExecutionRequest,
    ClipsExecutionResponse,
    FormatRequest,
    HealthResponse,
    ImageRenderRequest,
    ImageRenderResponse,
    MediaMetadataResponse,
    MediaOperationResponse,
    MusicRenderRequest,
    MusicRenderResponse,
    PathRequest,
    SegmentsResponse,
    SeriesPlanRequest,
    SeriesPlanResponse,
    StoryOperationRequest,
    StoryOperationResponse,
    StoryPromptPreviewRequest,
    StoryPromptPreviewResponse,
    StoryWriteRequest,
    StoryWriteResponse,
    VideoRenderRequest,
    VideoRenderResponse,
    VoiceRenderRequest,
    VoiceRenderResponse,
)
from kinoforge.service.runtime import runtime
from kinoforge.service.segments import SegmentCatalog
from kinoforge.service.series_runtime import series_runtime
from kinoforge.service.story_runtime import story_runtime

app = FastAPI(
    title="Kinoforge API",
    summary="VideoGPT generation engine",
    description=(
        "Source-available clips, story, and series generation engine. Cloud core owns durable "
        "jobs and calls this service through REST. API exposes clips execution, Story "
        "writing and edits, plus supporting media operations. Protected endpoints require "
        "ServiceToken when "
        "KINOFORGE_SERVICE_TOKEN is configured."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "Service", "description": "Health and service metadata."},
        {
            "name": "Discovery",
            "description": "Engine segments and their implementation status.",
        },
        {
            "name": "Clips",
            "description": "Execute complete long-video to short-clips generation jobs.",
        },
        {
            "name": "Story",
            "description": "Write story plans and generate scene media from frozen definitions.",
        },
        {
            "name": "Series",
            "description": "Plan connected episodes for a recurring show from a premise and cast.",
        },
        {
            "name": "Media",
            "description": "Probe, render, and aspect-format shared-volume media.",
        },
        {
            "name": "Transcription",
            "description": "Transcribe narration timing and align canonical text for captions.",
        },
    ],
)


def _shared_root() -> Path:
    return Path(os.getenv("KINOFORGE_SHARED_ROOT") or "/app/output").resolve()


def _shared_path(raw: str, *, must_exist: bool = False) -> Path:
    path = Path(raw).resolve()
    root = _shared_root()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="path is outside shared root")
    if must_exist and not path.is_file():
        raise HTTPException(status_code=404, detail=f"media not found: {path.name}")
    return path


def _require_available(code_name: JobKind) -> None:
    if code_name not in (JobKind.CLIPS, JobKind.STORY):
        raise HTTPException(
            status_code=501,
            detail=f"segment is not implemented: {code_name.value}",
        )


def _require_clips(code_name: JobKind) -> None:
    if code_name is not JobKind.CLIPS:
        raise HTTPException(status_code=404, detail="operation is available only for clips")


@app.get(
    "/health",
    tags=["Service"],
    summary="Service health",
    response_model=HealthResponse,
)
def health() -> dict:
    return {"ok": True, "service": "kinoforge", "version": "0.1.0"}


@app.get(
    "/v1/segments",
    tags=["Discovery"],
    summary="List generation segments",
    description=(
        "Returns stable metadata for every Kinoforge segment, including planned segments. "
        "Consumers must inspect status before using execute_path."
    ),
    dependencies=[Depends(require_service)],
    response_model=SegmentsResponse,
)
def list_segments() -> SegmentsResponse:
    return SegmentCatalog.list()


@app.post(
    "/v1/segments/{code_name}/execute",
    tags=["Clips"],
    summary="Execute clips pipeline",
    description=(
        "Runs transcription, moment discovery, scoring, base rendering, and output formatting. "
        "Paths must live below KINOFORGE_SHARED_ROOT. Returns new durable state, artifacts, "
        "meter events, and execution logs for cloud core to persist."
    ),
    dependencies=[Depends(require_service)],
    response_model=ClipsExecutionResponse,
)
def execute_segment(code_name: JobKind, request: ClipsExecutionRequest) -> dict:
    _require_clips(code_name)
    audio_raw = str(request.input.get("audio_path") or "")
    video_raw = str(request.input.get("video_path") or "")
    if not audio_raw and not video_raw:
        raise HTTPException(status_code=400, detail="input needs audio_path or video_path")
    if video_raw:
        request.input["video_path"] = str(_shared_path(video_raw, must_exist=True))
    if audio_raw:
        request.input["audio_path"] = str(_shared_path(audio_raw, must_exist=True))
    workspace = _shared_path(request.workspace)
    request.workspace = str(workspace)
    try:
        control = executions.begin(request.job_id)
    except ExecutionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        return runtime().execute(request, is_cancelled=control.is_cancelled)
    finally:
        executions.finish(request.job_id)


@app.post(
    "/v1/segments/{code_name}/executions/{execution_id}/cancel",
    tags=["Clips"],
    summary="Cancel active segment execution",
    description=(
        "Requests cooperative cancellation. Cloud core remains owner of durable job state and "
        "marks its job cancelled independently."
    ),
    dependencies=[Depends(require_service)],
    response_model=CancellationResponse,
)
def cancel_execution(code_name: JobKind, execution_id: str) -> dict:
    _require_available(code_name)
    return {
        "accepted": executions.cancel(execution_id),
        "execution_id": execution_id,
    }


@app.post(
    "/v1/segments/story/write",
    tags=["Story"],
    summary="Write a story (screenwriter stage)",
    description=(
        "Runs the resolved screenwriter agent over the story context and returns the "
        "StoryPlan (logline, style, characters, per-scene prompt + narration), plus meter "
        "events and logs. Stateless: the caller persists the story on its project record."
    ),
    dependencies=[Depends(require_service)],
    response_model=StoryWriteResponse,
)
def write_story(request: StoryWriteRequest) -> dict:
    try:
        control = executions.begin(request.job_id)
    except ExecutionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        return story_runtime().write(request, is_cancelled=control.is_cancelled)
    finally:
        executions.finish(request.job_id)


@app.post(
    "/v1/segments/story/operate",
    tags=["Story"],
    summary="Edit, translate, or direct an existing story",
    description=(
        "Runs one stateless Story operation from frozen definitions: rewrite_scene, "
        "rewrite_characters, translate, or direct_shots. Caller persists returned changes."
    ),
    dependencies=[Depends(require_service)],
    response_model=StoryOperationResponse,
)
def operate_story(request: StoryOperationRequest) -> dict:
    return story_runtime().operate(request)


@app.post(
    "/v1/segments/series/plan",
    tags=["Series"],
    summary="Plan connected episodes for a series",
    description=(
        "Runs the resolved showrunner over a series premise and cast and returns proposed "
        "episode ideas (title + description), plus meter events and logs. Stateless: the caller "
        "persists the series and creates the episodes the user approves."
    ),
    dependencies=[Depends(require_service)],
    response_model=SeriesPlanResponse,
)
def plan_series(request: SeriesPlanRequest) -> dict:
    return series_runtime().plan(request)


@app.post(
    "/v1/segments/series/media/portrait",
    tags=["Series"],
    summary="Render one recurring cast portrait",
    description=(
        "Builds a recurring-character portrait from frozen presets and returns image bytes. "
        "Stateless: caller stores and selects portrait."
    ),
    dependencies=[Depends(require_service)],
    response_model=ImageRenderResponse,
)
def render_series_portrait(request: ImageRenderRequest) -> dict:
    if request.mode != "character":
        raise HTTPException(422, "Series portrait mode must be character")
    return media_runtime().render_image(request)


@app.post(
    "/v1/segments/story/media/image",
    tags=["Story"],
    summary="Render one scene image or character sheet",
    description=(
        "Builds the image prompt from the frozen presets, calls the gateway, and returns the "
        "image bytes plus any learned prompt ceiling. Stateless: the caller stores the bytes, "
        "composes reference sheets, places assets, and bills."
    ),
    dependencies=[Depends(require_service)],
    response_model=ImageRenderResponse,
)
def render_image(request: ImageRenderRequest) -> dict:
    return media_runtime().render_image(request)


@app.post(
    "/v1/segments/story/media/video",
    tags=["Story"],
    summary="Render one scene clip",
    description=(
        "Builds the clip prompt (shot, cast, cinematography) from the frozen presets, calls the "
        "gateway, and returns the clip bytes plus any learned prompt ceiling. Stateless: the "
        "caller runs the director pass, stores the clip, places assets, and bills."
    ),
    dependencies=[Depends(require_service)],
    response_model=VideoRenderResponse,
)
def render_clip(request: VideoRenderRequest) -> dict:
    return media_runtime().render_clip(request)


@app.post(
    "/v1/segments/story/media/music",
    tags=["Story"],
    summary="Render the story music bed",
    description=(
        "Derives an instrumental mood from the story and sizes one bed to its scenes, calls the "
        "gateway, and returns the audio bytes. Stateless: the caller stores the track and bills."
    ),
    dependencies=[Depends(require_service)],
    response_model=MusicRenderResponse,
)
def render_music(request: MusicRenderRequest) -> dict:
    return media_runtime().render_music(request)


@app.post(
    "/v1/segments/story/media/voice",
    tags=["Story"],
    summary="Render one narration voiceover",
    description=(
        "Synthesizes one narration line through resolved TTS route. Caller stores audio, "
        "aligns captions, updates project progress, and bills."
    ),
    dependencies=[Depends(require_service)],
    response_model=VoiceRenderResponse,
)
def render_voice(request: VoiceRenderRequest) -> dict:
    return media_runtime().render_voice(request)


@app.post(
    "/v1/segments/story/media/prompts",
    tags=["Story"],
    summary="Preview resolved image and video prompts",
    description="Builds exact provider prompts without running or billing media inference.",
    dependencies=[Depends(require_service)],
    response_model=StoryPromptPreviewResponse,
)
def preview_story_prompts(request: StoryPromptPreviewRequest) -> dict:
    return media_runtime().preview_prompts(request)


@app.post(
    "/v1/segments/{code_name}/media/probe",
    tags=["Media"],
    summary="Probe media metadata",
    dependencies=[Depends(require_service)],
    response_model=MediaMetadataResponse,
)
def probe_media(code_name: JobKind, request: PathRequest) -> dict:
    _require_clips(code_name)
    return get_video_metadata(_shared_path(request.path, must_exist=True))


@app.post(
    "/v1/segments/{code_name}/media/render-base",
    tags=["Media"],
    summary="Render base clip",
    dependencies=[Depends(require_service)],
    response_model=MediaOperationResponse,
)
def render_base(code_name: JobKind, request: BaseRenderRequest) -> dict:
    _require_clips(code_name)
    source = _shared_path(request.source, must_exist=True)
    output = _shared_path(request.output)
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


@app.post(
    "/v1/segments/{code_name}/media/format",
    tags=["Media"],
    summary="Format media aspect ratio",
    dependencies=[Depends(require_service)],
    response_model=MediaOperationResponse,
)
def format_media(code_name: JobKind, request: FormatRequest) -> dict:
    _require_clips(code_name)
    source = _shared_path(request.source, must_exist=True)
    output = _shared_path(request.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = get_video_metadata(source)
    ok = apply_format_with_aspect_ratio(
        source,
        output,
        request.aspect_ratio,
        info,
        moment={},
        fill=request.fill,
    )
    return {"ok": ok, "path": str(output) if ok else ""}


@app.post(
    "/v1/segments/{code_name}/transcription/align",
    tags=["Transcription"],
    summary="Align narration transcript",
    description=(
        "Transcribes narration through Infrelay, then maps canonical input text onto recognized "
        "word timing for accurate captions."
    ),
    dependencies=[Depends(require_service)],
    response_model=AlignmentResponse,
)
def align_transcription(code_name: JobKind, request: AlignRequest) -> dict:
    _require_clips(code_name)
    source = _shared_path(request.path, must_exist=True)
    infrelay = InfrelayClient(
        os.getenv("INFRELAY_URL") or "",
        os.getenv("INFRELAY_SERVICE_TOKEN") or "",
        request.owner,
    )

    def transcribe(*args, **kwargs):
        try:
            return infrelay.transcribe(*args, **kwargs)
        except InfrelayError as exc:
            raise TranscriptionError(str(exc)) from exc

    engine = Transcriber(
        transcription=dict(request.transcription),
        cache_dir=Path(os.getenv("KINOFORGE_CACHE_DIR") or _shared_root() / ".kinoforge-cache"),
        transcribe_bytes=transcribe,
    )
    segments = engine.transcribe_words(source, request.language or None)
    return {
        "segments": segments,
        "aligned": align_words(request.text, segments, request.duration),
    }
