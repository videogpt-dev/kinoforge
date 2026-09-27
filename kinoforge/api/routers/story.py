"""Story segment: screenwriter write, stateless edit operations, and per-scene media renders."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from kinoforge.service.executions import ExecutionConflict, executions
from kinoforge.service.runtimes import MediaRuntime
from kinoforge.schemas import (
    ImageRenderRequest,
    ImageRenderResponse,
    MusicRenderRequest,
    MusicRenderResponse,
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
from kinoforge.service.runtimes import StoryRuntime

router = APIRouter(tags=["Story"])


@router.post(
    "/v1/segments/story/write",
    summary="Write a story (screenwriter stage)",
    description=(
        "Runs the resolved screenwriter agent over the story context and returns the "
        "StoryPlan (logline, style, characters, per-scene prompt + narration), plus meter "
        "events and logs. Stateless: the caller persists the story on its project record."
    ),
    response_model=StoryWriteResponse,
)
def write_story(request: StoryWriteRequest) -> dict:
    try:
        control = executions.begin(request.job_id)
    except ExecutionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        return StoryRuntime.from_env().write(request, is_cancelled=control.is_cancelled)
    finally:
        executions.finish(request.job_id)


@router.post(
    "/v1/segments/story/operate",
    summary="Edit, translate, or direct an existing story",
    description=(
        "Runs one stateless Story operation from frozen definitions: rewrite_scene, "
        "rewrite_characters, translate, or direct_shots. Caller persists returned changes."
    ),
    response_model=StoryOperationResponse,
)
def operate_story(request: StoryOperationRequest) -> dict:
    return StoryRuntime.from_env().operate(request)


@router.post(
    "/v1/segments/story/media/image",
    summary="Render one scene image or character sheet",
    description=(
        "Builds the image prompt from the frozen presets, calls the gateway, and returns the "
        "image bytes plus any learned prompt ceiling. Stateless: the caller stores the bytes, "
        "composes reference sheets, places assets, and bills."
    ),
    response_model=ImageRenderResponse,
)
def render_image(request: ImageRenderRequest) -> dict:
    return MediaRuntime.from_env().render_image(request)


@router.post(
    "/v1/segments/story/media/video",
    summary="Render one scene clip",
    description=(
        "Builds the clip prompt (shot, cast, cinematography) from the frozen presets, calls the "
        "gateway, and returns the clip bytes plus any learned prompt ceiling. Stateless: the "
        "caller runs the director pass, stores the clip, places assets, and bills."
    ),
    response_model=VideoRenderResponse,
)
def render_clip(request: VideoRenderRequest) -> dict:
    return MediaRuntime.from_env().render_clip(request)


@router.post(
    "/v1/segments/story/media/music",
    summary="Render the story music bed",
    description=(
        "Derives an instrumental mood from the story and sizes one bed to its scenes, calls the "
        "gateway, and returns the audio bytes. Stateless: the caller stores the track and bills."
    ),
    response_model=MusicRenderResponse,
)
def render_music(request: MusicRenderRequest) -> dict:
    return MediaRuntime.from_env().render_music(request)


@router.post(
    "/v1/segments/story/media/voice",
    summary="Render one narration voiceover",
    description=(
        "Synthesizes one narration line through resolved TTS route. Caller stores audio, "
        "aligns captions, updates project progress, and bills."
    ),
    response_model=VoiceRenderResponse,
)
def render_voice(request: VoiceRenderRequest) -> dict:
    return MediaRuntime.from_env().render_voice(request)


@router.post(
    "/v1/segments/story/media/prompts",
    summary="Preview resolved image and video prompts",
    description="Builds exact provider prompts without running or billing media inference.",
    response_model=StoryPromptPreviewResponse,
)
def preview_story_prompts(request: StoryPromptPreviewRequest) -> dict:
    return MediaRuntime.from_env().preview_prompts(request)
