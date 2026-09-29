"""Story segment schemas: write context/request/response and the edit-operation request."""

from enum import StrEnum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from kinoforge.schemas.common import (
    DefinitionBundleRequest,
    LoggableRequest,
    LogEntryResponse,
    MeterEventResponse,
)


class StoryContextRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = Field(description="Story title the screenwriter writes from.")
    description: str = ""
    scene_count: int = Field(default=8, ge=1, le=64)
    aspect_ratio: str = "9:16"
    language: str = ""
    genre: str = ""
    cast: List[Dict[str, Any]] = Field(default_factory=list)
    premise: str = ""
    series_name: str = ""
    series_episodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Frozen ordered episode slate/history for continuity-aware screenwriters.",
    )
    series_position: int = Field(default=0, ge=0)
    mature: bool = False


class StoryWriteRequest(LoggableRequest):
    job_id: str = Field(description="Caller-owned execution identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = Field(default="", description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    context: StoryContextRequest = Field(description="What the screenwriter writes from.")
    agent: Dict[str, Any] = Field(
        description="Resolved screenwriter: ordered stages, persona, per-stage model overrides.",
    )
    pick: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved default text route {provider, model} for stages without overrides.",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved output-token budgets and other runtime config.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen story prompts and fragments resolved by the caller.",
    )


class StoryWriteResponse(BaseModel):
    result: Dict[str, Any] = Field(
        description="Run result: {ok, story, warnings, usage} or {ok: False, error}.",
    )
    meter_events: List[MeterEventResponse] = Field(default_factory=list)
    logs: List[LogEntryResponse] = Field(default_factory=list)


class StoryOperation(StrEnum):
    REWRITE_SCENE = "rewrite_scene"
    REWRITE_CHARACTERS = "rewrite_characters"
    TRANSLATE = "translate"
    DIRECT_SHOTS = "direct_shots"
    SUGGEST_FIELD = "suggest_field"


class StoryOperationRequest(LoggableRequest):
    job_id: str = Field(description="Caller-owned operation identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = ""
    owner: str = ""
    operation: StoryOperation
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation input: project/story, scene index, language, or instructions.",
    )
    agent: Dict[str, Any] = Field(default_factory=dict)
    pick: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Resolved rewrite and translation token budgets."
    )
    definitions: DefinitionBundleRequest


class StoryOperationResponse(BaseModel):
    result: Dict[str, Any]
    meter_events: List[MeterEventResponse] = Field(default_factory=list)
    logs: List[LogEntryResponse] = Field(default_factory=list)
