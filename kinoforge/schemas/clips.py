"""Clips segment schemas: options (with validation), the execution request, and its response."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kinoforge.observ import active
from kinoforge.schemas.common import (
    DefinitionBundleRequest,
    ExecutionResultResponse,
    ExecutionState,
    ExecutionStateResponse,
    LogEntryResponse,
    MeterEventResponse,
)


class ClipsOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    clip_count: int = Field(
        default=10, ge=1, le=100,
        description="Maximum number of ranked moments returned or rendered.",
    )
    min_length: float = Field(default=20, gt=0, description="Preferred minimum clip duration (s).")
    max_length: float = Field(default=60, gt=0, description="Maximum clip duration in seconds.")
    formats: List[str] = Field(
        default_factory=lambda: ["9:16"],
        description="Output aspect ratios such as 9:16, 16:9, or 1:1.",
    )
    quality: Literal["high", "medium", "low"] = "high"
    generate_captions: bool = True
    analyze_only: bool = Field(
        default=False, description="Find and rank moments without rendering clip files."
    )
    language: Optional[str] = Field(
        default=None, description="Optional BCP 47 transcription language; null enables detection."
    )
    whisper_model: Optional[str] = None
    min_interest_score: float = Field(default=0.3, ge=0, le=1)
    ai_provider: Optional[str] = None
    moment_route: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved inference provider and model route supplied by runtime.",
    )
    anti_hallucination: bool = False
    try_youtube_subs: bool = True
    force: bool = False

    @model_validator(mode="after")
    def validate_clip_window(self) -> "ClipsOptionsRequest":
        if self.max_length <= self.min_length:
            active().warning(
                "rejected clips options: max_length must exceed min_length",
                min_length=self.min_length, max_length=self.max_length,
            )
            raise ValueError("max_length must be greater than min_length")
        if not self.formats:
            active().warning("rejected clips options: no output format")
            raise ValueError("at least one output format is required")
        return self


class ClipsExecutionRequest(BaseModel):
    job_id: str = Field(description="Caller-owned execution identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = Field(description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    workspace: str = Field(description="Execution directory below KINOFORGE_SHARED_ROOT.")
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Clip job input. video_path must point inside shared root.",
    )
    options: ClipsOptionsRequest = Field(
        default_factory=ClipsOptionsRequest, description="Per-job clip generation choices"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved engine, transcription, rendering, scoring, and route configuration.",
    )
    definitions: DefinitionBundleRequest = Field(description="Frozen definitions resolved by Caller.")
    state: ExecutionState = Field(
        default_factory=ExecutionState, description="Optional durable state restored by Caller."
    )


class ClipsExecutionResponse(BaseModel):
    result: ExecutionResultResponse
    state: ExecutionStateResponse
    meter_events: List[MeterEventResponse] = Field(default_factory=list)
    logs: List[LogEntryResponse] = Field(default_factory=list)
