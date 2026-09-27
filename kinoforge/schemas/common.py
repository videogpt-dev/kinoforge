"""Shared request/response schemas: frozen-definition bundle, execution state, path input,
and the generic response primitives (meter events, log entries, artifacts, results)."""

from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EngineCompatibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: str = Field(description="Oldest compatible Kinoforge version.")
    maximum_exclusive: Optional[str] = Field(
        default=None,
        description="First incompatible Kinoforge version, when bounded.",
    )


class DefinitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["screenwriter", "agent", "preset", "fragment"]
    subtype: str = ""
    key: str
    name: str = ""
    category: str = ""
    body: str = ""
    extras: Dict[str, Any] = Field(default_factory=dict)
    editable: Optional[bool] = None
    visible: Optional[bool] = None


class DefinitionBundleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    id: str
    version: str
    engine: EngineCompatibilityRequest
    definitions: List[DefinitionRequest] = Field(default_factory=list)


class ExecutionState(BaseModel):
    record: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Existing project record supplied by Caller when resuming execution.",
    )
    transcript: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Existing transcript segments supplied when transcription already completed.",
    )
    has_clips: bool = Field(
        default=False,
        description="Whether Caller storage already contains rendered clips for this job.",
    )


class PathRequest(BaseModel):
    path: str = Field(description="Media path below KINOFORGE_SHARED_ROOT.")


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


class MeterEventResponse(BaseModel):
    action: str
    qty: float
    variant: str = ""


class LogEntryResponse(BaseModel):
    # debug rides along too: stage-lifeline reports and @logged enter/exit are captured as debug.
    level: Literal["debug", "info", "success", "warning", "error"]
    text: str


class ArtifactResponse(BaseModel):
    path: str
    media: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResultResponse(BaseModel):
    status: str
    artifacts: List[ArtifactResponse] = Field(default_factory=list)
    stages: List[str] = Field(default_factory=list)
    error: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class ExecutionStateResponse(BaseModel):
    record: Optional[Dict[str, Any]] = None
    transcript: Optional[List[Dict[str, Any]]] = None


class MediaMetadataResponse(BaseModel):
    width: int
    height: int
    duration: float
    aspect_ratio: float
    fps: float


class MediaOperationResponse(BaseModel):
    ok: bool
    path: str


class CancellationResponse(BaseModel):
    accepted: bool
    execution_id: str


class AlignmentResponse(BaseModel):
    segments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recognizer transcript with word timestamps when available.",
    )
    aligned: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Canonical text aligned to recognized word timing.",
    )
