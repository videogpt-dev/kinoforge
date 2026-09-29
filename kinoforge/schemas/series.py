"""Series segment schemas: showrunner context, plan request, and plan response."""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from kinoforge.schemas.common import (
    DefinitionBundleRequest,
    LoggableRequest,
    LogEntryResponse,
    MeterEventResponse,
)


class SeriesContextRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    premise: str = ""
    style: str = ""
    aspect_ratio: str = "9:16"
    cast: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recurring cast, each carrying at least name and look.",
    )
    episodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Existing ordered episode summaries supplied to the showrunner.",
    )


class SeriesPlanRequest(LoggableRequest):
    job_id: str = Field(description="Caller-owned operation identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = Field(default="", description="Caller-owned durable series identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    series: SeriesContextRequest = Field(description="Premise and cast the showrunner plans from.")
    count: int = Field(default=6, ge=1, le=12, description="How many episode ideas to propose.")
    pick: Dict[str, Any] = Field(
        default_factory=dict, description="Resolved default text route {provider, model}."
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved output-token budgets and other runtime config.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen showrunner prompts resolved by the caller.",
    )


class SeriesPlanResponse(BaseModel):
    result: Dict[str, Any] = Field(
        description="Plan result: {ok, episodes} or {ok: False, error}.",
    )
    meter_events: List[MeterEventResponse] = Field(default_factory=list)
    logs: List[LogEntryResponse] = Field(default_factory=list)
