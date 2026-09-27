"""Segment discovery schemas: what /v1/segments advertises to callers."""

from enum import StrEnum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SegmentStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"


class SegmentFeatureResponse(BaseModel):
    code_name: str = Field(description="Stable feature identifier.")
    name: str
    description: str
    icon: str = Field(description="Frontend-neutral Lucide icon name.")
    status: SegmentStatus


class SegmentDefinitionRequirementResponse(BaseModel):
    type: Literal["screenwriter", "agent", "preset", "fragment"]
    key: str = Field(description="Stable catalog definition key.")
    purpose: str
    required: bool = True


class SegmentResponse(BaseModel):
    id: str = Field(description="Stable segment identifier.")
    code_name: str = Field(description="Code-facing JobKind value.")
    name: str = Field(description="Short product name.")
    label: str = Field(description="Human-readable product label.")
    description: str
    icon: str = Field(description="Frontend-neutral Lucide icon name.")
    status: SegmentStatus
    features: List[SegmentFeatureResponse]
    definition_requirements: List[SegmentDefinitionRequirementResponse] = Field(
        default_factory=list,
        description="Definition identifiers caller must resolve and send in execution bundle.",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="Current REST API version when implemented.",
    )
    execute_path: Optional[str] = Field(
        default=None,
        description="Execution endpoint when implemented.",
    )


class SegmentsResponse(BaseModel):
    segments: List[SegmentResponse]
