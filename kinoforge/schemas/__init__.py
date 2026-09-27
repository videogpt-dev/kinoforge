"""Kinoforge service request/response schemas, grouped by domain.

Split from a single 665-line module into per-domain files; this package re-exports every
name so `from kinoforge.schemas import X` is unchanged for routers and runtimes."""

from kinoforge.schemas.clips import (
    ClipsExecutionRequest,
    ClipsExecutionResponse,
    ClipsOptionsRequest,
)
from kinoforge.schemas.common import (
    AlignmentResponse,
    ArtifactResponse,
    CancellationResponse,
    DefinitionBundleRequest,
    DefinitionRequest,
    EngineCompatibilityRequest,
    ExecutionResultResponse,
    ExecutionState,
    ExecutionStateResponse,
    HealthResponse,
    LogEntryResponse,
    MediaMetadataResponse,
    MediaOperationResponse,
    MeterEventResponse,
    PathRequest,
)
from kinoforge.schemas.discovery import (
    SegmentDefinitionRequirementResponse,
    SegmentFeatureResponse,
    SegmentResponse,
    SegmentsResponse,
    SegmentStatus,
)
from kinoforge.schemas.media import (
    AlignRequest,
    BaseRenderRequest,
    CropRequest,
    FormatRequest,
    ImageOptionsRequest,
    ImagePresetKeysRequest,
    ImageRenderRequest,
    ImageRenderResponse,
    MusicRenderRequest,
    MusicRenderResponse,
    StoryPromptPreviewRequest,
    StoryPromptPreviewResponse,
    VideoOptionsRequest,
    VideoRenderRequest,
    VideoRenderResponse,
    VoiceOptionsRequest,
    VoiceRenderRequest,
    VoiceRenderResponse,
)
from kinoforge.schemas.series import (
    SeriesContextRequest,
    SeriesPlanRequest,
    SeriesPlanResponse,
)
from kinoforge.schemas.story import (
    StoryContextRequest,
    StoryOperation,
    StoryOperationRequest,
    StoryOperationResponse,
    StoryWriteRequest,
    StoryWriteResponse,
)

__all__ = [
    # common
    "EngineCompatibilityRequest", "DefinitionRequest", "DefinitionBundleRequest",
    "ExecutionState", "PathRequest", "HealthResponse", "MeterEventResponse",
    "LogEntryResponse", "ArtifactResponse", "ExecutionResultResponse",
    "ExecutionStateResponse", "MediaMetadataResponse", "MediaOperationResponse",
    "CancellationResponse", "AlignmentResponse",
    # discovery
    "SegmentStatus", "SegmentFeatureResponse", "SegmentDefinitionRequirementResponse",
    "SegmentResponse", "SegmentsResponse",
    # clips
    "ClipsOptionsRequest", "ClipsExecutionRequest", "ClipsExecutionResponse",
    # story
    "StoryContextRequest", "StoryWriteRequest", "StoryWriteResponse", "StoryOperation",
    "StoryOperationRequest", "StoryOperationResponse",
    # series
    "SeriesContextRequest", "SeriesPlanRequest", "SeriesPlanResponse",
    # media
    "ImagePresetKeysRequest", "ImageOptionsRequest", "ImageRenderRequest",
    "ImageRenderResponse", "VideoOptionsRequest", "VideoRenderRequest",
    "VideoRenderResponse", "MusicRenderRequest", "MusicRenderResponse",
    "VoiceOptionsRequest", "VoiceRenderRequest", "VoiceRenderResponse",
    "StoryPromptPreviewRequest", "StoryPromptPreviewResponse", "CropRequest",
    "BaseRenderRequest", "FormatRequest", "AlignRequest",
]
