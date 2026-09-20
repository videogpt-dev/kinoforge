from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        description="Existing project record supplied by cloud core when resuming execution.",
    )
    transcript: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Existing transcript segments supplied when transcription already completed.",
    )
    has_clips: bool = Field(
        default=False,
        description="Whether cloud storage already contains rendered clips for this job.",
    )


class ClipsOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    clip_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of ranked moments returned or rendered.",
    )
    min_length: float = Field(
        default=20,
        gt=0,
        description="Preferred minimum clip duration in seconds.",
    )
    max_length: float = Field(
        default=60,
        gt=0,
        description="Maximum clip duration in seconds.",
    )
    formats: List[str] = Field(
        default_factory=lambda: ["9:16"],
        description="Output aspect ratios such as 9:16, 16:9, or 1:1.",
    )
    quality: Literal["high", "medium", "low"] = "high"
    generate_captions: bool = True
    analyze_only: bool = Field(
        default=False,
        description="Find and rank moments without rendering clip files.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional BCP 47 transcription language; null enables detection.",
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
            raise ValueError("max_length must be greater than min_length")
        if not self.formats:
            raise ValueError("at least one output format is required")
        return self


class ClipsExecutionRequest(BaseModel):
    job_id: str = Field(description="Caller-owned execution identifier.")
    project_id: str = Field(description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    workspace: str = Field(description="Execution directory below KINOFORGE_SHARED_ROOT.")
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Clip job input. video_path must point inside shared root.",
    )
    options: ClipsOptionsRequest = Field(
        default_factory=ClipsOptionsRequest,
        description="Per-job clip generation choices. Cloud Core or self-host runtime owns limits.",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved engine, transcription, rendering, scoring, and route configuration.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen definitions resolved by cloud core or self-host runtime.",
    )
    state: ExecutionState = Field(
        default_factory=ExecutionState,
        description="Optional durable state restored by cloud core.",
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


class StoryWriteRequest(BaseModel):
    job_id: str = Field(description="Caller-owned execution identifier.")
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
    meter_events: List["MeterEventResponse"] = Field(default_factory=list)
    logs: List["LogEntryResponse"] = Field(default_factory=list)


class StoryOperation(StrEnum):
    REWRITE_SCENE = "rewrite_scene"
    REWRITE_CHARACTERS = "rewrite_characters"
    TRANSLATE = "translate"
    DIRECT_SHOTS = "direct_shots"
    SUGGEST_FIELD = "suggest_field"


class StoryOperationRequest(BaseModel):
    job_id: str = Field(description="Caller-owned operation identifier.")
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
        default_factory=dict,
        description="Resolved rewrite and translation token budgets.",
    )
    definitions: DefinitionBundleRequest


class StoryOperationResponse(BaseModel):
    result: Dict[str, Any]
    meter_events: List["MeterEventResponse"] = Field(default_factory=list)
    logs: List["LogEntryResponse"] = Field(default_factory=list)


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


class SeriesPlanRequest(BaseModel):
    job_id: str = Field(description="Caller-owned operation identifier.")
    project_id: str = Field(default="", description="Caller-owned durable series identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    series: SeriesContextRequest = Field(description="Premise and cast the showrunner plans from.")
    count: int = Field(default=6, ge=1, le=12, description="How many episode ideas to propose.")
    pick: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved default text route {provider, model}.",
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
    meter_events: List["MeterEventResponse"] = Field(default_factory=list)
    logs: List["LogEntryResponse"] = Field(default_factory=list)


class ImagePresetKeysRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: str = Field(description="Default scene look preset key.")
    portrait: str = Field(description="Character portrait preset key.")
    negative: str = Field(description="Negative preset key.")


class ImageOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspect_ratio: str = "9:16"
    mature: bool = False
    seed: Optional[int] = None
    apply_negative: bool = Field(
        default=False,
        description="Attach the negative preset to the provider request (scene mode only).",
    )
    enhance: bool = Field(
        default=False,
        description="Rewrite the prompt through a text model before generation.",
    )
    enhance_style: str = Field(
        default="",
        description="Optional style to steer the prompt rewrite (ignored when enhance is off).",
    )


class ImageRenderRequest(BaseModel):
    job_id: str = Field(description="Caller-owned execution identifier.")
    project_id: str = Field(default="", description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    mode: Literal["scene", "character"] = Field(
        description="Render a single scene image or the character reference sheet.",
    )
    story: Dict[str, Any] = Field(
        default_factory=dict,
        description="Story record: characters, style, scenes.",
    )
    scene: Dict[str, Any] = Field(
        default_factory=dict,
        description="The scene to illustrate (scene mode). Ignored for character mode.",
    )
    rec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal project shape {input: {image_preset, ...}} the prompt builder reads.",
    )
    preset_keys: ImagePresetKeysRequest = Field(description="Resolved preset keys to render from.")
    pick: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved image route {provider, model}.",
    )
    options: ImageOptionsRequest = Field(default_factory=ImageOptionsRequest)
    reference_b64: str = Field(default="", description="Optional reference image, base64 encoded.")
    prompt_limit: int = Field(
        default=0,
        ge=0,
        description="Provider prompt-length ceiling resolved by the caller; 0 when unknown.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen image presets resolved by the caller.",
    )


class ImageRenderResponse(BaseModel):
    image_b64: str = Field(description="Generated image, base64 encoded.")
    prompt: str = Field(default="", description="Final provider prompt, for tracing.")
    observed_limit: int = Field(
        default=0,
        description="Provider-declared prompt ceiling learned this call; 0 when none.",
    )
    logs: List["LogEntryResponse"] = Field(default_factory=list)


class VideoOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seconds: float = Field(gt=0, description="Seconds of film to buy for this clip.")
    resolution: int = Field(default=720, gt=0)
    aspect_ratio: str = "9:16"
    mature: bool = False
    enhance: bool = Field(
        default=False,
        description="Rewrite the prompt through a text model before generation.",
    )
    enhance_style: str = Field(
        default="",
        description="Optional style to steer the prompt rewrite (ignored when enhance is off).",
    )


class VideoRenderRequest(BaseModel):
    job_id: str = Field(description="Caller-owned execution identifier.")
    project_id: str = Field(default="", description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    story: Dict[str, Any] = Field(
        default_factory=dict, description="Story record: characters, style."
    )
    scene: Dict[str, Any] = Field(
        default_factory=dict,
        description="The scene to animate, carrying its resolved shot direction.",
    )
    rec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal project shape {input: {image_preset, ...}} the prompt builder reads.",
    )
    preset_keys: ImagePresetKeysRequest = Field(description="Resolved preset keys to render from.")
    pick: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved video route {provider, model}.",
    )
    options: VideoOptionsRequest = Field(description="Clip duration and framing.")
    image_b64: str = Field(default="", description="Optional seed still image, base64 encoded.")
    prompt_limit: int = Field(
        default=0,
        ge=0,
        description="Provider prompt-length ceiling resolved by the caller; 0 when unknown.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen image presets (with motion) resolved by the caller.",
    )


class VideoRenderResponse(BaseModel):
    video_b64: str = Field(description="Generated clip (mp4), base64 encoded.")
    observed_limit: int = Field(
        default=0,
        description="Provider-declared prompt ceiling learned this call; 0 when none.",
    )
    logs: List["LogEntryResponse"] = Field(default_factory=list)


class MusicRenderRequest(BaseModel):
    job_id: str = Field(description="Caller-owned execution identifier.")
    project_id: str = Field(default="", description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    story: Dict[str, Any] = Field(
        default_factory=dict,
        description="Story record: style, logline and scenes (sizes the bed).",
    )
    music_key: str = Field(description="Resolved music preset key to render the mood from.")
    pick: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved music route {provider, model}.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen music preset resolved by the caller.",
    )


class MusicRenderResponse(BaseModel):
    music_b64: str = Field(description="Generated music track, base64 encoded.")
    logs: List["LogEntryResponse"] = Field(default_factory=list)


class VoiceOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: str = ""
    language: str = ""


class VoiceRenderRequest(BaseModel):
    job_id: str
    project_id: str = ""
    owner: str = ""
    text: str = Field(min_length=1, description="Narration text to synthesize.")
    pick: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved TTS route {provider, model}.",
    )
    options: VoiceOptionsRequest = Field(default_factory=VoiceOptionsRequest)


class VoiceRenderResponse(BaseModel):
    audio_b64: str = Field(description="Generated speech audio, base64 encoded.")
    logs: List["LogEntryResponse"] = Field(default_factory=list)


class StoryPromptPreviewRequest(BaseModel):
    owner: str = ""
    story: Dict[str, Any] = Field(default_factory=dict)
    scene: Dict[str, Any] = Field(default_factory=dict)
    rec: Dict[str, Any] = Field(default_factory=dict)
    preset_keys: ImagePresetKeysRequest
    image_limit: int = Field(default=0, ge=0)
    video_limit: int = Field(default=0, ge=0)
    definitions: DefinitionBundleRequest


class StoryPromptPreviewResponse(BaseModel):
    image_prompt: str
    negative_prompt: str
    video_prompt: str


class PathRequest(BaseModel):
    path: str = Field(description="Media path below KINOFORGE_SHARED_ROOT.")


class CropRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "CropRequest":
        if self.x + self.w > 1.001 or self.y + self.h > 1.001:
            raise ValueError("crop must fit inside source frame")
        return self


class BaseRenderRequest(BaseModel):
    execution_id: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
        description="Caller-owned render ID used for cooperative cancellation.",
    )
    source: str = Field(description="Source media path below shared root.")
    output: str = Field(description="Destination media path below shared root.")
    in_sec: float = Field(ge=0, description="Clip start time in seconds.")
    out_sec: float = Field(gt=0, description="Clip end time in seconds.")
    crop: Optional[CropRequest] = Field(
        default=None,
        description="Optional crop coordinates produced by clip analysis.",
    )
    src_width: int = Field(gt=1, description="Source width in pixels.")
    src_height: int = Field(gt=1, description="Source height in pixels.")
    quality: Literal["max", "high", "standard"] = Field(
        default="high",
        description="Render quality preset.",
    )

    @model_validator(mode="after")
    def validate_window(self) -> "BaseRenderRequest":
        if self.out_sec <= self.in_sec:
            raise ValueError("out_sec must be greater than in_sec")
        return self


class FormatRequest(BaseModel):
    source: str = Field(description="Source media path below shared root.")
    output: str = Field(description="Destination media path below shared root.")
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = Field(
        default="9:16",
        description="Target aspect ratio.",
    )
    fill: Literal["blur", "bars"] = Field(default="blur", description="Background fill mode.")


class AlignRequest(BaseModel):
    path: str = Field(description="Narration media path below shared root.")
    text: str = Field(description="Canonical narration text to align.")
    duration: float = Field(description="Narration duration in seconds.")
    language: str = Field(default="", description="BCP 47 language hint; empty enables detection.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    transcription: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved transcription provider, model, and tuning.",
    )


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


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


class MeterEventResponse(BaseModel):
    action: str
    qty: float
    variant: str = ""


class LogEntryResponse(BaseModel):
    level: Literal["info", "success", "warning", "error"]
    text: str


class ClipsExecutionResponse(BaseModel):
    result: ExecutionResultResponse
    state: ExecutionStateResponse
    meter_events: List[MeterEventResponse] = Field(default_factory=list)
    logs: List[LogEntryResponse] = Field(default_factory=list)


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
