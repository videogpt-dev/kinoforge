"""Media schemas: per-scene image/video/music/voice renders, prompt preview, and the
shared-volume base-render / crop / format / align operations (with input validation)."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kinoforge.observ import active
from kinoforge.schemas.common import DefinitionBundleRequest, LoggableRequest, LogEntryResponse
from kinoforge.segments.clips.render.media import AspectRatio, FillStyle, MasterQuality


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
        default=False, description="Rewrite the prompt through a text model before generation."
    )
    enhance_style: str = Field(
        default="",
        description="Optional style to steer the prompt rewrite (ignored when enhance is off).",
    )


class ImageRenderRequest(LoggableRequest):
    job_id: str = Field(description="Caller-owned execution identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = Field(default="", description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    mode: Literal["scene", "character"] = Field(
        description="Render a single scene image or the character reference sheet.",
    )
    story: Dict[str, Any] = Field(
        default_factory=dict, description="Story record: characters, style, scenes."
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
        default_factory=dict, description="Resolved image route {provider, model}."
    )
    options: ImageOptionsRequest = Field(default_factory=ImageOptionsRequest)
    reference_b64: str = Field(default="", description="Optional reference image, base64 encoded.")
    prompt_limit: int = Field(
        default=0, ge=0,
        description="Provider prompt-length ceiling resolved by the caller; 0 when unknown.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen image presets resolved by the caller.",
    )


class ImageRenderResponse(BaseModel):
    image_b64: str = Field(description="Generated image, base64 encoded.")
    prompt: str = Field(default="", description="Final provider prompt, for tracing.")
    observed_limit: int = Field(
        default=0, description="Provider-declared prompt ceiling learned this call; 0 when none."
    )
    logs: List[LogEntryResponse] = Field(default_factory=list)


class VideoOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seconds: float = Field(gt=0, description="Seconds of film to buy for this clip.")
    resolution: int = Field(default=720, gt=0)
    aspect_ratio: str = "9:16"
    mature: bool = False
    enhance: bool = Field(
        default=False, description="Rewrite the prompt through a text model before generation."
    )
    enhance_style: str = Field(
        default="",
        description="Optional style to steer the prompt rewrite (ignored when enhance is off).",
    )


class VideoRenderRequest(LoggableRequest):
    job_id: str = Field(description="Caller-owned execution identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
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
        default_factory=dict, description="Resolved video route {provider, model}."
    )
    options: VideoOptionsRequest = Field(description="Clip duration and framing.")
    image_b64: str = Field(default="", description="Optional seed still image, base64 encoded.")
    prompt_limit: int = Field(
        default=0, ge=0,
        description="Provider prompt-length ceiling resolved by the caller; 0 when unknown.",
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen image presets (with motion) resolved by the caller.",
    )


class VideoRenderResponse(BaseModel):
    video_b64: str = Field(description="Generated clip (mp4), base64 encoded.")
    observed_limit: int = Field(
        default=0, description="Provider-declared prompt ceiling learned this call; 0 when none."
    )
    logs: List[LogEntryResponse] = Field(default_factory=list)


class MusicRenderRequest(LoggableRequest):
    job_id: str = Field(description="Caller-owned execution identifier.")
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = Field(default="", description="Caller-owned durable project identifier.")
    owner: str = Field(default="", description="Opaque tenant or owner identifier.")
    story: Dict[str, Any] = Field(
        default_factory=dict,
        description="Story record: style, logline and scenes (sizes the bed).",
    )
    music_key: str = Field(description="Resolved music preset key to render the mood from.")
    pick: Dict[str, Any] = Field(
        default_factory=dict, description="Resolved music route {provider, model}."
    )
    definitions: DefinitionBundleRequest = Field(
        description="Frozen music preset resolved by the caller.",
    )


class MusicRenderResponse(BaseModel):
    music_b64: str = Field(description="Generated music track, base64 encoded.")
    logs: List[LogEntryResponse] = Field(default_factory=list)


class VoiceOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: str = ""
    language: str = ""


class VoiceRenderRequest(LoggableRequest):
    job_id: str
    idempotency_key: str = Field(
        default="", description="Optional caller key correlating this run (and retries) in logs."
    )
    project_id: str = ""
    owner: str = ""
    text: str = Field(min_length=1, description="Narration text to synthesize.")
    pick: Dict[str, Any] = Field(
        default_factory=dict, description="Resolved TTS route {provider, model}."
    )
    options: VoiceOptionsRequest = Field(default_factory=VoiceOptionsRequest)


class VoiceRenderResponse(BaseModel):
    audio_b64: str = Field(description="Generated speech audio, base64 encoded.")
    logs: List[LogEntryResponse] = Field(default_factory=list)


class StoryPromptPreviewRequest(LoggableRequest):
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


class CropRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "CropRequest":
        if self.x + self.w > 1.001 or self.y + self.h > 1.001:
            active().warning(
                "rejected crop: falls outside source frame", x=self.x, y=self.y, w=self.w, h=self.h
            )
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
        default=None, description="Optional crop coordinates produced by clip analysis."
    )
    src_width: int = Field(gt=1, description="Source width in pixels.")
    src_height: int = Field(gt=1, description="Source height in pixels.")
    quality: MasterQuality = Field(
        default=MasterQuality.HIGH, description="Render quality preset."
    )

    @model_validator(mode="after")
    def validate_window(self) -> "BaseRenderRequest":
        if self.out_sec <= self.in_sec:
            active().warning(
                "rejected base render: out_sec must exceed in_sec",
                in_sec=self.in_sec, out_sec=self.out_sec,
            )
            raise ValueError("out_sec must be greater than in_sec")
        return self


class FormatRequest(BaseModel):
    source: str = Field(description="Source media path below shared root.")
    output: str = Field(description="Destination media path below shared root.")
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.PORTRAIT, description="Target aspect ratio."
    )
    fill: FillStyle = Field(default=FillStyle.BLUR, description="Background fill mode.")


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
