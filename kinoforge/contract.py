from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelRef:
    """One inference route: which provider serves the call and which of its models."""

    provider: str = ""
    model: str = ""

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ModelRef":
        value = value or {}
        return cls(str(value.get("provider") or "").strip(), str(value.get("model") or "").strip())

    def with_overrides(self, provider: str = "", model: str = "") -> "ModelRef":
        return ModelRef(provider.strip() or self.provider, model.strip() or self.model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model or 'default'}"


@dataclass(frozen=True)
class ImageSpec:
    aspect_ratio: str = "9:16"
    seed: Optional[int] = None
    mature: bool = False
    negative: str = ""
    reference: Optional[bytes] = None
    enhance: bool = False
    enhance_style: str = ""


@dataclass(frozen=True)
class VideoSpec:
    seconds: float
    resolution: int
    aspect_ratio: str
    mature: bool = False
    image: Optional[bytes] = None
    enhance: bool = False
    enhance_style: str = ""
    dialogue: str = ""
    music: bool = True


class JobKind(StrEnum):
    CLIPS = "clips"
    SERIES = "series"
    STORY = "story"


class MeterAction(StrEnum):
    CLIP_RENDER = "clip_render"
    CLIP_CAPTIONS = "clip_captions"
    CLIP_VARIANT = "clip_variant"
    TRANSCRIBE_MINUTE = "transcribe_minute"
    AGENT_RUN = "agent_run"


@runtime_checkable
class Meter(Protocol):
    def __call__(self, action: MeterAction, qty: float, variant: str = "") -> None: ...


def _no_meter(action: MeterAction, qty: float, variant: str = "") -> None:
    return None
