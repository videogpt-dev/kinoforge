from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable


class AITask(StrEnum):
    """What an inference call is for; the gateway routes each task (Infrelay's AITask). The
    kinds are the general-purpose tasks."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MUSIC = "music"
    TRANSCRIBE = "transcribe"
    VIRAL_CLIP_TRANSCRIBE = "viralClip.transcribe"
    VIRAL_CLIP_FIND = "viralClip.find"
    STORY_WRITE = "story.write"
    STORY_IDEAS = "story.ideas"
    STORY_SUGGEST = "story.suggest"
    STORY_DIRECT = "story.direct"
    STORY_TRANSLATE = "story.translate"
    STORY_CAST = "story.cast"
    STORY_IMAGE = "story.image"
    STORY_VIDEO = "story.video"
    STORY_VOICE = "story.voice"
    STORY_MUSIC = "story.music"
    STORY_CAPTIONS = "story.captions"


AUTO = "auto"


@dataclass(frozen=True)
class ModelRef:
    """One inference pick: a model (blank is `auto`, the task's route on the gateway) on a
    provider (blank is any that serves it), with fallback picks tried after it and models the
    gateway must not try."""

    provider: str = ""
    model: str = ""
    fallback: Tuple["ModelRef", ...] = ()
    exclude: Tuple["ModelRef", ...] = ()

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ModelRef":
        value = value or {}
        fallback = value.get("fallback") or ()
        return cls(
            cls._clean(value.get("provider")),
            cls._clean(value.get("model")),
            tuple(cls.from_mapping(item) for item in fallback if isinstance(item, Mapping)),
        )

    @staticmethod
    def _clean(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text.lower() == AUTO else text

    @property
    def auto(self) -> bool:
        return not self.model

    def with_overrides(self, provider: str = "", model: str = "") -> "ModelRef":
        if not (provider.strip() or model.strip()):
            return self
        return ModelRef(provider.strip() or self.provider, model.strip() or self.model)

    def excluding(self, *refs: "ModelRef") -> "ModelRef":
        """This pick, never asking those models again: an auto pick moves to the route's next
        step."""
        return replace(self, exclude=tuple(dict.fromkeys(self.exclude + refs)))

    def inference(self) -> Dict[str, Any]:
        """The gateway's `inference` field for this pick."""
        out: Dict[str, Any] = {"model": self.model or AUTO, "provider": self.provider or AUTO}
        for name, refs in (("fallback", self.fallback), ("exclude", self.exclude)):
            if refs:
                out[name] = [{"model": r.model or AUTO, "provider": r.provider or AUTO}
                             for r in refs]
        return out

    def __str__(self) -> str:
        return f"{self.provider or AUTO}/{self.model or AUTO}"


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
    AGENT_RUN = "agent_run"


@runtime_checkable
class Meter(Protocol):
    def __call__(self, action: MeterAction, qty: float, variant: str = "") -> None: ...


def no_meter(action: MeterAction, qty: float, variant: str = "") -> None:
    return None
