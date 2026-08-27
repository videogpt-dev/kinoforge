from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from kinoforge.definitions import DefinitionBundle


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
class ProjectStore(Protocol):
    def workdir(self, job_id: str) -> Path: ...
    def save_record(self, job_id: str, record: Dict[str, Any]) -> None: ...
    def load_record(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    def source_path(self, job_id: str) -> Optional[Path]: ...
    def transcript_path(self, job_id: str) -> Path: ...
    def save_transcript(self, job_id: str, transcript: List[Dict[str, Any]]) -> None: ...
    def load_transcript(self, job_id: str) -> Optional[List[Dict[str, Any]]]: ...
    def clips_path(self, job_id: str) -> Path: ...
    def list_clips(self, job_id: str) -> List[Dict[str, Any]]: ...
    def append_clips(
        self, job_id: str, project: Dict[str, Any], moments: List[Dict[str, Any]]
    ) -> List[Optional[str]]: ...
    def clip_path(self, job_id: str, clip_id: str) -> Path: ...
    def mark_rendered(
        self,
        job_id: str,
        clip_id: str,
        file: str = "clip.mp4",
        transcript: Optional[List[Dict[str, Any]]] = None,
    ) -> None: ...


@runtime_checkable
class Meter(Protocol):
    def __call__(self, action: MeterAction, qty: float, variant: str = "") -> None: ...


def _no_meter(action: MeterAction, qty: float, variant: str = "") -> None:
    return None


@dataclass
class Context:
    store: ProjectStore
    owner: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    definitions: DefinitionBundle = field(default_factory=DefinitionBundle.empty)
    infrelay_url: str = ""
    meter: Meter = _no_meter


@dataclass
class Job:
    kind: JobKind
    job_id: str
    input: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    path: Path
    media: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    status: str = "ok"
    artifacts: List[Artifact] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    kind: JobKind

    def run(self, job: Job, ctx: Context) -> Result: ...
