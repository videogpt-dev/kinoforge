from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.contract import Meter
from kinoforge.segments.common import Worker

Segments = List[Dict[str, Any]]


class ClipStage(StrEnum):
    TRANSCRIBE = "transcribe"
    MOMENTS = "moments"
    CLIPS = "clips"


@dataclass(kw_only=True)
class ClipWorker(Worker):
    """Everything one clips execution reads and writes: inputs, merged config, the state a
    previous run left (record, transcript, has_clips), and the result being built."""

    job_id: str
    config: Dict[str, Any]
    workdir: Path
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    meter: Optional[Meter] = None
    record: Optional[Dict[str, Any]] = None
    transcript: Segments = field(default_factory=list)
    has_clips: bool = False
    clip_ids: List[Optional[str]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = {
            "video_path": str(self.video_path or ""),
            "audio_path": str(self.audio_path or ""),
            "clips": [], "moments": [], "transcript": None,
            **self.data,
        }
        super().__post_init__()

    @property
    def media_path(self) -> Path:
        """What the find stages read: the audio when the host sent it, else the video."""
        path = self.audio_path or self.video_path
        if path is None:
            raise ValueError("clips job requires input.video_path or input.audio_path")
        return path

    @property
    def slug(self) -> str:
        return str(self.config.get("slug") or "")
