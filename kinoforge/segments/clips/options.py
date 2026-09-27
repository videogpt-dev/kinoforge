from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class JobOptions:
    clips: int = 10
    formats: List[str] = field(default_factory=lambda: ["9:16"])
    quality: str = "high"
    ai: Optional[str] = None
    captions: bool = True
    transcript_source: str = "auto"
    language: Optional[str] = None
    whisper_model: Optional[str] = None
    min_interest: Optional[float] = None
    mode: str = "auto"
    min_length: Optional[float] = None
    max_length: Optional[float] = None
    cookies: Optional[str] = None
    force: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "JobOptions":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})
