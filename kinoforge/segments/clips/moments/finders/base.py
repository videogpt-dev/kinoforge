from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from kinoforge.segments.clips.moments.moment import Moment


@dataclass(frozen=True)
class MomentSpec:
    min_len: float
    max_len: float
    count: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MomentSpec":
        return cls(float(config["min_length"]), float(config["max_length"]),
                   int(config["clip_count"]))


class MomentFinder(ABC):
    """Reads a transcript and returns moments ranked best first."""

    name: str = ""

    @abstractmethod
    def find(self, transcript: List[Dict], spec: MomentSpec) -> List[Dict]: ...

    @staticmethod
    def ranked(moments: List[Dict]) -> List[Dict]:
        return sorted(moments, key=lambda m: Moment(m).score, reverse=True)
