"""Story-generation core: turn a brief into the context a story agent writes from.

A story is written by a *story agent*, a pipeline of one or more LLM stages (researcher,
writer, art-director...). The final `story_json` stage returns the structured scene
breakdown; its strict schema comes from the `contract` fragment. Stage execution lives in
`run_agent.RunAgent`.
"""

import json
import re
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any, Dict, List, Mapping

from kinoforge.segments.story.formats import VideoAspectRatio, prompt_label
from kinoforge.segments.story.languages import language_name
from kinoforge.segments.story.ports import StoryPorts

# Em dash, en dash, horizontal bar: house rule is commas or periods, so every text chokepoint
# runs generated copy through strip_dashes.
_DASH_RE = re.compile(r"\s*[—–―]+\s*")


class AudienceMode(StrEnum):
    GENERAL = "general"
    MATURE = "mature"


def strip_dashes(text: str) -> str:
    """Replace em/en/bar dashes (with any surrounding spaces) by a comma and a space."""
    return _DASH_RE.sub(", ", text or "")


def clean_text(value: object) -> str:
    return strip_dashes(str(value or "")).strip()


@dataclass(frozen=True)
class StoryBrief:
    """What the caller asks a story agent to write. cast/premise/series_* are set for Series
    episodes so the writer reuses the show's recurring characters."""

    title: str
    description: str = ""
    scene_count: int = 8
    aspect_ratio: object = VideoAspectRatio.PORTRAIT
    language: str = ""
    genre: str = ""
    cast: List[Dict] = field(default_factory=list)
    premise: str = ""
    series_name: str = ""
    series_episodes: List[Dict] = field(default_factory=list)
    series_position: int = 0
    mature: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StoryBrief":
        known = {f.name for f in fields(cls)}
        return cls(**{key: val for key, val in value.items() if key in known})


class Screenwriter:
    """Context builder for a story agent, over injected ports."""

    def __init__(self, ports: StoryPorts) -> None:
        self.ports = ports

    def build_context(self, brief: StoryBrief) -> Dict:
        """The runtime facts an agent writes from (independent of which agent runs)."""
        series_cast = [
            {
                "name": str(character.get("name") or "").strip(),
                "look": str(character.get("look") or "").strip(),
                "personality": str(character.get("personality") or "").strip(),
            }
            for character in brief.cast or []
            if str(character.get("name") or "").strip()
        ]
        return {
            "title": (brief.title or "").strip(),
            "description": (brief.description or "").strip(),
            "scene_count": max(1, int(brief.scene_count or 8)),
            "output_format": prompt_label(brief.aspect_ratio),
            "language": (brief.language or "").strip(),
            "language_rule": self.ports.language_rule(
                "logline and all narration", language_name(brief.language)
            ),
            "genre": (brief.genre or "").strip(),
            "premise": (brief.premise or "").strip(),
            "series_name": (brief.series_name or "").strip(),
            "series_cast": json.dumps(series_cast, ensure_ascii=False),
            "series_history": json.dumps(brief.series_episodes or [], ensure_ascii=False),
            "series_position": max(0, int(brief.series_position or 0)),
            "audience": (AudienceMode.MATURE if brief.mature else AudienceMode.GENERAL).value,
        }
