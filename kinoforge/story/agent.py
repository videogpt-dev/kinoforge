"""Story-generation core: turn a title (+ optional description) into a StoryPlan
the rest of the AI video pipeline builds on.

A story is written by a *story agent* — a pipeline of one or more LLM stages
(researcher -> writer -> art-director ...). The final `story_json` stage returns a
structured scene breakdown: a logline, a reusable visual style + characters for
cross-scene consistency, and per-scene image prompt + narration. The
strict-JSON schema is appended from the `contract` fragment, so a user-authored
agent shapes the persona and direction while the shared contract keeps every
stage's output parseable.

Stage execution lives in `run_agent.RunAgent`; this module owns shared context, text
calls and the showrunner episode planner.
"""

import json
import re
from enum import StrEnum
from typing import Dict, List, Optional, Tuple

from kinoforge.story.formats import VideoAspectRatio, prompt_label
from kinoforge.story.languages import language_name
from kinoforge.story.ports import StoryPorts

# Em dash, en dash, horizontal bar. The house rule is commas or periods, never these — a
# generated story or suggestion that carries one reads as machine-written, so every text
# chokepoint runs it through strip_dashes before the copy reaches the user.
_DASH_RE = re.compile(r"\s*[—–―]+\s*")


class AudienceMode(StrEnum):
    GENERAL = "general"
    MATURE = "mature"


def strip_dashes(text: str) -> str:
    """Replace em/en/bar dashes (with any surrounding spaces) by a comma and a space."""
    return _DASH_RE.sub(", ", text or "")


def chat(
    ports: StoryPorts,
    purpose: str,
    system: str,
    user: str,
    pick: Optional[Dict] = None,
    project: str = "",
    model: str = "",
    provider: str = "",
    temperature: float = 0.85,
    max_tokens: int = 3000,
    bill: bool = True,
) -> Tuple[str, Dict]:
    """One text call, through the studio's single text client — which resolves the engine
    for this purpose, retries a rate-limited model, records what it cost and traces what
    was said. `provider`/`model` are a stage's explicit override and outrank the project's
    pin. A model belongs to one provider, so a stage that names a model should name its
    provider too; naming the provider alone gets that provider's default model."""
    over = dict(pick or {})
    if provider:
        over["provider"] = provider
    if model:
        over["model"] = model
    return ports.complete(
        purpose,
        system or ports.prompt("story_system"),
        user,
        pick=over,
        project=project,
        temperature=temperature,
        max_tokens=max_tokens,
        bill=bill,
    )


class Screenwriter:
    """Context builder and showrunner for a story agent, over injected ports."""

    def __init__(self, ports: StoryPorts) -> None:
        self.ports = ports

    def build_context(
        self,
        title: str,
        description: str = "",
        scene_count: int = 8,
        aspect_ratio: object = VideoAspectRatio.PORTRAIT,
        language: str = "",
        genre: str = "",
        cast: Optional[List[Dict]] = None,
        premise: str = "",
        series_name: str = "",
        series_episodes: Optional[List[Dict]] = None,
        series_position: int = 0,
        mature: bool = False,
    ) -> Dict:
        """The runtime facts an agent writes from (independent of which agent runs).
        `cast`/`premise` are set for Series episodes so the writer reuses the show's
        recurring characters instead of inventing new ones."""
        scene_count = max(1, int(scene_count or 8))
        cast = cast or []
        series_cast = [
            {
                "name": str(character.get("name") or "").strip(),
                "look": str(character.get("look") or "").strip(),
                "personality": str(character.get("personality") or "").strip(),
            }
            for character in cast
            if str(character.get("name") or "").strip()
        ]
        language_rule = self.ports.language_rule(
            "logline and all narration", language_name(language)
        )
        return {
            "title": (title or "").strip(),
            "description": (description or "").strip(),
            "scene_count": scene_count,
            "output_format": prompt_label(aspect_ratio),
            "language": (language or "").strip(),
            "language_rule": language_rule,
            "genre": (genre or "").strip(),
            "premise": (premise or "").strip(),
            "series_name": (series_name or "").strip(),
            "series_cast": json.dumps(series_cast, ensure_ascii=False),
            "series_history": json.dumps(series_episodes or [], ensure_ascii=False),
            "series_position": max(0, int(series_position or 0)),
            "audience": AudienceMode.MATURE.value if mature else AudienceMode.GENERAL.value,
        }
