"""Stateful runtime for multi-stage story agents."""

import json
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple

from kinoforge.contract import MeterAction
from kinoforge.story.agent import AudienceMode, chat, strip_dashes
from kinoforge.story.ports import StoryPorts
from kinoforge.story.quality import validate_story

StageCallback = Callable[[int, int, str], None]
CancellationChecker = Callable[[], bool]


def _not_cancelled() -> bool:
    return False


def _normalize(raw: Dict) -> Dict:
    def _s(value) -> str:
        return strip_dashes(str(value or "")).strip()

    characters = [
        {"name": _s(character.get("name")), "description": _s(character.get("description"))}
        for character in (raw.get("characters") or [])
        if isinstance(character, dict) and (character.get("name") or character.get("description"))
    ]

    scenes = []
    for scene in raw.get("scenes") or []:
        if not isinstance(scene, dict) or not _s(scene.get("prompt")):
            continue
        scenes.append(
            {
                "prompt": _s(scene.get("prompt")),
                "narration": _s(scene.get("narration")),
                "motion": bool(scene.get("motion")),
            }
        )

    return {
        "logline": _s(raw.get("logline")),
        "style": _s(raw.get("style")),
        "characters": characters,
        "scenes": scenes,
    }


def _stage_temperature(stage: Dict) -> float:
    try:
        return float(stage["temperature"])
    except (KeyError, TypeError, ValueError):
        return 0.85


class RunAgent:
    """Run one agent definition against one story context.

    Instance owns mutable state accumulated across stages: intermediate outputs,
    generated story, token totals and models used. Call `run()` for result shape
    expected by queue worker.
    """

    def __init__(
        self,
        ports: StoryPorts,
        agent: Dict,
        ctx: Dict,
        on_stage: Optional[StageCallback] = None,
        pick: Optional[Dict] = None,
        project: str = "",
        is_cancelled: Optional[CancellationChecker] = None,
    ) -> None:
        self.ports = ports
        self.agent = agent
        self.ctx = ctx
        self.on_stage = on_stage
        self.pick = dict(pick or {})
        self.project = project
        self.is_cancelled = is_cancelled or _not_cancelled
        self.stages: List[Dict] = []
        self.outputs: List[Tuple[str, str]] = []
        self.story: Optional[Dict] = None
        self.total_in = 0
        self.total_out = 0
        self.model_used = ""
        self.models_used: List[str] = []
        self.requested_models: List[str] = []

    def _reset(self) -> None:
        self.stages = [
            stage for stage in (self.agent.get("stages") or []) if isinstance(stage, dict)
        ]
        self.outputs = []
        self.story = None
        self.total_in = 0
        self.total_out = 0
        self.model_used = ""
        self.models_used = []
        self.requested_models = []

    def _usage_summary(self) -> Dict:
        return {
            "model": self.model_used,
            "models": self.models_used,
            "requested_models": self.requested_models,
            "input_tokens": self.total_in,
            "output_tokens": self.total_out,
        }

    def _build_prompt(self, stage: Dict, idea: str, is_json: bool) -> str:
        parts = [idea]
        parts += [f"{role.upper()}:\n{text}" for role, text in self.outputs]

        instructions = (stage.get("instructions") or "").strip()
        if instructions:
            parts.append(instructions)
        if is_json:
            parts.append(self.ctx["language_rule"])
            parts.append(self.ports.contract(self.ctx["scene_count"]))
        return "\n\n".join(part for part in parts if part)

    def _record_usage(self, usage: Dict) -> None:
        self.total_in += usage["input_tokens"]
        self.total_out += usage["output_tokens"]
        self.model_used = usage["model"]
        if self.model_used and self.model_used not in self.models_used:
            self.models_used.append(self.model_used)
        requested = usage.get("requested_model") or self.model_used
        if requested and requested not in self.requested_models:
            self.requested_models.append(requested)

    def _run_stage(self, stage: Dict, role: str, idea: str) -> Optional[Dict]:
        is_json = stage.get("format") == "story_json"
        prompt = self._build_prompt(stage, idea, is_json)
        budget = self.ports.story_budget(stage.get("max_output_tokens"))

        try:
            text, usage = chat(
                self.ports,
                "story",
                # One persona per agent, applied to every stage. Global default is the
                # last resort.
                self.agent.get("persona") or self.ports.prompt("story_system"),
                prompt,
                pick=self.pick,
                project=self.project,
                model=(stage.get("model") or "").strip(),
                provider=(stage.get("provider") or "").strip(),
                temperature=_stage_temperature(stage),
                max_tokens=budget,
            )
        except Exception as error:
            model = (stage.get("model") or "").strip() or self.pick.get("model") or ""
            provider = (stage.get("provider") or "").strip() or self.pick.get("provider") or ""
            which = (
                f" (model '{model}'" + (f" on {provider}" if provider else "") + ")"
                if model
                else ""
            )
            return {
                "ok": False,
                "error": (
                    f"Story '{role}' stage{which} failed: {error}. "
                    "Change this stage's model in the agent editor."
                ),
            }

        self._record_usage(usage)
        self.ports.meter(MeterAction.AGENT_RUN, 1)

        if is_json:
            # Parse first: a model can hit the token budget yet still emit complete, valid
            # JSON, so trust the payload over finish_reason and only complain if it truly
            # does not parse. When it does not, the finish_reason picks the message.
            try:
                self.story = _normalize(self.ports.parse_json(text))
            except (ValueError, json.JSONDecodeError):
                if usage.get("finish_reason") == "length":
                    return {
                        "ok": False,
                        "error": (
                            f"the story was cut off at {usage['output_tokens']} tokens. Ask for "
                            "fewer scenes, or raise the stage's token budget. A different model "
                            "will not help."
                        ),
                        "usage": self._usage_summary(),
                    }
                return {
                    "ok": False,
                    "error": (
                        "the model did not return valid JSON. Retry, "
                        "or pick another model in Settings."
                    ),
                    "usage": self._usage_summary(),
                }

        self.outputs.append((role, text))
        return None

    def run(self) -> Dict:
        """Execute stages in order and return {ok, story, usage} or error."""
        self._reset()
        if not self.stages:
            return {"ok": False, "error": "agent has no stages"}
        if not self.ctx.get("title"):
            return {"ok": False, "error": "title is required"}

        idea = self.ports.prompt(
            "story_idea",
            title=self.ctx["title"],
            description=self.ctx["description"],
            scene_count=self.ctx["scene_count"],
            format=self.ctx["output_format"],
            genre=self.ctx.get("genre", ""),
            premise=self.ctx.get("premise", ""),
            series_name=self.ctx.get("series_name", ""),
            series_cast=self.ctx.get("series_cast", "[]"),
            series_history=self.ctx.get("series_history", "[]"),
            series_position=self.ctx.get("series_position", 0),
            audience=self.ctx.get("audience", AudienceMode.GENERAL.value),
        )

        for index, stage in enumerate(self.stages):
            if self.is_cancelled():
                return {"ok": False, "cancelled": True, "error": "story execution cancelled"}
            role = (stage.get("role") or f"stage {index + 1}").strip()
            if self.on_stage:
                self.on_stage(index, len(self.stages), role)
            failure = self._run_stage(stage, role, idea)
            if failure:
                return failure
            if self.is_cancelled():
                return {
                    "ok": False,
                    "cancelled": True,
                    "error": "story execution cancelled",
                    "usage": self._usage_summary(),
                }

        if self.story is None:
            try:
                self.story = _normalize(self.ports.parse_json(self.outputs[-1][1]))
            except (ValueError, json.JSONDecodeError):
                return {
                    "ok": False,
                    "error": "agent produced no valid story JSON",
                    "usage": self._usage_summary(),
                }

        if not self.story["scenes"]:
            return {
                "ok": False,
                "error": "story had no usable scenes",
                "usage": self._usage_summary(),
            }

        blocking, advisory = validate_story(self.story, self.ctx)
        if blocking:
            return {
                "ok": False,
                "error": (
                    "Story can't be filmed: "
                    + "; ".join(blocking)
                    + ". Retry, or change the story model in Settings."
                ),
                "usage": self._usage_summary(),
            }

        return {
            "ok": True,
            "story": self.story,
            "warnings": advisory,
            "usage": self._usage_summary(),
        }


def run_agent(
    ports: StoryPorts,
    agent: Dict,
    ctx: Dict,
    on_stage: Optional[StageCallback] = None,
    pick: Optional[Dict] = None,
    project: str = "",
) -> Dict:
    """Compatibility function for existing queue and service callers."""
    return RunAgent(ports, agent, ctx, on_stage=on_stage, pick=pick, project=project).run()
