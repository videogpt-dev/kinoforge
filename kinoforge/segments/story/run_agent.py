"""Stateful runtime for multi-stage story agents."""

import json
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple

from kinoforge.contract import MeterAction, ModelRef
from kinoforge.segments.story.agent import AudienceMode, clean_text as _s
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.quality import StoryQualityGate

StageCallback = Callable[[int, int, str], None]
CancellationChecker = Callable[[], bool]


def _not_cancelled() -> bool:
    return False


def _normalize(raw: Dict) -> Dict:
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
        route: Optional[ModelRef] = None,
        is_cancelled: Optional[CancellationChecker] = None,
    ) -> None:
        self.ports = ports
        self.agent = agent
        self.ctx = ctx
        self.on_stage = on_stage
        self.route = route or ModelRef()
        self.is_cancelled = is_cancelled or _not_cancelled
        self._reset()

    def _reset(self) -> None:
        self.stages: List[Dict] = [
            stage for stage in (self.agent.get("stages") or []) if isinstance(stage, dict)
        ]
        self.outputs: List[Tuple[str, str]] = []
        self.story: Optional[Dict] = None
        self.total_in = 0
        self.total_out = 0
        self.model_used = ""
        self.models_used: List[str] = []
        self.requested_models: List[str] = []

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

    def _failure(self, error: str, **extra: object) -> Dict:
        return {"ok": False, "error": error, **extra, "usage": self._usage_summary()}

    def _run_stage(self, stage: Dict, role: str, idea: str) -> Optional[Dict]:
        """None on success; else the failure result for run() to return."""
        is_json = stage.get("format") == "story_json"
        # A stage may pin its own provider/model over the project's route.
        route = self.route.with_overrides(
            str(stage.get("provider") or ""), str(stage.get("model") or "")
        )
        try:
            text, usage = self.ports.complete(
                "story",
                # One persona per agent, applied to every stage; global default last.
                self.agent.get("persona") or self.ports.prompt("story_system"),
                self._build_prompt(stage, idea, is_json),
                route=route,
                temperature=_stage_temperature(stage),
                max_tokens=self.ports.story_budget(stage.get("max_output_tokens")),
            )
        except Exception as error:
            return {"ok": False, "error": _stage_error(role, route, error)}
        self._record_usage(usage)
        self.ports.meter(MeterAction.AGENT_RUN, 1)
        if is_json:
            failure = self._take_story(text, usage)
            if failure:
                return failure
        self.outputs.append((role, text))
        return None

    def _take_story(self, text: str, usage: Dict) -> Optional[Dict]:
        """Parse first: a model can hit the token budget yet still emit valid JSON, so trust
        the payload and only use finish_reason to word the error when it does not parse."""
        try:
            self.story = _normalize(self.ports.parse_json(text))
            return None
        except (ValueError, json.JSONDecodeError):
            if usage.get("finish_reason") == "length":
                return self._failure(
                    f"the story was cut off at {usage['output_tokens']} tokens. Ask for fewer "
                    "scenes, or raise the stage's token budget. A different model will not help."
                )
            return self._failure(
                "the model did not return valid JSON. Retry, or pick another model in Settings."
            )

    def _idea(self) -> str:
        ctx = self.ctx
        return self.ports.prompt(
            "story_idea",
            title=ctx["title"], description=ctx["description"], scene_count=ctx["scene_count"],
            format=ctx["output_format"], genre=ctx.get("genre", ""),
            premise=ctx.get("premise", ""), series_name=ctx.get("series_name", ""),
            series_cast=ctx.get("series_cast", "[]"),
            series_history=ctx.get("series_history", "[]"),
            series_position=ctx.get("series_position", 0),
            audience=ctx.get("audience", AudienceMode.GENERAL.value),
        )

    def _run_stages(self, idea: str) -> Optional[Dict]:
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
                return self._failure("story execution cancelled", cancelled=True)
        return None

    def _finish(self) -> Dict:
        """The final story (falling back to the last stage's output) through the quality gate."""
        if self.story is None:
            try:
                self.story = _normalize(self.ports.parse_json(self.outputs[-1][1]))
            except (ValueError, json.JSONDecodeError):
                return self._failure("agent produced no valid story JSON")
        if not self.story["scenes"]:
            return self._failure("story had no usable scenes")
        blocking, advisory = StoryQualityGate(self.story, self.ctx).evaluate()
        if blocking:
            return self._failure(
                f"Story can't be filmed: {'; '.join(blocking)}. "
                "Retry, or change the story model in Settings."
            )
        return {"ok": True, "story": self.story, "warnings": advisory,
                "usage": self._usage_summary()}

    def run(self) -> Dict:
        """Execute stages in order and return {ok, story, usage} or error."""
        self._reset()
        if not self.stages:
            return {"ok": False, "error": "agent has no stages"}
        if not self.ctx.get("title"):
            return {"ok": False, "error": "title is required"}
        return self._run_stages(self._idea()) or self._finish()


def _stage_error(role: str, route: ModelRef, error: Exception) -> str:
    which = ""
    if route.model:
        where = f" on {route.provider}" if route.provider else ""
        which = f" (model '{route.model}'{where})"
    return (f"Story '{role}' stage{which} failed: {error}. "
            "Change this stage's model in the agent editor.")
