"""Stateful runtime for multi-stage story agents."""

import json
from collections.abc import Callable

from kinoforge.contract import AITask, MeterAction, ModelRef
from kinoforge.segments.story.agent import AudienceMode, clean_text
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.quality import StoryQualityGate

StageCallback = Callable[[int, int, str], None]
CancellationChecker = Callable[[], bool]

TEXT_TEMPERATURE = 0.85
JSON_TEMPERATURE = 0.7
JSON_ATTEMPTS = 2
JSON_REMINDER = "Your previous reply was not valid JSON. Reply with the JSON object only."


class StoryShape:
    """Coerces a model's story JSON into the shape the pipeline expects."""

    FALSE_WORDS = frozenset({"", "false", "no", "n", "0", "off", "none", "null"})

    @classmethod
    def flag(cls, value: object) -> bool:
        """A model may write a boolean as text ("false"); read that as the word means."""
        if isinstance(value, str):
            return value.strip().lower() not in cls.FALSE_WORDS
        return bool(value)

    @classmethod
    def normalize(cls, raw: dict) -> dict:
        characters = [
            {
                "name": clean_text(character.get("name")),
                "description": clean_text(character.get("description")),
            }
            for character in (raw.get("characters") or [])
            if isinstance(character, dict)
            and (character.get("name") or character.get("description"))
        ]
        scenes = [
            {
                "prompt": clean_text(scene.get("prompt")),
                "narration": clean_text(scene.get("narration")),
                "motion": cls.flag(scene.get("motion")),
            }
            for scene in (raw.get("scenes") or [])
            if isinstance(scene, dict) and clean_text(scene.get("prompt"))
        ]
        return {
            "logline": clean_text(raw.get("logline")),
            "style": clean_text(raw.get("style")),
            "characters": characters,
            "scenes": scenes,
        }


class StageUtils:
    @staticmethod
    def temperature(stage: dict, is_json: bool) -> float:
        """The stage's own temperature, else a cooler default for strict JSON output."""
        try:
            return float(stage["temperature"])
        except (KeyError, TypeError, ValueError):
            return JSON_TEMPERATURE if is_json else TEXT_TEMPERATURE

    @staticmethod
    def tokens(usage: dict, key: str) -> int:
        try:
            return int(usage.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def error(role: str, route: ModelRef, error: Exception) -> str:
        which = ""
        if route.model:
            where = f" on {route.provider}" if route.provider else ""
            which = f" (model '{route.model}'{where})"
        return (f"Story '{role}' stage{which} failed: {error}. "
                "Change this stage's model in the agent editor.")


class RunAgent:
    """Run one agent definition against one story context.

    Instance owns mutable state accumulated across stages: intermediate outputs,
    generated story, token totals and models used. Call `run()` for result shape
    expected by queue worker.
    """

    def __init__(
        self,
        ports: StoryPorts,
        agent: dict,
        ctx: dict,
        on_stage: StageCallback | None = None,
        route: ModelRef | None = None,
        is_cancelled: CancellationChecker | None = None,
    ) -> None:
        self.ports = ports
        self.agent = agent
        self.ctx = ctx
        self.on_stage = on_stage
        self.route = route or ModelRef()
        self.is_cancelled = is_cancelled or (lambda: False)
        self._reset()

    def _reset(self) -> None:
        self.stages: list[dict] = [
            stage for stage in (self.agent.get("stages") or []) if isinstance(stage, dict)
        ]
        self.outputs: list[tuple[str, str]] = []
        self.story: dict | None = None
        self.total_in = 0
        self.total_out = 0
        self.model_used = ""
        self.models_used: list[str] = []
        self.requested_models: list[str] = []

    def _usage_summary(self) -> dict:
        return {
            "model": self.model_used,
            "models": self.models_used,
            "requested_models": self.requested_models,
            "input_tokens": self.total_in,
            "output_tokens": self.total_out,
        }

    def _build_prompt(self, stage: dict, idea: str, is_json: bool) -> str:
        parts = [idea]
        parts += [f"{role.upper()}:\n{text}" for role, text in self.outputs]

        instructions = (stage.get("instructions") or "").strip()
        if instructions:
            parts.append(instructions)
        if is_json:
            parts.append(self.ctx["language_rule"])
            parts.append(self.ports.contract(self.ctx["scene_count"]))
        return "\n\n".join(part for part in parts if part)

    def _record_usage(self, usage: dict) -> None:
        self.total_in += StageUtils.tokens(usage, "input_tokens")
        self.total_out += StageUtils.tokens(usage, "output_tokens")
        self.model_used = str(usage.get("model") or self.model_used)
        if self.model_used and self.model_used not in self.models_used:
            self.models_used.append(self.model_used)
        requested = usage.get("requested_model") or self.model_used
        if requested and requested not in self.requested_models:
            self.requested_models.append(requested)

    def _failure(self, error: str, **extra: object) -> dict:
        return {"ok": False, "error": error, **extra, "usage": self._usage_summary()}

    def _run_stage(self, stage: dict, role: str, idea: str) -> dict | None:
        """None on success; else the failure result for run() to return. A JSON stage whose
        reply does not parse is asked once more; the agent fee is charged once per stage."""
        is_json = stage.get("format") == "story_json"
        # A stage may pin its own provider/model over the project's route.
        route = self.route.with_overrides(
            str(stage.get("provider") or ""), str(stage.get("model") or "")
        )
        prompt = self._build_prompt(stage, idea, is_json)
        temperature = StageUtils.temperature(stage, is_json)
        attempts = JSON_ATTEMPTS if is_json else 1
        for attempt in range(attempts):
            try:
                text, usage = self.ports.complete(
                    AITask.STORY_WRITE,
                    # One persona per agent, applied to every stage; global default last.
                    self.agent.get("persona") or self.ports.prompt("story_system"),
                    prompt,
                    route=route,
                    temperature=temperature,
                    max_tokens=self.ports.story_budget(stage.get("max_output_tokens")),
                )
            except Exception as error:
                return self._failure(StageUtils.error(role, route, error))
            self._record_usage(usage)
            if attempt == 0:
                self.ports.meter(MeterAction.AGENT_RUN, 1)
            failure = self._take_story(text, usage) if is_json else None
            if failure is None:
                self.outputs.append((role, text))
                return None
            if usage.get("finish_reason") == "length" or attempt == attempts - 1:
                return failure
            prompt = f"{prompt}\n\n{JSON_REMINDER}"
            temperature = min(temperature, JSON_TEMPERATURE)
        return None

    def _take_story(self, text: str, usage: dict) -> dict | None:
        """Parse first: a model can hit the token budget yet still emit valid JSON, so trust
        the payload and only use finish_reason to word the error when it does not parse."""
        try:
            self.story = StoryShape.normalize(self.ports.parse_json(text))
            return None
        except (ValueError, json.JSONDecodeError):
            if usage.get("finish_reason") == "length":
                spent = StageUtils.tokens(usage, "output_tokens")
                return self._failure(
                    f"the story was cut off at {spent} tokens. Ask for fewer scenes, or raise "
                    "the stage's token budget. A different model will not help."
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

    def _run_stages(self, idea: str) -> dict | None:
        for index, stage in enumerate(self.stages):
            if self.is_cancelled():
                return self._failure("story execution cancelled", cancelled=True)
            role = (stage.get("role") or f"stage {index + 1}").strip()
            if self.on_stage:
                self.on_stage(index, len(self.stages), role)
            failure = self._run_stage(stage, role, idea)
            if failure:
                return failure
            if self.is_cancelled():
                return self._failure("story execution cancelled", cancelled=True)
        return None

    def _finish(self) -> dict:
        """The final story (falling back to the last stage's output) through the quality gate."""
        if self.story is None:
            try:
                self.story = StoryShape.normalize(self.ports.parse_json(self.outputs[-1][1]))
            except (ValueError, json.JSONDecodeError):
                return self._failure("agent produced no valid story JSON")
        if not self.story["scenes"]:
            return self._failure("story had no usable scenes")
        trimmed = self._trim_to_requested(self.story)
        scenes = self.story["scenes"]
        if self.ctx.get("require_motion") and not any(s["motion"] for s in scenes):
            # Video Mode with no scene marked would buy no clips at all; animate every scene.
            for scene in scenes:
                scene["motion"] = True
        blocking, advisory = StoryQualityGate(self.story, self.ctx).evaluate()
        if trimmed:
            advisory.append(trimmed)
        if blocking:
            return self._failure(
                f"Story can't be filmed: {'; '.join(blocking)}. "
                "Retry, or change the story model in Settings."
            )
        return {"ok": True, "story": self.story, "warnings": advisory,
                "usage": self._usage_summary()}

    def _trim_to_requested(self, story: dict) -> str:
        """Keep only the requested number of scenes: every extra scene would be paid for
        later as an image or clip. Returns the warning to show, or "" when nothing changed."""
        expected = int(self.ctx.get("scene_count") or 0)
        written = len(story["scenes"])
        if not expected or written <= expected:
            return ""
        story["scenes"] = story["scenes"][:expected]
        return f"the writer returned {written} scenes; kept the first {expected}"

    def run(self) -> dict:
        """Execute stages in order and return {ok, story, usage} or error."""
        self._reset()
        if not self.stages:
            return {"ok": False, "error": "agent has no stages"}
        if not self.ctx.get("title"):
            return {"ok": False, "error": "title is required"}
        return self._run_stages(self._idea()) or self._finish()

