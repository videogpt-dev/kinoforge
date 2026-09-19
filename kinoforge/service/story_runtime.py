"""Stateless write-story stage over the REST seam.

Wires the injected story writer to Infrelay text and a resolved DefinitionBundle: the
caller freezes the screenwriter agent, prompts and fragments, and passes the text route;
this runtime executes the stages and returns the StoryPlan, meter events and logs. It owns
no durable state, cloud core (or the self-host runtime) persists the story on the project.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple

from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.definitions.models import DefinitionKind
from kinoforge.service.events import EventLogger, EventMeter
from kinoforge.service.infrelay import InfrelayClient
from kinoforge.service.models import StoryOperation, StoryOperationRequest, StoryWriteRequest
from kinoforge.story.agent import Screenwriter
from kinoforge.story.operations import StoryOperations
from kinoforge.story.ports import StoryPorts
from kinoforge.story.run_agent import RunAgent

CONTRACT_KEY = "prompts.fragments.contract"
LANGUAGE_KEY = "prompts.fragments.language"


def _parse_json(text: str) -> Dict[str, Any]:
    """The JSON object out of a completion, whatever the model wrapped it in."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start : end + 1]
    return json.loads(s)


class StoryRuntime:
    def __init__(self, infrelay_url: str, infrelay_token: str) -> None:
        self._infrelay_url = infrelay_url
        self._infrelay_token = infrelay_token

    def _ports(
        self, request: StoryWriteRequest | StoryOperationRequest
    ) -> tuple[StoryPorts, EventLogger, EventMeter, Dict[str, Any]]:
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0")
        renderer = DefinitionRenderer(bundle)
        infrelay = InfrelayClient(self._infrelay_url, self._infrelay_token, request.owner)
        logger = EventLogger()
        meter = EventMeter()
        budgets = dict(request.config.get("budgets") or {})
        default_pick = dict(request.pick or {})

        def complete(
            purpose: str,
            system: str,
            user: str,
            *,
            pick: Dict[str, Any],
            project: str,
            temperature: float,
            max_tokens: int,
            bill: bool) -> Tuple[str, Dict[str, Any]]:
            over = {**default_pick, **(pick or {})}
            provider = str(over.get("provider") or "")
            model = str(over.get("model") or "")
            if not provider:
                raise RuntimeError("story text route is missing a provider")
            return infrelay.text(
                provider, model, system, user, temperature=temperature, max_tokens=max_tokens
            )

        def complete_json(
            purpose: str,
            system: str,
            user: str,
            *,
            pick: Dict[str, Any],
            temperature: float,
            max_tokens: int) -> Dict[str, Any]:
            text, _ = complete(
                purpose,
                system,
                user,
                pick=pick,
                project="",
                temperature=temperature,
                max_tokens=max_tokens,
                bill=True)
            return _parse_json(text)

        def prompt(key: str, **params: object) -> str:
            return renderer.render(DefinitionKind.AGENT, key, **params)

        def contract(scene_count: int) -> str:
            return renderer.render(DefinitionKind.FRAGMENT, CONTRACT_KEY, scene_count=scene_count)

        def language_rule(subject: str, language: str) -> str:
            return renderer.render(
                DefinitionKind.FRAGMENT, LANGUAGE_KEY, subject=subject, language=language
            )

        def story_budget(stage_override: object = None) -> int:
            limit = int(budgets.get("story") or 0)
            if stage_override in (None, ""):
                return limit
            try:
                requested = int(str(stage_override))
            except (TypeError, ValueError):
                return limit
            return min(requested, limit) if requested > 0 else limit

        ports = StoryPorts(
            complete=complete,
            complete_json=complete_json,
            parse_json=_parse_json,
            prompt=prompt,
            contract=contract,
            language_rule=language_rule,
            story_budget=story_budget,
            meter=meter)
        return ports, logger, meter, default_pick

    def write(
        self,
        request: StoryWriteRequest,
        is_cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        ports, logger, meter, default_pick = self._ports(request)

        context = request.context
        ctx = Screenwriter(ports).build_context(
            title=context.title,
            description=context.description,
            scene_count=context.scene_count,
            aspect_ratio=context.aspect_ratio,
            language=context.language,
            genre=context.genre,
            cast=context.cast,
            premise=context.premise,
            series_name=context.series_name,
            series_episodes=context.series_episodes,
            series_position=context.series_position,
            mature=context.mature)
        # Extra caller-set context flags the writer reads (e.g. require_motion for the
        # Video-Mode quality check) ride through the request's open context.
        extras = context.model_dump()
        if "require_motion" in extras:
            ctx["require_motion"] = bool(extras["require_motion"])

        def on_stage(index: int, total: int, role: str) -> None:
            logger.info(f"Story stage {index + 1}/{total}: {role}")

        result = RunAgent(
            ports,
            request.agent,
            ctx,
            on_stage=on_stage,
            pick=default_pick,
            project=request.project_id,
            is_cancelled=is_cancelled).run()
        return {"result": result, "meter_events": meter.events, "logs": logger.entries}

    def operate(self, request: StoryOperationRequest) -> Dict[str, Any]:
        ports, logger, meter, default_pick = self._ports(request)
        operations = StoryOperations(ports)
        payload = request.payload
        budgets = request.config.get("budgets") or {}
        if request.operation is StoryOperation.REWRITE_SCENE:
            result = operations.rewrite_scene(
                dict(payload.get("record") or {}),
                int(payload.get("index") or 0),
                request.agent,
                str(payload.get("instructions") or ""),
                default_pick,
                int(budgets.get("rewrite") or 0))
        elif request.operation is StoryOperation.REWRITE_CHARACTERS:
            result = operations.rewrite_characters(
                dict(payload.get("record") or {}),
                request.agent,
                default_pick,
                int(budgets.get("rewrite") or 0))
        elif request.operation is StoryOperation.TRANSLATE:
            result = operations.translate(
                dict(payload.get("story") or {}),
                str(payload.get("language") or ""),
                default_pick,
                int(budgets.get("translation") or 0))
        elif request.operation is StoryOperation.DIRECT_SHOTS:
            result = operations.direct_shots(
                dict(payload.get("record") or {}),
                default_pick,
                int(budgets.get("rewrite") or 0))
        else:
            result = operations.suggest_field(
                str(payload.get("kind") or ""),
                str(payload.get("title") or ""),
                str(payload.get("description") or ""),
                str(payload.get("idea") or ""),
                default_pick,
                str(request.config.get("fallback_model") or ""))
        return {"result": result, "meter_events": meter.events, "logs": logger.entries}


def story_runtime() -> StoryRuntime:
    return StoryRuntime(
        os.getenv("INFRELAY_URL") or "",
        os.getenv("INFRELAY_SERVICE_TOKEN") or "")
