"""Stateless story stages over the REST seam.

Wires the injected story writer/operations to Infrelay text and a resolved DefinitionBundle:
the caller freezes the screenwriter agent, prompts and fragments and passes the text route;
this runtime runs the stages and returns the StoryPlan, meter events and logs. It owns no
durable state, the caller persists the story on its project."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional

from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.observ import KinoLogger, bind, build_logger, logged, reset
from kinoforge.segments.story.agent import Screenwriter
from kinoforge.segments.story.operations import StoryOperations
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.run_agent import RunAgent
from kinoforge.service.meter import EventMeter
from kinoforge.schemas import StoryOperation, StoryOperationRequest, StoryWriteRequest
from kinoforge.service.settings import ServiceSettings
from kinoforge.service.runtimes.text_ports import InfrelayTextPorts

CONTRACT_KEY = "prompts.fragments.contract"
LANGUAGE_KEY = "prompts.fragments.language"


@dataclass(frozen=True)
class StorySession:
    """One resolved story run: the wired ports plus its per-request logger, meter, and pick."""

    ports: StoryPorts
    logger: KinoLogger
    meter: EventMeter
    pick: Dict[str, Any]

    def response(self, result: Any) -> Dict[str, Any]:
        return {"result": result, "meter_events": self.meter.events, "logs": self.logger.entries}


class StoryRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    @classmethod
    def from_env(cls) -> "StoryRuntime":
        return cls(ServiceSettings.from_env())

    def _session(self, request: StoryWriteRequest | StoryOperationRequest) -> StorySession:
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True), engine_version="0.1.0"
        )
        meter = EventMeter()
        pick = dict(request.pick or {})
        ports = InfrelayTextPorts(
            self._settings.infrelay(request.owner),
            DefinitionRenderer(bundle),
            default_pick=pick,
            budgets=dict(request.config.get("budgets") or {}),
            label="story",
            meter=meter,
            contract_key=CONTRACT_KEY,
            language_key=LANGUAGE_KEY,
        ).to_ports()
        logger = build_logger(
            job_id=request.project_id, segment="story", idempotency_key=request.idempotency_key
        )
        return StorySession(ports=ports, logger=logger, meter=meter, pick=pick)

    @logged
    def write(
        self,
        request: StoryWriteRequest,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        session = self._session(request)
        ctx = self._writer_context(session.ports, request)

        def on_stage(index: int, total: int, role: str) -> None:
            session.logger.info(f"Story stage {index + 1}/{total}: {role}", stage=role)

        token = bind(session.logger)
        try:
            result = RunAgent(
                session.ports,
                request.agent,
                ctx,
                on_stage=on_stage,
                pick=session.pick,
                project=request.project_id,
                is_cancelled=is_cancelled,
            ).run()
        finally:
            reset(token)
        return session.response(result)

    @staticmethod
    def _writer_context(ports: StoryPorts, request: StoryWriteRequest) -> Dict[str, Any]:
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
            mature=context.mature,
        )
        # Extra caller-set flags the writer reads (e.g. require_motion for the Video-Mode
        # quality check) ride through the request's open context.
        extras = context.model_dump()
        if "require_motion" in extras:
            ctx["require_motion"] = bool(extras["require_motion"])
        return ctx

    @logged
    def operate(self, request: StoryOperationRequest) -> Dict[str, Any]:
        session = self._session(request)
        operations = StoryOperations(session.ports)
        token = bind(session.logger)
        try:
            result = self._dispatch(
                operations,
                request,
                request.payload,
                request.config.get("budgets") or {},
                session.pick,
            )
        finally:
            reset(token)
        return session.response(result)

    @staticmethod
    def _dispatch(
        operations: StoryOperations,
        request: StoryOperationRequest,
        payload: Dict[str, Any],
        budgets: Dict[str, Any],
        default_pick: Dict[str, Any],
    ) -> Dict[str, Any]:
        match request.operation:
            case StoryOperation.REWRITE_SCENE:
                return operations.rewrite_scene(
                    dict(payload.get("record") or {}),
                    int(payload.get("index") or 0),
                    request.agent,
                    str(payload.get("instructions") or ""),
                    default_pick,
                    int(budgets.get("rewrite") or 0),
                )
            case StoryOperation.REWRITE_CHARACTERS:
                return operations.rewrite_characters(
                    dict(payload.get("record") or {}),
                    request.agent,
                    default_pick,
                    int(budgets.get("rewrite") or 0),
                )
            case StoryOperation.TRANSLATE:
                return operations.translate(
                    dict(payload.get("story") or {}),
                    str(payload.get("language") or ""),
                    default_pick,
                    int(budgets.get("translation") or 0),
                )
            case StoryOperation.DIRECT_SHOTS:
                return operations.direct_shots(
                    dict(payload.get("record") or {}),
                    default_pick,
                    int(budgets.get("rewrite") or 0),
                )
            case _:
                return operations.suggest_field(
                    str(payload.get("kind") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("description") or ""),
                    str(payload.get("idea") or ""),
                    default_pick,
                    str(request.config.get("fallback_model") or ""),
                )
