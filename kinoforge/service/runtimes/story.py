"""Stateless story stages over the REST seam.

Wires the injected story writer/operations to Infrelay text and a resolved DefinitionBundle:
the caller freezes the screenwriter agent, prompts and fragments and passes the text route;
this runtime runs the stages and returns the StoryPlan, meter events and logs. It owns no
durable state, the caller persists the story on its project."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

from kinoforge.contract import ModelRef
from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.observ import KinoLogger, bind, build_logger, logged, reset
from kinoforge.segments.story.agent import Screenwriter, StoryBrief
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
    """One resolved story run: the wired ports plus its per-request logger, meter, and route."""

    ports: StoryPorts
    logger: KinoLogger
    meter: EventMeter
    route: ModelRef

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
        ports = InfrelayTextPorts(
            self._settings.infrelay(request.owner),
            DefinitionRenderer(bundle),
            budgets=dict(request.config.get("budgets") or {}),
            label="story",
            meter=meter,
            contract_key=CONTRACT_KEY,
            language_key=LANGUAGE_KEY,
        ).to_ports()
        logger = build_logger(
            job_id=request.project_id, segment="story",
            idempotency_key=request.idempotency_key, level=request.log_level,
        )
        return StorySession(
            ports=ports, logger=logger, meter=meter, route=ModelRef.from_mapping(request.pick)
        )

    @logged
    def write(
        self,
        request: StoryWriteRequest,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        session = self._session(request)
        ctx = self._writer_context(session.ports, request)
        token = bind(session.logger)
        try:
            result = RunAgent(
                session.ports,
                request.agent,
                ctx,
                on_stage=partial(self._log_stage, session.logger),
                route=session.route,
                is_cancelled=is_cancelled,
            ).run()
        finally:
            reset(token)
        return session.response(result)

    @staticmethod
    def _log_stage(logger: KinoLogger, index: int, total: int, role: str) -> None:
        logger.info(f"Story stage {index + 1}/{total}: {role}", stage=role)

    @staticmethod
    def _writer_context(ports: StoryPorts, request: StoryWriteRequest) -> Dict[str, Any]:
        fields = request.context.model_dump()
        ctx = Screenwriter(ports).build_context(StoryBrief.from_mapping(fields))
        # Caller-set flags the writer reads (e.g. require_motion for the Video-Mode quality
        # check) ride through the request's open context.
        if "require_motion" in fields:
            ctx["require_motion"] = bool(fields["require_motion"])
        return ctx

    @logged
    def operate(self, request: StoryOperationRequest) -> Dict[str, Any]:
        session = self._session(request)
        token = bind(session.logger)
        try:
            result = self._dispatch(StoryOperations(session.ports), request, session.route)
        finally:
            reset(token)
        return session.response(result)

    @staticmethod
    def _dispatch(
        operations: StoryOperations, request: StoryOperationRequest, route: ModelRef
    ) -> Dict[str, Any]:
        payload = request.payload
        budgets = request.config.get("budgets") or {}
        match request.operation:
            case StoryOperation.REWRITE_SCENE:
                return operations.rewrite_scene(
                    dict(payload.get("record") or {}),
                    int(payload.get("index") or 0),
                    request.agent,
                    str(payload.get("instructions") or ""),
                    route,
                    int(budgets.get("rewrite") or 0),
                )
            case StoryOperation.REWRITE_CHARACTERS:
                return operations.rewrite_characters(
                    dict(payload.get("record") or {}),
                    request.agent,
                    route,
                    int(budgets.get("rewrite") or 0),
                )
            case StoryOperation.TRANSLATE:
                return operations.translate(
                    dict(payload.get("story") or {}),
                    str(payload.get("language") or ""),
                    route,
                    int(budgets.get("translation") or 0),
                )
            case StoryOperation.DIRECT_SHOTS:
                return operations.direct_shots(
                    dict(payload.get("record") or {}),
                    route,
                    int(budgets.get("rewrite") or 0),
                )
            case _:
                return operations.suggest_field(
                    str(payload.get("kind") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("description") or ""),
                    str(payload.get("idea") or ""),
                    route,
                    str(request.config.get("fallback_model") or ""),
                )
