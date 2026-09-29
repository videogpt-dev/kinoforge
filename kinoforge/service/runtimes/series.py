"""Stateless series-planning stage over the REST seam.

Runs the resolved showrunner over a premise and cast and returns proposed episode ideas,
meter events and logs. Stateless: the caller persists the series and creates the episodes the
user approves. Series planning uses agent prompts only, so the contract/language fragments are
absent (InfrelayTextPorts raises if they are asked for)."""

from __future__ import annotations

from typing import Any, Dict

from kinoforge.contract import ModelRef
from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.observ import bind, build_logger, logged, reset
from kinoforge.segments.series.showrunner import Showrunner
from kinoforge.service.meter import EventMeter
from kinoforge.schemas import SeriesPlanRequest
from kinoforge.service.settings import ServiceSettings
from kinoforge.service.runtimes.text_ports import InfrelayTextPorts


class SeriesRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    @classmethod
    def from_env(cls) -> "SeriesRuntime":
        return cls(ServiceSettings.from_env())

    @logged
    def plan(self, request: SeriesPlanRequest) -> Dict[str, Any]:
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True), engine_version="0.1.0"
        )
        logger = build_logger(
            job_id=request.project_id, segment="series",
            idempotency_key=request.idempotency_key, level=request.log_level,
        )
        meter = EventMeter()
        ports = InfrelayTextPorts(
            self._settings.infrelay(request.owner),
            DefinitionRenderer(bundle),
            budgets=dict(request.config.get("budgets") or {}),
            label="series",
            meter=meter,
        ).to_ports()

        token = bind(logger)
        try:
            result = Showrunner(ports).plan(
                request.series.model_dump(), request.count, ModelRef.from_mapping(request.pick)
            )
        finally:
            reset(token)
        return {"result": result, "meter_events": meter.events, "logs": logger.entries}
