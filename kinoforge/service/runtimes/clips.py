from pathlib import Path
from typing import Any, Callable, Dict, Optional

from kinoforge.contract import Context, Job, JobKind, Result
from kinoforge.definitions import DefinitionBundle
from kinoforge.observ import KinoLogger, bind, build_logger, logged, reset
from kinoforge.schemas import ClipsExecutionRequest
from kinoforge.segments.clips.project import write_project
from kinoforge.segments.clips.runner import ClipsRunner
from kinoforge.service.executions import ExecutionStore
from kinoforge.service.meter import EventMeter
from kinoforge.service.runtimes.moment_engines import MomentEngines
from kinoforge.service.runtimes.transcription import gateway_transcriber
from kinoforge.service.settings import ServiceSettings


class SourceDurationLimit:
    """Rejects a source longer than config.limits.source_max_seconds (0 = no limit)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._seconds = float((config.get("limits") or {}).get("source_max_seconds") or 0)

    def __call__(self, duration: float) -> Optional[str]:
        if self._seconds and duration > self._seconds:
            return (f"Video is {duration / 60:.0f} minutes, "
                    f"over {self._seconds / 60:.0f} minute limit.")
        return None


class ClipsRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    @classmethod
    def from_env(cls) -> "ClipsRuntime":
        return cls(ServiceSettings.from_env())

    @logged
    def execute(
        self,
        request: ClipsExecutionRequest,
        *,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        # The runner works from {**config, **options}; see README "Clips request".
        config = {**request.config, **request.options.model_dump()}
        logger = build_logger(
            job_id=request.project_id, segment=JobKind.CLIPS.value,
            idempotency_key=request.idempotency_key, level=request.log_level,
        )
        meter = EventMeter()
        definitions = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True), engine_version="0.1.0"
        )
        store = self._store(request)
        self._log_request(logger, request, config, definitions)
        runner = ClipsRunner(
            logger=logger,
            report_stage=lambda stage, status: logger.debug(
                f"stage {stage.value}:{status.value}", stage=stage.value, status=status.value
            ),
            check_source_duration=SourceDurationLimit(config),
            transcribe_video=gateway_transcriber(
                self._settings, request.owner, request.config["transcription"], meter
            ).transcribe_video,
            save_project=write_project,
            moment_provider=MomentEngines(self._settings.infrelay(request.owner)),
            is_cancelled=is_cancelled,
        )
        context = Context(
            store=store, owner=request.owner, config=dict(request.config),
            definitions=definitions, infrelay_url=self._settings.infrelay_url, meter=meter,
        )
        job = Job(kind=JobKind.CLIPS, job_id=request.project_id, input=dict(request.input),
                  options=request.options.model_dump())
        token = bind(logger)
        try:
            result = runner.run(job, context)
        finally:
            reset(token)
        return self._response(request, result, store, definitions, meter, logger)

    @staticmethod
    def _source(request: ClipsExecutionRequest) -> Path:
        return Path(str(request.input.get("video_path") or request.input.get("audio_path") or ""))

    @classmethod
    def _store(cls, request: ClipsExecutionRequest) -> ExecutionStore:
        state = request.state
        return ExecutionStore(
            Path(request.workspace), request.project_id, cls._source(request),
            record=state.record, transcript=state.transcript, has_clips=state.has_clips,
        )

    def _log_request(
        self, logger: KinoLogger, request: ClipsExecutionRequest, config: Dict[str, Any],
        definitions: DefinitionBundle,
    ) -> None:
        source = self._source(request)
        logger.debug(
            f"clips execute: project={request.project_id} source={source.name}",
            job_id=request.job_id, workspace=request.workspace, source=str(source),
            owner=request.owner or None, infrelay_url=self._settings.infrelay_url,
            definitions=definitions.version, config_keys=sorted(config),
        )
        logger.debug(
            "clips moment settings",
            moment_finder=config.get("moment_finder") or "auto",
            moment_route=config.get("moment_route") or {},
            context_window=config.get("context_window"),
            clip_count=config.get("clip_count"), min_length=config.get("min_length"),
            max_length=config.get("max_length"),
            whisper=(config.get("transcription") or {}).get("model"),
        )

    @staticmethod
    def _response(
        request: ClipsExecutionRequest, result: Result, store: ExecutionStore,
        definitions: DefinitionBundle, meter: EventMeter, logger: KinoLogger,
    ) -> Dict[str, Any]:
        result.data["execution"] = {
            "id": request.job_id,
            "project_id": request.project_id,
            "segment": JobKind.CLIPS.value,
            "definition_bundle": definitions.version,
        }
        return {
            "result": {
                "status": result.status,
                "artifacts": [
                    {"path": str(a.path), "media": a.media, "meta": a.meta}
                    for a in result.artifacts
                ],
                "stages": result.stages,
                "error": result.error,
                "data": result.data,
            },
            "state": {"record": store.record, "transcript": store.transcript},
            "meter_events": meter.events,
            "logs": logger.entries,
        }
