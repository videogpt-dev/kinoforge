from pathlib import Path
from typing import Any, Callable, Dict, Optional

from kinoforge.contract import JobKind
from kinoforge.definitions import DefinitionBundle
from kinoforge.observ import KinoLogger, bind, build_logger, logged, reset
from kinoforge.schemas import ClipsExecutionRequest
from kinoforge.segments.clips.pipeline import ClipsPipeline
from kinoforge.segments.clips.run import ClipRun
from kinoforge.service.meter import EventMeter
from kinoforge.service.runtimes.moment_engines import MomentEngines
from kinoforge.service.runtimes.transcription import gateway_transcriber
from kinoforge.service.settings import ServiceSettings


def _path(value: Any) -> Optional[Path]:
    return Path(value) if value else None


class ClipsRuntime:
    """HTTP request in, ClipRun through the pipeline, response dict out."""

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
        # Stages read {**config, **options}; see README "Clips request".
        config = {**request.config, **request.options.model_dump()}
        logger = build_logger(
            job_id=request.project_id, segment=JobKind.CLIPS.value,
            idempotency_key=request.idempotency_key, level=request.log_level,
        )
        meter = EventMeter()
        definitions = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True), engine_version="0.1.0"
        )
        workspace = Path(request.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        self._log_request(logger, request, config, definitions)
        run = ClipRun(
            job_id=request.project_id, config=config, workdir=workspace, logger=logger,
            video_path=_path(request.input.get("video_path")),
            audio_path=_path(request.input.get("audio_path")),
            meter=meter,
            record=request.state.record,
            transcript=list(request.state.transcript or []),
            has_clips=request.state.has_clips,
        )
        if is_cancelled is not None:
            run.is_cancelled = is_cancelled
        token = bind(logger)
        try:
            ClipsPipeline(
                transcribe_video=gateway_transcriber(
                    self._settings, request.owner, request.config["transcription"], meter
                ).transcribe_video,
                moment_engine=MomentEngines(self._settings.infrelay(request.owner))(
                    config, definitions
                ),
            ).run(run)
        finally:
            reset(token)
        return self._response(request, run, definitions, meter, logger)

    def _log_request(
        self, logger: KinoLogger,
        request: ClipsExecutionRequest,
        config: Dict[str, Any],
        definitions: DefinitionBundle,
    ) -> None:
        source = Path(str(request.input.get("video_path") or request.input.get("audio_path") or ""))
        logger.debug(
            f"clips execute: project={request.project_id} source={source.name}",
            job_id=request.job_id,
            workspace=request.workspace,
            source=str(source),
            owner=request.owner or None,
            infrelay_url=self._settings.infrelay_url,
            definitions=definitions.version,
            config_keys=sorted(config),
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
        request: ClipsExecutionRequest, run: ClipRun,
        definitions: DefinitionBundle, meter: EventMeter, logger: KinoLogger,
    ) -> Dict[str, Any]:
        run.data["execution"] = {
            "id": request.job_id,
            "project_id": request.project_id,
            "segment": JobKind.CLIPS.value,
            "definition_bundle": definitions.version,
        }
        return {
            "result": {
                "status": run.status,
                "artifacts": run.artifacts,
                "stages": run.stages,
                "error": run.error,
                "data": run.data,
            },
            "state": {"record": run.record, "transcript": run.transcript or None},
            "meter_events": meter.events,
            "logs": logger.entries,
        }
