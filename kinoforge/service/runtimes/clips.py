from pathlib import Path
from typing import Any, Callable, Dict, Optional

from kinoforge.segments.clips.moments.ai_engine import AiMomentEngine
from kinoforge.segments.clips.moments.offline_engine import OfflineMomentEngine
from kinoforge.segments.clips.project import write_project
from kinoforge.segments.clips.runner import ClipsRunner
from kinoforge.segments.clips.transcription import Transcriber
from kinoforge.contract import Context, Job, JobKind
from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.observ import bind, build_logger, logged, reset
from kinoforge.service.meter import EventMeter
from kinoforge.schemas import ClipsExecutionRequest
from kinoforge.service.settings import ServiceSettings
from kinoforge.service.executions import ExecutionStore


def _duration_checker(config: Dict[str, Any]):
    seconds = float((config.get("limits") or {}).get("source_max_seconds") or 0)

    def check(duration: float) -> Optional[str]:
        if seconds and duration > seconds:
            return f"Video is {duration / 60:.0f} minutes, over {seconds / 60:.0f} minute limit."
        return None

    return check


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
        source = Path(str(request.input.get("video_path") or request.input.get("audio_path") or ""))
        project_id = request.project_id
        state = request.state
        store = ExecutionStore(
            Path(request.workspace),
            project_id,
            source,
            record=state.record,
            transcript=state.transcript,
            has_clips=state.has_clips,
        )
        logger = build_logger(
            job_id=project_id,
            segment=JobKind.CLIPS.value,
            idempotency_key=request.idempotency_key,
        )
        meter = EventMeter()
        definitions = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0",
        )
        infrelay = self._settings.infrelay(request.owner)
        transcription = dict(request.config["transcription"])
        transcriber = Transcriber(
            transcription=transcription,
            cache_dir=self._settings.cache_dir,
            transcribe_bytes=infrelay.transcribe,
            meter=meter,
        )

        def moment_provider(config: Dict[str, Any], ctx: Context) -> object:
            route = config.get("moment_route") or {}
            if config.get("anti_hallucination") or not route:
                return OfflineMomentEngine()
            provider = str(route["provider"])
            model = str(route["model"])
            renderer = DefinitionRenderer(ctx.definitions)
            return AiMomentEngine(
                provider,
                model,
                min_clip_length=float(config["min_length"]),
                max_clip_length=float(config["max_length"]),
                tuning=dict(config["scoring"]),
                context_window=int(config.get("context_window") or 1_000_000),
                complete=lambda prompt, **params: infrelay.complete(
                    provider,
                    model,
                    prompt,
                    max_tokens=int(params["max_tokens"]),
                    temperature=float(params["temperature"]),
                ),
                render=renderer,
            )

        runner = ClipsRunner(
            logger=logger,
            report_stage=lambda stage, status: logger.debug(
                f"stage {stage.value}:{status.value}", stage=stage.value, status=status.value
            ),
            check_source_duration=_duration_checker(
                {**request.config, **request.options.model_dump()}
            ),
            transcribe_video=transcriber.transcribe_video,
            save_project=write_project,
            moment_provider=moment_provider,
            is_cancelled=is_cancelled,
        )
        context = Context(
            store=store,
            owner=request.owner,
            config=dict(request.config),
            definitions=definitions,
            infrelay_url=self._settings.infrelay_url,
            meter=meter,
        )
        token = bind(logger)  # ambient logger: any method can log the "why" via observ.current()
        try:
            result = runner.run(
                Job(
                    kind=JobKind.CLIPS,
                    job_id=project_id,
                    input=dict(request.input),
                    options=request.options.model_dump(),
                ),
                context,
            )
        finally:
            reset(token)
        result.data["execution"] = {
            "id": request.job_id,
            "project_id": project_id,
            "segment": JobKind.CLIPS.value,
            "definition_bundle": definitions.version,
        }
        return {
            "result": {
                "status": result.status,
                "artifacts": [
                    {
                        "path": str(artifact.path),
                        "media": artifact.media,
                        "meta": artifact.meta,
                    }
                    for artifact in result.artifacts
                ],
                "stages": result.stages,
                "error": result.error,
                "data": result.data,
            },
            "state": {
                "record": store.record,
                "transcript": store.transcript,
            },
            "meter_events": meter.events,
            "logs": logger.entries,
        }


