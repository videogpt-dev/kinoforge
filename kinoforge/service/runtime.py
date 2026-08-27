import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from kinoforge.clips.moments.ai_engine import AiMomentEngine
from kinoforge.clips.moments.offline_engine import OfflineMomentEngine
from kinoforge.clips.project import write_project
from kinoforge.clips.runner import ClipsRunner
from kinoforge.clips.transcription import Transcriber
from kinoforge.contract import Context, Job, JobKind
from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.service.events import EventLogger, EventMeter
from kinoforge.service.infrelay import InfrelayClient
from kinoforge.service.models import ClipsExecutionRequest
from kinoforge.service.store import ExecutionStore


def _duration_checker(config: Dict[str, Any]):
    seconds = float((config.get("limits") or {}).get("source_max_seconds") or 0)

    def check(duration: float) -> Optional[str]:
        if seconds and duration > seconds:
            return f"Video is {duration / 60:.0f} minutes, over {seconds / 60:.0f} minute limit."
        return None

    return check


class ClipsRuntime:
    def __init__(self, infrelay_url: str, infrelay_token: str, cache_dir: Path) -> None:
        self._infrelay_url = infrelay_url
        self._infrelay_token = infrelay_token
        self._cache_dir = cache_dir

    def execute(
        self,
        request: ClipsExecutionRequest,
        *,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        source = Path(str(request.input["video_path"]))
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
        logger = EventLogger()
        meter = EventMeter()
        definitions = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0",
        )
        infrelay = InfrelayClient(
            self._infrelay_url,
            self._infrelay_token,
            request.owner,
        )
        transcription = dict(request.config["transcription"])
        transcriber = Transcriber(
            transcription=transcription,
            cache_dir=self._cache_dir,
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
            report_stage=lambda _stage, _status: None,
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
            infrelay_url=self._infrelay_url,
            meter=meter,
        )
        result = runner.run(
            Job(
                kind=JobKind.CLIPS,
                job_id=project_id,
                input=dict(request.input),
                options=request.options.model_dump(),
            ),
            context,
        )
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


def runtime() -> ClipsRuntime:
    shared = Path(os.getenv("KINOFORGE_SHARED_ROOT") or "/app/output")
    cache = Path(os.getenv("KINOFORGE_CACHE_DIR") or shared / ".kinoforge-cache")
    return ClipsRuntime(
        os.getenv("INFRELAY_URL") or "",
        os.getenv("INFRELAY_SERVICE_TOKEN") or "",
        cache,
    )
