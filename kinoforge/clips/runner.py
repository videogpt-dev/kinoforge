from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from kinoforge.clips.moments import extract_auto_moments, score_and_rank_moments
from kinoforge.clips.render.clip_processor import extract_clips
from kinoforge.clips.render.formatter import format_clips_multi_platform, get_video_metadata
from kinoforge.contract import Artifact, Context, Job, JobKind, ProjectStore, Result


class ClipStage(StrEnum):
    TRANSCRIBE = "transcribe"
    MOMENTS = "moments"
    CLIPS = "clips"


class StageStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineLogger(Protocol):
    def info(self, text: str) -> None: ...
    def success(self, text: str) -> None: ...
    def warning(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...


StageReporter = Callable[[ClipStage, StageStatus], None]
DurationChecker = Callable[[float], Optional[str]]
TranscribeVideo = Callable[..., List[Dict[str, Any]]]
SaveProject = Callable[
    [
        Path,
        Dict[str, Any],
        str,
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        ProjectStore,
    ],
    None,
]
MomentProviderFactory = Callable[[Dict[str, Any], Context], object]
CancellationChecker = Callable[[], bool]


def _not_cancelled() -> bool:
    return False


class ClipsRunner:
    kind = JobKind.CLIPS

    def __init__(
        self,
        logger: PipelineLogger,
        report_stage: StageReporter,
        check_source_duration: DurationChecker,
        transcribe_video: TranscribeVideo,
        save_project: SaveProject,
        moment_provider: MomentProviderFactory,
        is_cancelled: Optional[CancellationChecker] = None,
    ) -> None:
        self._logger = logger
        self._report_stage = report_stage
        self._check_source_duration = check_source_duration
        self._transcribe_video = transcribe_video
        self._save_project = save_project
        self._moment_provider = moment_provider
        self._is_cancelled = is_cancelled or _not_cancelled

    def run(self, job: Job, ctx: Context) -> Result:
        if job.kind != self.kind:
            raise ValueError(f"ClipsRunner cannot run {job.kind}")
        raw_video_path = job.input.get("video_path")
        if not raw_video_path:
            raise ValueError("clips job requires input.video_path")

        config = {**ctx.config, **job.options}
        return self._process(Path(raw_video_path), job.job_id, config, ctx)

    def _stage(self, result: Result, stage: ClipStage, status: StageStatus) -> None:
        self._report_stage(stage, status)
        result.stages.append(f"{stage.value}:{status.value}")

    @staticmethod
    def _fail(result: Result, message: str) -> None:
        result.status = "failed"
        result.error = message
        result.data["errors"].append(message)

    def _stop_if_cancelled(
        self,
        result: Result,
        stage: Optional[ClipStage] = None,
    ) -> bool:
        if not self._is_cancelled():
            return False
        if stage is not None:
            self._stage(result, stage, StageStatus.CANCELLED)
        result.status = "cancelled"
        result.error = ""
        result.data["cancelled"] = True
        self._logger.warning("Execution cancelled")
        return True

    def _process(
        self,
        video_path: Path,
        job_id: str,
        config: Dict[str, Any],
        ctx: Context,
    ) -> Result:
        result = Result(
            data={
                "video_path": str(video_path),
                "clips": [],
                "moments": [],
                "transcript": None,
                "errors": [],
            }
        )
        video_out = ctx.store.workdir(job_id)
        slug = config.get("slug", "")
        provider = self._moment_provider(config, ctx)

        try:
            if self._stop_if_cancelled(result):
                return result
            self._logger.info(f"Starting: {video_path.name}")

            if (
                config.get("skip_already_processed")
                and not config.get("is_regenerate")
                and not config.get("force")
                and slug
                and ctx.store.list_clips(job_id)
            ):
                self._logger.info("Skipping: already has clips (skip_already_processed)")
                result.status = "skipped"
                return result

            duration_error = self._check_source_duration(
                get_video_metadata(video_path).get("duration") or 0
            )
            if duration_error:
                self._logger.warning(duration_error)
                self._fail(result, duration_error)
                return result

            self._stage(result, ClipStage.TRANSCRIBE, StageStatus.RUNNING)
            self._logger.info("Transcribing")
            try:
                restart = bool(config.get("force"))
                fresh = restart or not config.get("try_youtube_subs", True)
                pretranscript = None if restart else config.get("pretranscript")
                saved = ctx.store.load_transcript(job_id) if not fresh else None

                if pretranscript:
                    transcript = pretranscript
                    self._stage(result, ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
                    self._logger.success(
                        f"Using YouTube captions ({len(transcript)} segments), skipped transcription"
                    )
                elif saved:
                    transcript = saved
                    self._stage(result, ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
                    self._logger.success(
                        f"Reusing saved transcript ({len(transcript)} segments)"
                    )
                else:
                    transcript = self._transcribe_video(
                        video_path,
                        output_dir=video_out,
                        model_size=config.get("whisper_model") or None,
                        language=config.get("language") or None,
                    )
                    self._stage(result, ClipStage.TRANSCRIBE, StageStatus.DONE)
                result.data["transcript"] = transcript
                if transcript:
                    ctx.store.save_transcript(job_id, transcript)
                self._logger.success(f"Transcription complete ({len(transcript)} segments)")
                if self._stop_if_cancelled(result, ClipStage.TRANSCRIBE):
                    return result
            except Exception as exc:
                self._stage(result, ClipStage.TRANSCRIBE, StageStatus.FAILED)
                message = f"Transcription failed: {exc!s}"
                self._fail(result, message)
                self._logger.error(f"Transcription error: {exc}")
                return result

            self._stage(result, ClipStage.MOMENTS, StageStatus.RUNNING)
            self._logger.info("Finding moments")
            discovered = False
            try:
                moments: List[Dict[str, Any]] = []
                if provider is not None and hasattr(provider, "discover_moments"):
                    moments = (
                        provider.discover_moments(
                            transcript,
                            config["min_length"],
                            config["max_length"],
                            config["clip_count"],
                        )
                        or []
                    )
                    discovered = bool(moments)
                    if discovered:
                        self._logger.success(
                            f"Read the transcript and chose {len(moments)} moments"
                        )
                if not moments:
                    moments = extract_auto_moments(
                        video_path=video_path,
                        transcript=transcript,
                        min_length=config["min_length"],
                        max_length=config["max_length"],
                        target_clips=config["clip_count"],
                        verbose=config["verbose"],
                    )
                    self._logger.success(f"Moment extraction: {len(moments)} moments found")
                result.data["moments"] = moments
                if self._stop_if_cancelled(result, ClipStage.MOMENTS):
                    return result
            except Exception as exc:
                self._stage(result, ClipStage.MOMENTS, StageStatus.FAILED)
                message = f"Moment extraction failed: {exc!s}"
                self._fail(result, message)
                self._logger.error(f"Moment extraction error: {exc}")
                return result

            if not moments:
                self._stage(result, ClipStage.MOMENTS, StageStatus.DONE)
                self._logger.warning("No moments extracted from video")
                return result

            if discovered:
                moments = sorted(
                    moments,
                    key=lambda moment: float(moment.get("score") or 0),
                    reverse=True,
                )
            else:
                if provider and hasattr(provider, "filter_moments"):
                    try:
                        filtered = provider.filter_moments(moments, transcript)
                        if filtered:
                            moments = filtered
                            self._logger.success(
                                f"Filtered moments using {getattr(provider, 'name', 'provider')}"
                            )
                    except Exception as exc:
                        self._logger.warning(f"Provider filtering failed: {exc}")

                self._logger.info("Scoring moments")
                try:
                    if provider and hasattr(provider, "score_moments"):
                        moments = provider.score_moments(moments, transcript)
                        self._logger.success(
                            f"Scored moments with {getattr(provider, 'name', 'provider')}"
                        )
                    else:
                        moments = score_and_rank_moments(moments, transcript)
                        self._logger.success("Moments scored and ranked")
                except Exception as exc:
                    self._logger.warning(f"Moment scoring failed: {exc}")

            threshold = config.get("min_interest_score") or 0.0
            if threshold > 0 and moments:
                scores = [float(moment.get("score") or 0) for moment in moments]
                scale = 100.0 if max(scores) > 10 else 10.0
                ranked = [
                    moment
                    for moment in moments
                    if (float(moment.get("score") or 0) / scale) >= threshold
                ]
                if not ranked:
                    self._logger.warning(
                        f"No moments scored ≥ {threshold:.2f}; using top results instead."
                    )
                    ranked = moments
                else:
                    self._logger.info(
                        f"Kept {len(ranked)}/{len(moments)} moments scoring ≥ {threshold:.2f}."
                    )
            else:
                ranked = moments

            min_len = float(config.get("min_length") or 0)
            max_len = float(config.get("max_length") or 0)
            for moment in ranked:
                if (
                    max_len
                    and float(moment.get("end") or 0) - float(moment.get("start") or 0) > max_len
                ):
                    moment["end"] = float(moment["start"]) + max_len
                    moment["duration"] = max_len
            if min_len:
                usable = [
                    moment
                    for moment in ranked
                    if float(moment.get("end") or 0) - float(moment.get("start") or 0) >= min_len
                ]
                if usable:
                    ranked = usable
                else:
                    self._logger.warning(
                        f"Every moment is shorter than {min_len:.0f}s; keeping them anyway."
                    )

            used_moments = ranked[: config["clip_count"]]
            result.data["used_moments"] = used_moments
            created_ids: List[Optional[str]] = []
            try:
                self._save_project(
                    video_path,
                    config,
                    slug,
                    used_moments,
                    transcript,
                    ctx.store,
                )
                self._logger.success("Saved project.json (editor source of truth)")
                project = ctx.store.load_record(job_id)
                if project is not None:
                    created_ids = ctx.store.append_clips(job_id, project, used_moments)
                    self._logger.success(
                        f"Added {sum(1 for clip_id in created_ids if clip_id)} clips to the pool"
                    )
            except Exception as exc:
                self._logger.warning(f"Could not write project/clips: {exc}")
            self._stage(result, ClipStage.MOMENTS, StageStatus.DONE)

            if config.get("analyze_only"):
                self._stage(result, ClipStage.CLIPS, StageStatus.SKIPPED)
                self._logger.success("Analyze-only: skipping clip extraction (open in editor)")
                return result

            if self._stop_if_cancelled(result):
                return result

            self._stage(result, ClipStage.CLIPS, StageStatus.RUNNING)
            try:
                work = video_out / "_work"
                (work / "clips").mkdir(parents=True, exist_ok=True)
                formatted_dir = work / "formatted"
                formatted_dir.mkdir(parents=True, exist_ok=True)

                clip_paths = extract_clips(
                    video_path=video_path,
                    moments=used_moments,
                    output_dir=work / "clips",
                    quality=config["quality"],
                    max_workers=config["processing"]["max_workers"],
                    meter=ctx.meter,
                )
                if self._stop_if_cancelled(result, ClipStage.CLIPS):
                    return result

                formats = config["formats"] or ["9:16"]
                rendering = dict(config["rendering"])
                rendering["burn_subtitles"] = bool(
                    config.get("generate_captions", rendering["burn_subtitles"])
                )
                format_clips_multi_platform(
                    clip_paths=clip_paths,
                    moments=used_moments,
                    output_dir=formatted_dir,
                    formats=formats,
                    transcript=transcript,
                    rendering=rendering,
                    processing=config["processing"],
                    meter=ctx.meter,
                )
                if self._stop_if_cancelled(result, ClipStage.CLIPS):
                    return result

                transcript_segments = (ctx.store.load_record(job_id) or {}).get("transcript", [])
                placed = 0
                output_paths: List[str] = []
                for index, clip_id in enumerate(created_ids, 1):
                    if not clip_id:
                        continue
                    candidates = (
                        sorted(
                            formatted_dir.glob(
                                f"clip_{index:02d}_{formats[0].replace(':', 'x')}.mp4"
                            )
                        )
                        or sorted(formatted_dir.glob(f"clip_{index:02d}_*.mp4"))
                        or sorted((work / "clips").glob(f"clip_{index:02d}_raw.mp4"))
                    )
                    if not candidates:
                        continue
                    destination = ctx.store.clip_path(job_id, clip_id) / "clip.mp4"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(candidates[0]), str(destination))
                    ctx.store.mark_rendered(job_id, clip_id, "clip.mp4", transcript_segments)
                    output_paths.append(str(destination))
                    result.artifacts.append(
                        Artifact(
                            path=destination,
                            media="video/mp4",
                            meta={"clip_id": clip_id, "moment_index": index - 1},
                        )
                    )
                    placed += 1
                result.data["clips"] = output_paths
                self._logger.success(f"Auto-exported {placed} clips")
                self._stage(result, ClipStage.CLIPS, StageStatus.DONE)
            except Exception as exc:
                self._stage(result, ClipStage.CLIPS, StageStatus.FAILED)
                message = f"Clip extraction failed: {exc!s}"
                self._fail(result, message)
                self._logger.error(f"Clip extraction error: {exc}")
            finally:
                shutil.rmtree(video_out / "_work", ignore_errors=True)

            self._logger.success("Processing complete!")
            return result

        except Exception as exc:
            self._logger.error(f"Unexpected error: {exc}")
            self._fail(result, f"Unexpected error: {exc!s}")
            return result
