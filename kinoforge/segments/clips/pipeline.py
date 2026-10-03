from __future__ import annotations

from kinoforge.segments.clips.moments.finders import MomentFinder
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg
from kinoforge.segments.clips.stages.moments import Moments
from kinoforge.segments.clips.stages.render import Render
from kinoforge.segments.clips.stages.transcribe import Transcribe, TranscribeVideo
from kinoforge.segments.clips.worker import ClipWorker
from kinoforge.segments.common import Pipeline


class ClipsPipeline(Pipeline[ClipWorker]):
    """One clips execution: two guards, then transcribe, moments, render."""

    def __init__(self, transcribe_video: TranscribeVideo, moment_finder: MomentFinder) -> None:
        super().__init__(
            Transcribe(transcribe_video),
            Moments(moment_finder),
            Render(),
        )

    def ready(self, worker: ClipWorker) -> bool:
        worker.logger.info(f"Starting: {worker.media_path.name}")
        return not self._already_processed(worker) and self._duration_ok(worker)

    @staticmethod
    def _already_processed(worker: ClipWorker) -> bool:
        config = worker.config
        skip = bool(
            config.get("skip_already_processed")
            and not config.get("is_regenerate")
            and not config.get("force")
            and worker.slug
            and worker.has_clips
        )
        if skip:
            worker.logger.info("Skipping: already has clips (skip_already_processed)")
            worker.status = "skipped"
        return skip

    @staticmethod
    def _duration_ok(worker: ClipWorker) -> bool:
        limit = float((worker.config.get("limits") or {}).get("source_max_seconds") or 0)
        if not limit:
            return True
        duration = float(Ffmpeg.probe(worker.media_path).get("duration") or 0)
        if duration <= limit:
            return True
        message = f"Video is {duration / 60:.0f} minutes, over {limit / 60:.0f} minute limit."
        worker.logger.warning(message)
        worker.fail(message)
        return False
