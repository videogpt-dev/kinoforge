from __future__ import annotations

from kinoforge.segments.clips.moments.finders import MomentFinder
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg
from kinoforge.segments.clips.run import ClipRun
from kinoforge.segments.clips.stages.moments import Moments
from kinoforge.segments.clips.stages.render import Render
from kinoforge.segments.clips.stages.transcribe import Transcribe, TranscribeVideo


class ClipsPipeline:
    """One clips execution: two guards, then transcribe, moments, render."""

    def __init__(self, transcribe_video: TranscribeVideo, moment_finder: MomentFinder) -> None:
        self._stages = (Transcribe(transcribe_video), Moments(moment_finder), Render())

    def run(self, run: ClipRun) -> None:
        try:
            if run.stopped():
                return
            run.logger.info(f"Starting: {run.media_path.name}")
            if self._already_processed(run) or not self._duration_ok(run):
                return
            for stage in self._stages:
                if not stage(run):
                    return
        except Exception as exc:
            run.logger.error(f"Unexpected error: {exc}")
            run.fail(f"Unexpected error: {exc!s}")

    @staticmethod
    def _already_processed(run: ClipRun) -> bool:
        config = run.config
        skip = bool(
            config.get("skip_already_processed")
            and not config.get("is_regenerate")
            and not config.get("force")
            and run.slug
            and run.has_clips
        )
        if skip:
            run.logger.info("Skipping: already has clips (skip_already_processed)")
            run.status = "skipped"
        return skip

    @staticmethod
    def _duration_ok(run: ClipRun) -> bool:
        limit = float((run.config.get("limits") or {}).get("source_max_seconds") or 0)
        if not limit:
            return True
        duration = float(Ffmpeg.probe(run.media_path).get("duration") or 0)
        if duration <= limit:
            return True
        message = f"Video is {duration / 60:.0f} minutes, over {limit / 60:.0f} minute limit."
        run.logger.warning(message)
        run.fail(message)
        return False
