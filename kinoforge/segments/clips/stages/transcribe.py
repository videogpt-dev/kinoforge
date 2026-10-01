from __future__ import annotations

from typing import Callable, Optional

from kinoforge.segments.clips.run import ClipRun, ClipStage, Segments, StageStatus

TranscribeVideo = Callable[..., Segments]


class Transcribe:
    """Fills run.transcript: host-supplied captions, else the transcript a previous run left
    in state, else a fresh transcription."""

    def __init__(self, transcribe_video: TranscribeVideo) -> None:
        self._transcribe_video = transcribe_video

    def __call__(self, run: ClipRun) -> bool:
        run.stage(ClipStage.TRANSCRIBE, StageStatus.RUNNING)
        run.logger.info("Transcribing")
        try:
            transcript = self._reused(run) or self._fresh(run)
        except Exception as exc:
            run.stage(ClipStage.TRANSCRIBE, StageStatus.FAILED)
            run.fail(f"Transcription failed: {exc!s}")
            run.logger.error(f"Transcription error: {exc}")
            return False
        run.data["transcript"] = transcript
        if transcript:
            run.transcript = transcript
        run.logger.success(f"Transcription complete ({len(transcript)} segments)")
        return not run.stopped(ClipStage.TRANSCRIBE)

    @staticmethod
    def _reused(run: ClipRun) -> Optional[Segments]:
        if run.config.get("force"):
            return None
        pretranscript = run.config.get("pretranscript")
        if pretranscript:
            run.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
            run.logger.success(
                f"Using YouTube captions ({len(pretranscript)} segments), skipped transcription"
            )
            return pretranscript
        if run.transcript:
            run.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
            run.logger.success(f"Reusing saved transcript ({len(run.transcript)} segments)")
            return run.transcript
        return None

    def _fresh(self, run: ClipRun) -> Segments:
        transcript = self._transcribe_video(
            run.media_path,
            output_dir=run.workdir,
            model_size=run.config.get("whisper_model") or None,
            language=run.config.get("language") or None,
        )
        run.stage(ClipStage.TRANSCRIBE, StageStatus.DONE)
        return transcript
