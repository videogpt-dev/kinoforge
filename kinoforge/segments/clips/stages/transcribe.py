from __future__ import annotations

from typing import Callable, Optional

from kinoforge.segments.clips.worker import ClipStage, ClipWorker, Segments
from kinoforge.segments.common import Stage, StageStatus

TranscribeVideo = Callable[..., Segments]


class Transcribe(Stage[ClipWorker]):
    """Fills worker.transcript: host-supplied captions, else the transcript a previous run left
    in state, else a fresh transcription."""

    def __init__(self, transcribe_video: TranscribeVideo) -> None:
        self._transcribe_video = transcribe_video

    def __call__(self, worker: ClipWorker) -> bool:
        worker.stage(ClipStage.TRANSCRIBE, StageStatus.RUNNING)
        worker.logger.info("Transcribing")
        try:
            transcript = self._rounded(self._reused(worker) or self._fresh(worker))
        except Exception as exc:
            worker.stage(ClipStage.TRANSCRIBE, StageStatus.FAILED)
            worker.fail(f"Transcription failed: {exc!s}")
            worker.logger.error(f"Transcription error: {exc}")
            return False
        worker.data["transcript"] = transcript
        if transcript:
            worker.transcript = transcript
        worker.logger.success(f"Transcription complete ({len(transcript)} segments)")
        return not worker.stopped(ClipStage.TRANSCRIBE)

    @staticmethod
    def _rounded(transcript: Segments) -> Segments:
        return [{**s, "start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3)}
                for s in transcript]

    @staticmethod
    def _reused(worker: ClipWorker) -> Optional[Segments]:
        if worker.config.get("force"):
            return None
        pretranscript = worker.config.get("pretranscript")
        if pretranscript:
            worker.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
            worker.logger.success(
                f"Using YouTube captions ({len(pretranscript)} segments), skipped transcription"
            )
            return pretranscript
        if worker.transcript:
            worker.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
            worker.logger.success(f"Reusing saved transcript ({len(worker.transcript)} segments)")
            return worker.transcript
        return None

    def _fresh(self, worker: ClipWorker) -> Segments:
        transcript = self._transcribe_video(
            worker.media_path,
            output_dir=worker.workdir,
            model_size=worker.config.get("whisper_model") or None,
            language=worker.config.get("language") or None,
        )
        worker.stage(ClipStage.TRANSCRIBE, StageStatus.DONE)
        return transcript
