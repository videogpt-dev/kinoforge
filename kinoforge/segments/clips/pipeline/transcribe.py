from __future__ import annotations

from typing import Any, Dict, List, Optional

from kinoforge.segments.clips.pipeline.context import (
    ClipRun,
    ClipStage,
    StageStatus,
    TranscribeVideo,
)


def transcribe(run: ClipRun, transcribe_video: TranscribeVideo) -> Optional[List[Dict[str, Any]]]:
    """Sets run.transcript and returns it, or None when the run must stop (failed/cancelled)."""
    sc = run.sc
    sc.stage(ClipStage.TRANSCRIBE, StageStatus.RUNNING)
    sc.logger.info("Transcribing")
    try:
        transcript = _reused(run) or _fresh(run, transcribe_video)
        sc.result.data["transcript"] = transcript
        if transcript:
            run.ctx.store.save_transcript(run.job_id, transcript)
        sc.logger.success(f"Transcription complete ({len(transcript)} segments)")
        if sc.stopped(ClipStage.TRANSCRIBE):
            return None
        run.transcript = transcript
        return transcript
    except Exception as exc:
        sc.stage(ClipStage.TRANSCRIBE, StageStatus.FAILED)
        sc.fail(f"Transcription failed: {exc!s}")
        sc.logger.error(f"Transcription error: {exc}")
        return None


def _reused(run: ClipRun) -> Optional[List[Dict[str, Any]]]:
    """Host-supplied captions, else a transcript saved by an earlier run (unless forced)."""
    config, sc = run.config, run.sc
    if config.get("force"):
        return None
    pretranscript = config.get("pretranscript")
    if pretranscript:
        sc.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
        sc.logger.success(
            f"Using YouTube captions ({len(pretranscript)} segments), skipped transcription"
        )
        return pretranscript
    saved = (run.ctx.store.load_transcript(run.job_id)
             if config.get("try_youtube_subs", True) else None)
    if saved:
        sc.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
        sc.logger.success(f"Reusing saved transcript ({len(saved)} segments)")
    return saved


def _fresh(run: ClipRun, transcribe_video: TranscribeVideo) -> List[Dict[str, Any]]:
    transcript = transcribe_video(
        run.media_path,
        output_dir=run.workdir,
        model_size=run.config.get("whisper_model") or None,
        language=run.config.get("language") or None,
    )
    run.sc.stage(ClipStage.TRANSCRIBE, StageStatus.DONE)
    return transcript
