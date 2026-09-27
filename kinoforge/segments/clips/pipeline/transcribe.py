from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.contract import Context
from kinoforge.segments.clips.pipeline.context import (
    ClipStage,
    StageCtx,
    StageStatus,
    TranscribeVideo,
)


def transcribe(
    sc: StageCtx,
    transcribe_video: TranscribeVideo,
    media_path: Path,
    video_out: Path,
    job_id: str,
    config: Dict[str, Any],
    ctx: Context,
) -> Optional[List[Dict[str, Any]]]:
    """Returns the transcript, or None when the run must stop (failed or cancelled)."""
    sc.stage(ClipStage.TRANSCRIBE, StageStatus.RUNNING)
    sc.logger.info("Transcribing")
    try:
        restart = bool(config.get("force"))
        fresh = restart or not config.get("try_youtube_subs", True)
        pretranscript = None if restart else config.get("pretranscript")
        saved = ctx.store.load_transcript(job_id) if not fresh else None

        if pretranscript:
            transcript = pretranscript
            sc.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
            sc.logger.success(
                f"Using YouTube captions ({len(transcript)} segments), skipped transcription"
            )
        elif saved:
            transcript = saved
            sc.stage(ClipStage.TRANSCRIBE, StageStatus.SKIPPED)
            sc.logger.success(f"Reusing saved transcript ({len(transcript)} segments)")
        else:
            transcript = transcribe_video(
                media_path,
                output_dir=video_out,
                model_size=config.get("whisper_model") or None,
                language=config.get("language") or None,
            )
            sc.stage(ClipStage.TRANSCRIBE, StageStatus.DONE)
        sc.result.data["transcript"] = transcript
        if transcript:
            ctx.store.save_transcript(job_id, transcript)
        sc.logger.success(f"Transcription complete ({len(transcript)} segments)")
        if sc.stopped(ClipStage.TRANSCRIBE):
            return None
        return transcript
    except Exception as exc:
        sc.stage(ClipStage.TRANSCRIBE, StageStatus.FAILED)
        sc.fail(f"Transcription failed: {exc!s}")
        sc.logger.error(f"Transcription error: {exc}")
        return None
