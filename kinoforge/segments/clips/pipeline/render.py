from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.contract import Artifact, Context
from kinoforge.segments.clips.pipeline.context import ClipStage, StageCtx, StageStatus
from kinoforge.segments.clips.render.clip_processor import extract_clips
from kinoforge.segments.clips.render.formatter import format_clips_multi_platform


def render_clips(
    sc: StageCtx,
    video_path: Optional[Path],
    used_moments: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    created_ids: List[Optional[str]],
    video_out: Path,
    job_id: str,
    config: Dict[str, Any],
    ctx: Context,
) -> None:
    sc.stage(ClipStage.CLIPS, StageStatus.RUNNING)
    try:
        if video_path is None:
            sc.stage(ClipStage.CLIPS, StageStatus.FAILED)
            sc.fail("clip render requires input.video_path (audio-only run)")
            return
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
        if sc.stopped(ClipStage.CLIPS):
            return

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
        if sc.stopped(ClipStage.CLIPS):
            return

        _place_rendered(sc, formatted_dir, work, created_ids, formats, job_id, ctx)
        sc.stage(ClipStage.CLIPS, StageStatus.DONE)
    except Exception as exc:
        sc.stage(ClipStage.CLIPS, StageStatus.FAILED)
        sc.fail(f"Clip extraction failed: {exc!s}")
        sc.logger.error(f"Clip extraction error: {exc}")
    finally:
        shutil.rmtree(video_out / "_work", ignore_errors=True)
    sc.logger.success("Processing complete!")


def _place_rendered(
    sc: StageCtx,
    formatted_dir: Path,
    work: Path,
    created_ids: List[Optional[str]],
    formats: List[str],
    job_id: str,
    ctx: Context,
) -> None:
    """Move each finished clip into its store slot, mark it rendered, and record the artifact."""
    transcript_segments = (ctx.store.load_record(job_id) or {}).get("transcript", [])
    placed = 0
    output_paths: List[str] = []
    for index, clip_id in enumerate(created_ids, 1):
        if not clip_id:
            continue
        candidates = (
            sorted(formatted_dir.glob(f"clip_{index:02d}_{formats[0].replace(':', 'x')}.mp4"))
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
        sc.result.artifacts.append(
            Artifact(
                path=destination,
                media="video/mp4",
                meta={"clip_id": clip_id, "moment_index": index - 1},
            )
        )
        placed += 1
    sc.result.data["clips"] = output_paths
    sc.logger.success(f"Auto-exported {placed} clips")
