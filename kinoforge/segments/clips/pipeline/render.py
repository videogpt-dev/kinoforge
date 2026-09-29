from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.contract import Artifact
from kinoforge.segments.clips.pipeline.context import ClipRun, ClipStage, StageStatus
from kinoforge.segments.clips.render.clip_processor import extract_clips
from kinoforge.segments.clips.render.formatter import VariantFormatter
from kinoforge.segments.clips.render.media import AspectRatio


def render_clips(
    run: ClipRun, used: List[Dict[str, Any]], created_ids: List[Optional[str]]
) -> None:
    """Cut the chosen moments, format them to every requested aspect ratio, and place each
    finished clip in its pool slot. Scratch files live under <workdir>/_work and are removed."""
    sc = run.sc
    sc.stage(ClipStage.CLIPS, StageStatus.RUNNING)
    work = run.workdir / "_work"
    try:
        if run.video_path is None:
            sc.stage(ClipStage.CLIPS, StageStatus.FAILED)
            sc.fail("clip render requires input.video_path (audio-only run)")
            return
        if _render(run, used, work):
            _place_rendered(run, work, created_ids)
            sc.stage(ClipStage.CLIPS, StageStatus.DONE)
    except Exception as exc:
        sc.stage(ClipStage.CLIPS, StageStatus.FAILED)
        sc.fail(f"Clip extraction failed: {exc!s}")
        sc.logger.error(f"Clip extraction error: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    sc.logger.success("Processing complete!")


def _formats(run: ClipRun) -> List[str]:
    return [str(f) for f in run.config["formats"]] or [AspectRatio.PORTRAIT.value]


def _render(run: ClipRun, used: List[Dict[str, Any]], work: Path) -> bool:
    """False when cancelled between the cut and the format pass."""
    config, meter = run.config, run.ctx.meter
    clip_paths = extract_clips(
        run.video_path, used, work / "clips", config["quality"],
        max_workers=config["processing"]["max_workers"], meter=meter,
    )
    if run.sc.stopped(ClipStage.CLIPS):
        return False
    rendering = dict(config["rendering"])
    rendering["burn_subtitles"] = bool(
        config.get("generate_captions", rendering["burn_subtitles"])
    )
    VariantFormatter(
        _formats(run), run.transcript, rendering=rendering,
        processing=config["processing"], meter=meter,
    ).format_all(clip_paths, used, work / "formatted")
    return not run.sc.stopped(ClipStage.CLIPS)


def _rendered_file(work: Path, index: int, first_format: str) -> Optional[Path]:
    """The clip's primary-format render, else any format, else the raw cut."""
    formatted = work / "formatted"
    slug = AspectRatio(first_format).slug
    for pattern_dir, pattern in (
        (formatted, f"clip_{index:02d}_{slug}.mp4"),
        (formatted, f"clip_{index:02d}_*.mp4"),
        (work / "clips", f"clip_{index:02d}_raw.mp4"),
    ):
        found = sorted(pattern_dir.glob(pattern))
        if found:
            return found[0]
    return None


def _place_rendered(run: ClipRun, work: Path, created_ids: List[Optional[str]]) -> None:
    """Move each finished clip into its store slot, mark it rendered, and record the artifact."""
    store, sc = run.ctx.store, run.sc
    transcript = (store.load_record(run.job_id) or {}).get("transcript", [])
    first_format = _formats(run)[0]
    placed: List[str] = []
    for index, clip_id in enumerate(created_ids, 1):
        rendered = _rendered_file(work, index, first_format) if clip_id else None
        if rendered is None:
            continue
        destination = store.clip_path(run.job_id, clip_id) / "clip.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rendered), str(destination))
        store.mark_rendered(run.job_id, clip_id, "clip.mp4", transcript)
        placed.append(str(destination))
        sc.result.artifacts.append(Artifact(
            path=destination, media="video/mp4",
            meta={"clip_id": clip_id, "moment_index": index - 1},
        ))
    sc.result.data["clips"] = placed
    sc.logger.success(f"Auto-exported {len(placed)} clips")
