from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from kinoforge.segments.clips.render.clip_processor import ClipCutter
from kinoforge.segments.clips.render.formatter import VariantFormatter
from kinoforge.segments.clips.render.media import AspectRatio
from kinoforge.segments.clips.run import ClipRun, ClipStage, StageStatus


class Render:
    """Cuts the chosen moments, formats them to every requested aspect ratio, and places each
    finished clip at <workdir>/artifacts/<clip_id>/clip.mp4. Skipped on analyze-only runs."""

    def __call__(self, run: ClipRun) -> bool:
        if run.config.get("analyze_only"):
            run.stage(ClipStage.CLIPS, StageStatus.SKIPPED)
            run.logger.success("Analyze-only: skipping clip extraction (open in editor)")
            return True
        if run.stopped():
            return False
        run.stage(ClipStage.CLIPS, StageStatus.RUNNING)
        work = run.workdir / "_work"
        try:
            if run.video_path is None:
                run.stage(ClipStage.CLIPS, StageStatus.FAILED)
                run.fail("clip render requires input.video_path (audio-only run)")
                return False
            if self._render(run, run.video_path, work):
                self._place(run, work)
                run.stage(ClipStage.CLIPS, StageStatus.DONE)
        except Exception as exc:
            run.stage(ClipStage.CLIPS, StageStatus.FAILED)
            run.fail(f"Clip extraction failed: {exc!s}")
            run.logger.error(f"Clip extraction error: {exc}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        run.logger.success("Processing complete!")
        return True

    @staticmethod
    def _formats(run: ClipRun) -> List[str]:
        return [str(f) for f in run.config["formats"]] or [AspectRatio.PORTRAIT.value]

    def _render(self, run: ClipRun, video: Path, work: Path) -> bool:
        config, used = run.config, run.data["used_moments"]
        clip_paths = ClipCutter(video, config["quality"]).cut_all(
            used, work / "clips", max_workers=config["processing"]["max_workers"],
            meter=run.meter,
        )
        if run.stopped(ClipStage.CLIPS):
            return False
        rendering = dict(config["rendering"])
        rendering["burn_subtitles"] = bool(
            config.get("generate_captions", rendering["burn_subtitles"])
        )
        VariantFormatter(
            self._formats(run), run.transcript, rendering=rendering,
            processing=config["processing"], meter=run.meter,
        ).format_all(clip_paths, used, work / "formatted")
        return not run.stopped(ClipStage.CLIPS)

    @staticmethod
    def _rendered_file(work: Path, index: int, first_format: str) -> Optional[Path]:
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

    def _place(self, run: ClipRun, work: Path) -> None:
        first_format = self._formats(run)[0]
        placed: List[str] = []
        for index, clip_id in enumerate(run.clip_ids, 1):
            rendered = self._rendered_file(work, index, first_format) if clip_id else None
            if clip_id is None or rendered is None:
                continue
            destination = run.workdir / "artifacts" / clip_id / "clip.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rendered), str(destination))
            placed.append(str(destination))
            run.artifacts.append({
                "path": str(destination), "media": "video/mp4",
                "meta": {"clip_id": clip_id, "moment_index": index - 1},
            })
        run.data["clips"] = placed
        run.logger.success(f"Auto-exported {len(placed)} clips")
