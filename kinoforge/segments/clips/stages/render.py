from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from kinoforge.segments.clips.render.clip_processor import ClipCutter
from kinoforge.segments.clips.render.formatter import VariantFormatter
from kinoforge.segments.clips.render.media import AspectRatio
from kinoforge.segments.clips.worker import ClipStage, ClipWorker
from kinoforge.segments.common import Stage, StageStatus


class Render(Stage[ClipWorker]):
    """Cuts the chosen moments, formats them to every requested aspect ratio, and places each
    finished clip at <workdir>/artifacts/<clip_id>/clip.mp4. Skipped on analyze-only runs."""

    def __call__(self, worker: ClipWorker) -> bool:
        if worker.config.get("analyze_only"):
            worker.stage(ClipStage.CLIPS, StageStatus.SKIPPED)
            worker.logger.success("Analyze-only: skipping clip extraction (open in editor)")
            return True
        if worker.stopped():
            return False
        worker.stage(ClipStage.CLIPS, StageStatus.RUNNING)
        work = worker.workdir / "_work"
        try:
            if worker.video_path is None:
                worker.stage(ClipStage.CLIPS, StageStatus.FAILED)
                worker.fail("clip render requires input.video_path (audio-only run)")
                return False
            if self._render(worker, worker.video_path, work):
                self._place(worker, work)
                worker.stage(ClipStage.CLIPS, StageStatus.DONE)
        except Exception as exc:
            worker.stage(ClipStage.CLIPS, StageStatus.FAILED)
            worker.fail(f"Clip extraction failed: {exc!s}")
            worker.logger.error(f"Clip extraction error: {exc}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        worker.logger.success("Processing complete!")
        return True

    @staticmethod
    def _formats(worker: ClipWorker) -> List[str]:
        return [str(f) for f in worker.config["formats"]] or [AspectRatio.PORTRAIT.value]

    def _render(self, worker: ClipWorker, video: Path, work: Path) -> bool:
        config, used = worker.config, worker.data["used_moments"]
        clip_paths = ClipCutter(video, config["quality"]).cut_all(
            used, work / "clips", max_workers=config["processing"]["max_workers"],
            meter=worker.meter,
        )
        if worker.stopped(ClipStage.CLIPS):
            return False
        rendering = dict(config["rendering"])
        rendering["burn_subtitles"] = bool(
            config.get("generate_captions", rendering["burn_subtitles"])
        )
        VariantFormatter(
            self._formats(worker), worker.transcript, rendering=rendering,
            processing=config["processing"], meter=worker.meter,
        ).format_all(clip_paths, used, work / "formatted")
        return not worker.stopped(ClipStage.CLIPS)

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

    def _place(self, worker: ClipWorker, work: Path) -> None:
        first_format = self._formats(worker)[0]
        placed: List[str] = []
        for index, clip_id in enumerate(worker.clip_ids, 1):
            rendered = self._rendered_file(work, index, first_format) if clip_id else None
            if clip_id is None or rendered is None:
                continue
            destination = worker.workdir / "artifacts" / clip_id / "clip.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rendered), str(destination))
            placed.append(str(destination))
            worker.artifacts.append({
                "path": str(destination), "media": "video/mp4",
                "meta": {"clip_id": clip_id, "moment_index": index - 1},
            })
        worker.data["clips"] = placed
        wanted = sum(1 for clip_id in worker.clip_ids if clip_id)
        if not placed and wanted:
            raise RuntimeError(f"none of the {wanted} clips rendered")
        if len(placed) < wanted:
            worker.logger.warning(f"Rendered {len(placed)}/{wanted} clips; the rest failed")
        else:
            worker.logger.success(f"Auto-exported {len(placed)} clips")
