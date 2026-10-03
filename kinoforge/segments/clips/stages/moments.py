from __future__ import annotations

from kinoforge.segments.clips.moments.finders import MomentFinder, MomentSpec
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.project import ProjectRecord
from kinoforge.segments.clips.worker import ClipStage, ClipWorker, Segments
from kinoforge.segments.common import Stage, StageStatus


class Moments(Stage[ClipWorker]):
    """Finds, ranks and limits moments, then records them in the project and assigns clip ids."""

    def __init__(self, finder: MomentFinder) -> None:
        self._finder = finder

    def __call__(self, worker: ClipWorker) -> bool:
        worker.stage(ClipStage.MOMENTS, StageStatus.RUNNING)
        worker.logger.info("Finding moments")
        try:
            moments = self._find(worker)
        except Exception as exc:
            worker.stage(ClipStage.MOMENTS, StageStatus.FAILED)
            worker.fail(f"Moment extraction failed: {exc!s}")
            worker.logger.error(f"Moment extraction error: {exc}")
            return False
        worker.data["moments"] = moments
        if worker.stopped(ClipStage.MOMENTS):
            return False
        if not moments:
            worker.stage(ClipStage.MOMENTS, StageStatus.DONE)
            worker.logger.warning("No moments extracted from video")
            return False
        used = self._limit(worker, moments)[: worker.config["clip_count"]]
        worker.data["used_moments"] = used
        self._persist(worker, used)
        worker.stage(ClipStage.MOMENTS, StageStatus.DONE)
        return True

    def _find(self, worker: ClipWorker) -> Segments:
        preset = worker.config.get("preset_moments")
        if preset:
            worker.logger.success(f"Using {len(preset)} preset moments (find skipped)")
            return MomentFinder.ranked([dict(m) for m in preset])
        moments = self._finder.find(worker.transcript, MomentSpec.from_config(worker.config))
        worker.logger.success(f"Found {len(moments)} moments")
        return moments

    def _limit(self, worker: ClipWorker, ranked: Segments) -> Segments:
        return self._lengths(worker, self._threshold(worker, ranked))

    @staticmethod
    def _threshold(worker: ClipWorker, ranked: Segments) -> Segments:
        threshold = worker.config.get("min_interest_score") or 0.0
        if threshold <= 0 or not ranked:
            return ranked
        scale = Moment.score_scale(ranked)
        kept = [m for m in ranked if Moment(m).score / scale >= threshold]
        target = int(worker.config.get("clip_count") or 0)
        if len(kept) >= max(target, 1):
            worker.logger.info(f"Kept {len(kept)}/{len(ranked)} moments scoring ≥ {threshold:.2f}.")
            return kept
        if kept:
            worker.logger.info(
                f"Only {len(kept)}/{len(ranked)} scored ≥ {threshold:.2f}; keeping the top "
                f"{min(target or len(ranked), len(ranked))} so the requested count is met."
            )
        else:
            worker.logger.warning(f"No moments scored ≥ {threshold:.2f}; using top results instead.")
        return ranked

    @staticmethod
    def _lengths(worker: ClipWorker, ranked: Segments) -> Segments:
        max_len = float(worker.config.get("max_length") or 0)
        for moment in ranked:
            Moment(moment).clamp_to(max_len)
        min_len = float(worker.config.get("min_length") or 0)
        if not min_len:
            return ranked
        usable = [m for m in ranked if Moment(m).span >= min_len]
        if usable:
            return usable
        worker.logger.warning(f"Every moment is shorter than {min_len:.0f}s; keeping them anyway.")
        return ranked

    @staticmethod
    def _persist(worker: ClipWorker, used: Segments) -> None:
        try:
            worker.record = ProjectRecord.build(
                worker.video_path or worker.media_path, worker.config, worker.slug, used, worker.transcript,
                existing=worker.record,
            )
        except Exception as exc:
            worker.logger.warning(f"Could not write project/clips: {exc}")
            return
        worker.logger.success("Saved project.json (editor source of truth)")
        worker.clip_ids = [
            f"clip_{index:02d}" if Moment(m).end > Moment(m).start else None
            for index, m in enumerate(used, 1)
        ]
        worker.logger.success(f"Added {sum(1 for c in worker.clip_ids if c)} clips to the pool")
