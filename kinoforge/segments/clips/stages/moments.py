from __future__ import annotations

from typing import Tuple

from kinoforge.segments.clips.moments.engines import MomentEngine
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.moments.transcript import TranscriptText
from kinoforge.segments.clips.project import ProjectRecord
from kinoforge.segments.clips.run import ClipRun, ClipStage, Segments, StageStatus


class Moments:
    """Finds, ranks and limits moments, then records them in the project and assigns clip ids."""

    def __init__(self, engine: MomentEngine) -> None:
        self._engine = engine

    def __call__(self, run: ClipRun) -> bool:
        run.stage(ClipStage.MOMENTS, StageStatus.RUNNING)
        run.logger.info("Finding moments")
        try:
            moments, discovered = self._find(run)
        except Exception as exc:
            run.stage(ClipStage.MOMENTS, StageStatus.FAILED)
            run.fail(f"Moment extraction failed: {exc!s}")
            run.logger.error(f"Moment extraction error: {exc}")
            return False
        run.data["moments"] = moments
        if run.stopped(ClipStage.MOMENTS):
            return False
        if not moments:
            run.stage(ClipStage.MOMENTS, StageStatus.DONE)
            run.logger.warning("No moments extracted from video")
            return False
        if discovered:
            ranked = sorted(moments, key=lambda m: Moment(m).score, reverse=True)
        else:
            ranked = self._rank(run, moments)
        used = self._limit(run, ranked)[: run.config["clip_count"]]
        run.data["used_moments"] = used
        self._persist(run, used)
        run.stage(ClipStage.MOMENTS, StageStatus.DONE)
        return True

    def _find(self, run: ClipRun) -> Tuple[Segments, bool]:
        preset = run.config.get("preset_moments")
        if preset:
            run.logger.success(f"Using {len(preset)} preset moments (find skipped)")
            return [dict(m) for m in preset], True
        config = run.config
        moments = self._engine.discover_moments(
            run.transcript, config["min_length"], config["max_length"], config["clip_count"]
        )
        if moments:
            run.logger.success(f"Read the transcript and chose {len(moments)} moments")
            return moments, True
        moments = TranscriptText.windows(run.transcript, config["min_length"],
                                         config["max_length"])
        run.logger.success(f"Moment extraction: {len(moments)} candidate windows")
        return moments, False

    def _rank(self, run: ClipRun, moments: Segments) -> Segments:
        engine = self._engine
        try:
            filtered = engine.filter_moments(moments, run.transcript)
            if filtered:
                moments = filtered
                run.logger.success(f"Filtered moments using {engine.name}")
        except Exception as exc:
            run.logger.warning(f"Provider filtering failed: {exc}")
        run.logger.info("Scoring moments")
        try:
            moments = engine.score_moments(moments, run.transcript)
            run.logger.success(f"Scored moments with {engine.name}")
        except Exception as exc:
            run.logger.warning(f"Moment scoring failed: {exc}")
        return moments

    def _limit(self, run: ClipRun, ranked: Segments) -> Segments:
        return self._lengths(run, self._threshold(run, ranked))

    @staticmethod
    def _threshold(run: ClipRun, ranked: Segments) -> Segments:
        threshold = run.config.get("min_interest_score") or 0.0
        if threshold <= 0 or not ranked:
            return ranked
        scale = Moment.score_scale(ranked)
        kept = [m for m in ranked if Moment(m).score / scale >= threshold]
        target = int(run.config.get("clip_count") or 0)
        if len(kept) >= max(target, 1):
            run.logger.info(f"Kept {len(kept)}/{len(ranked)} moments scoring ≥ {threshold:.2f}.")
            return kept
        if kept:
            run.logger.info(
                f"Only {len(kept)}/{len(ranked)} scored ≥ {threshold:.2f}; keeping the top "
                f"{min(target or len(ranked), len(ranked))} so the requested count is met."
            )
        else:
            run.logger.warning(f"No moments scored ≥ {threshold:.2f}; using top results instead.")
        return ranked

    @staticmethod
    def _lengths(run: ClipRun, ranked: Segments) -> Segments:
        max_len = float(run.config.get("max_length") or 0)
        for moment in ranked:
            Moment(moment).clamp_to(max_len)
        min_len = float(run.config.get("min_length") or 0)
        if not min_len:
            return ranked
        usable = [m for m in ranked if Moment(m).span >= min_len]
        if usable:
            return usable
        run.logger.warning(f"Every moment is shorter than {min_len:.0f}s; keeping them anyway.")
        return ranked

    @staticmethod
    def _persist(run: ClipRun, used: Segments) -> None:
        try:
            run.record = ProjectRecord.build(
                run.video_path or run.media_path, run.config, run.slug, used, run.transcript,
                existing=run.record,
            )
        except Exception as exc:
            run.logger.warning(f"Could not write project/clips: {exc}")
            return
        run.logger.success("Saved project.json (editor source of truth)")
        run.clip_ids = [
            f"clip_{index:02d}" if Moment(m).end > Moment(m).start else None
            for index, m in enumerate(used, 1)
        ]
        run.logger.success(f"Added {sum(1 for c in run.clip_ids if c)} clips to the pool")
