from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from kinoforge.segments.clips.moments import HeuristicScorer, MomentExtractor
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.pipeline.context import (
    ClipRun,
    ClipStage,
    SaveProject,
    StageCtx,
    StageStatus,
)

Moments = List[Dict[str, Any]]


def find_moments(run: ClipRun, provider: Any) -> Optional[Tuple[Moments, bool]]:
    """(moments, discovered): preset moments, else the provider's own discovery, else the
    offline extractor. None when the run must stop."""
    sc = run.sc
    sc.stage(ClipStage.MOMENTS, StageStatus.RUNNING)
    sc.logger.info("Finding moments")
    try:
        moments, discovered = _preset(run) or _discovered(run, provider) or (_extracted(run), False)
        sc.result.data["moments"] = moments
        if sc.stopped(ClipStage.MOMENTS):
            return None
        return moments, discovered
    except Exception as exc:
        sc.stage(ClipStage.MOMENTS, StageStatus.FAILED)
        sc.fail(f"Moment extraction failed: {exc!s}")
        sc.logger.error(f"Moment extraction error: {exc}")
        return None


def _preset(run: ClipRun) -> Optional[Tuple[Moments, bool]]:
    """Source-light render: the host already found these (on the audio) in an analyze pass."""
    preset = run.config.get("preset_moments")
    if not preset:
        return None
    run.sc.logger.success(f"Using {len(preset)} preset moments (find skipped)")
    return [dict(m) for m in preset], True


def _discovered(run: ClipRun, provider: Any) -> Optional[Tuple[Moments, bool]]:
    if not hasattr(provider, "discover_moments"):
        return None
    config = run.config
    moments = provider.discover_moments(
        run.transcript, config["min_length"], config["max_length"], config["clip_count"]
    ) or []
    if not moments:
        return None
    run.sc.logger.success(f"Read the transcript and chose {len(moments)} moments")
    return moments, True


def _extracted(run: ClipRun) -> Moments:
    config = run.config
    moments = MomentExtractor(
        min_length=config["min_length"], max_length=config["max_length"],
        target_clips=config["clip_count"], verbose=config["verbose"],
    ).auto(run.media_path, run.transcript)
    run.sc.logger.success(f"Moment extraction: {len(moments)} moments found")
    return moments


def rank_moments(run: ClipRun, provider: Any, moments: Moments, discovered: bool) -> Moments:
    """Discovered moments come scored; others are filtered + scored by the provider when it
    can, else by the offline scorer."""
    if discovered:
        return sorted(moments, key=lambda moment: Moment(moment).score, reverse=True)
    sc, name = run.sc, getattr(provider, "name", "provider")
    if hasattr(provider, "filter_moments"):
        try:
            filtered = provider.filter_moments(moments, run.transcript)
            if filtered:
                moments = filtered
                sc.logger.success(f"Filtered moments using {name}")
        except Exception as exc:
            sc.logger.warning(f"Provider filtering failed: {exc}")
    sc.logger.info("Scoring moments")
    try:
        if hasattr(provider, "score_moments"):
            moments = provider.score_moments(moments, run.transcript)
            sc.logger.success(f"Scored moments with {name}")
        else:
            moments = HeuristicScorer.rank(moments)
            sc.logger.success("Moments scored and ranked")
    except Exception as exc:
        sc.logger.warning(f"Moment scoring failed: {exc}")
    return moments


def apply_limits(
    sc: StageCtx, ranked: List[Dict[str, Any]], config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    ranked = _apply_threshold(sc, ranked, config)
    return _apply_lengths(sc, ranked, config)


def _apply_threshold(
    sc: StageCtx, ranked: List[Dict[str, Any]], config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    threshold = config.get("min_interest_score") or 0.0
    if threshold <= 0 or not ranked:
        return ranked
    scale = Moment.score_scale(ranked)
    kept = [m for m in ranked if Moment(m).score / scale >= threshold]
    target = int(config.get("clip_count") or 0)
    # Trim to above-threshold only when that already meets the count; else keep the full
    # ordered list so the runner's clip_count slice is not starved.
    if len(kept) >= max(target, 1):
        sc.logger.info(f"Kept {len(kept)}/{len(ranked)} moments scoring ≥ {threshold:.2f}.")
        return kept
    if kept:
        sc.logger.info(
            f"Only {len(kept)}/{len(ranked)} scored ≥ {threshold:.2f}; keeping the top "
            f"{min(target or len(ranked), len(ranked))} so the requested count is met."
        )
    else:
        sc.logger.warning(f"No moments scored ≥ {threshold:.2f}; using top results instead.")
    return ranked


def _apply_lengths(
    sc: StageCtx, ranked: List[Dict[str, Any]], config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    max_len = float(config.get("max_length") or 0)
    for moment in ranked:
        Moment(moment).clamp_to(max_len)
    min_len = float(config.get("min_length") or 0)
    if not min_len:
        return ranked
    usable = [m for m in ranked if Moment(m).span >= min_len]
    if usable:
        return usable
    sc.logger.warning(f"Every moment is shorter than {min_len:.0f}s; keeping them anyway.")
    return ranked


def persist_moments(
    run: ClipRun, save_project: SaveProject, used: Moments
) -> List[Optional[str]]:
    """Write project.json and add the chosen moments to the clip pool; the new clip ids."""
    sc, store = run.sc, run.ctx.store
    try:
        save_project(run.video_path or run.media_path, run.config, run.slug, used,
                     run.transcript, store)
        sc.logger.success("Saved project.json (editor source of truth)")
        project = store.load_record(run.job_id)
        if project is None:
            return []
        created_ids = store.append_clips(run.job_id, project, used)
        sc.logger.success(f"Added {sum(1 for c in created_ids if c)} clips to the pool")
        return created_ids
    except Exception as exc:
        sc.logger.warning(f"Could not write project/clips: {exc}")
        return []
