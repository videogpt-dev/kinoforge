from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.contract import Context
from kinoforge.segments.clips.moments import MomentExtractor, score_and_rank_moments
from kinoforge.segments.clips.pipeline.context import (
    ClipStage,
    SaveProject,
    StageCtx,
    StageStatus,
)


def find_moments(
    sc: StageCtx,
    provider: object,
    transcript: List[Dict[str, Any]],
    media_path: Path,
    config: Dict[str, Any],
) -> Optional[tuple[List[Dict[str, Any]], bool]]:
    """Returns (moments, discovered), or None when the run must stop."""
    sc.stage(ClipStage.MOMENTS, StageStatus.RUNNING)
    sc.logger.info("Finding moments")
    discovered = False
    try:
        moments: List[Dict[str, Any]] = []
        # Source-light render hands the moments in and asks only to render them: the host already
        # found them (on the audio) in an earlier analyze pass, so skip discovery and use them.
        preset = config.get("preset_moments")
        if preset:
            moments = [dict(m) for m in preset]
            discovered = True
            sc.logger.success(f"Using {len(moments)} preset moments (find skipped)")
        elif provider is not None and hasattr(provider, "discover_moments"):
            moments = (
                provider.discover_moments(
                    transcript, config["min_length"], config["max_length"], config["clip_count"]
                )
                or []
            )
            discovered = bool(moments)
            if discovered:
                sc.logger.success(f"Read the transcript and chose {len(moments)} moments")
        if not moments:
            moments = MomentExtractor(
                min_length=config["min_length"],
                max_length=config["max_length"],
                target_clips=config["clip_count"],
                verbose=config["verbose"],
            ).auto(media_path, transcript)
            sc.logger.success(f"Moment extraction: {len(moments)} moments found")
        sc.result.data["moments"] = moments
        if sc.stopped(ClipStage.MOMENTS):
            return None
        return moments, discovered
    except Exception as exc:
        sc.stage(ClipStage.MOMENTS, StageStatus.FAILED)
        sc.fail(f"Moment extraction failed: {exc!s}")
        sc.logger.error(f"Moment extraction error: {exc}")
        return None


def rank_moments(
    sc: StageCtx,
    provider: object,
    moments: List[Dict[str, Any]],
    discovered: bool,
    transcript: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if discovered:
        return sorted(moments, key=lambda moment: float(moment.get("score") or 0), reverse=True)

    if provider and hasattr(provider, "filter_moments"):
        try:
            filtered = provider.filter_moments(moments, transcript)
            if filtered:
                moments = filtered
                sc.logger.success(
                    f"Filtered moments using {getattr(provider, 'name', 'provider')}"
                )
        except Exception as exc:
            sc.logger.warning(f"Provider filtering failed: {exc}")

    sc.logger.info("Scoring moments")
    try:
        if provider and hasattr(provider, "score_moments"):
            moments = provider.score_moments(moments, transcript)
            sc.logger.success(f"Scored moments with {getattr(provider, 'name', 'provider')}")
        else:
            moments = score_and_rank_moments(moments, transcript)
            sc.logger.success("Moments scored and ranked")
    except Exception as exc:
        sc.logger.warning(f"Moment scoring failed: {exc}")
    return moments


def apply_limits(
    sc: StageCtx, ranked: List[Dict[str, Any]], config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    threshold = config.get("min_interest_score") or 0.0
    if threshold > 0 and ranked:
        scores = [float(moment.get("score") or 0) for moment in ranked]
        scale = 100.0 if max(scores) > 10 else 10.0
        kept = [m for m in ranked if (float(m.get("score") or 0) / scale) >= threshold]
        if not kept:
            sc.logger.warning(f"No moments scored ≥ {threshold:.2f}; using top results instead.")
        else:
            ranked = kept
            sc.logger.info(f"Kept {len(kept)}/{len(scores)} moments scoring ≥ {threshold:.2f}.")

    min_len = float(config.get("min_length") or 0)
    max_len = float(config.get("max_length") or 0)
    for moment in ranked:
        if max_len and float(moment.get("end") or 0) - float(moment.get("start") or 0) > max_len:
            moment["end"] = float(moment["start"]) + max_len
            moment["duration"] = max_len
    if min_len:
        usable = [
            m for m in ranked if float(m.get("end") or 0) - float(m.get("start") or 0) >= min_len
        ]
        if usable:
            ranked = usable
        else:
            sc.logger.warning(f"Every moment is shorter than {min_len:.0f}s; keeping them anyway.")
    return ranked


def persist_moments(
    sc: StageCtx,
    save_project: SaveProject,
    source_path: Path,
    slug: str,
    used_moments: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    job_id: str,
    config: Dict[str, Any],
    ctx: Context,
) -> List[Optional[str]]:
    created_ids: List[Optional[str]] = []
    try:
        save_project(source_path, config, slug, used_moments, transcript, ctx.store)
        sc.logger.success("Saved project.json (editor source of truth)")
        project = ctx.store.load_record(job_id)
        if project is not None:
            created_ids = ctx.store.append_clips(job_id, project, used_moments)
            sc.logger.success(
                f"Added {sum(1 for clip_id in created_ids if clip_id)} clips to the pool"
            )
    except Exception as exc:
        sc.logger.warning(f"Could not write project/clips: {exc}")
    return created_ids
