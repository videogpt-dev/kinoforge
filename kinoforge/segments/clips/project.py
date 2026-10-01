"""Editor project record (project.json): source, probe, merged moments and transcript."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.render.probe import get_video_metadata

_PROBE_FIELDS = ("width", "height", "fps", "duration")


def _relative_source(video_path: Path, output_dir: Path) -> str:
    try:
        return str(video_path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return video_path.name


def _merge_moments(existing: List[Dict], moments: List[Dict]) -> List[Dict]:
    """Existing moments plus the new ones whose span is not already there, renumbered 1..n."""
    merged = list(existing)
    seen = {(round(Moment(m).start, 2), round(Moment(m).end, 2)) for m in merged}
    for moment in moments:
        view = Moment(moment)
        key = (round(view.start, 2), round(view.end, 2))
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "start": round(view.start, 3),
            "end": round(view.end, 3),
            "text": moment.get("text", ""),
            "score": round(float(moment.get("score", moment.get("ai_score", 0)) or 0), 2),
        })
    for index, moment in enumerate(merged, 1):
        moment["id"] = index
    return merged


def _transcript_rows(transcript: List[Dict]) -> List[Dict]:
    return [
        {"start": round(Moment(seg).start, 3), "end": round(Moment(seg).end, 3),
         "text": seg.get("text", "")}
        for seg in transcript or []
    ]


def build_record(
    video_path: Path,
    config: Dict[str, Any],
    slug: str,
    moments: List[Dict],
    transcript: List[Dict],
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The project record, with `moments` merged into the ones `existing` already holds."""
    probe = get_video_metadata(video_path)
    existing = existing or {}
    return {
        "slug": slug,
        "title": config.get("title") or slug,
        "source": _relative_source(video_path, Path(config["output_dir"])),
        **{field: probe[field] for field in _PROBE_FIELDS},
        "formats": config.get("formats", []),
        "moments": _merge_moments(existing.get("moments", []), moments),
        "transcript": _transcript_rows(transcript),
    }
