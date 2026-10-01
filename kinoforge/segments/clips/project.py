from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg

_PROBE_FIELDS = ("width", "height", "fps", "duration")


class ProjectRecord:
    """The editor's project record (project.json): source, probe, merged moments, transcript."""

    @classmethod
    def build(
        cls,
        video_path: Path,
        config: Dict[str, Any],
        slug: str,
        moments: List[Dict],
        transcript: List[Dict],
        existing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The record, with `moments` merged into the ones `existing` already holds."""
        probe = Ffmpeg.probe(video_path)
        return {
            "slug": slug,
            "title": config.get("title") or slug,
            "source": cls._relative_source(video_path, Path(config["output_dir"])),
            **{field: probe[field] for field in _PROBE_FIELDS},
            "formats": config.get("formats", []),
            "moments": cls._merge_moments((existing or {}).get("moments", []), moments),
            "transcript": [
                {"start": round(Moment(seg).start, 3), "end": round(Moment(seg).end, 3),
                 "text": seg.get("text", "")}
                for seg in transcript or []
            ],
        }

    @staticmethod
    def _relative_source(video_path: Path, output_dir: Path) -> str:
        try:
            return str(video_path.resolve().relative_to(output_dir.resolve()))
        except ValueError:
            return video_path.name

    @staticmethod
    def _merge_moments(existing: List[Dict], moments: List[Dict]) -> List[Dict]:
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
