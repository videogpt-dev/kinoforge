"""Editor project record builder, independent from product storage."""

from pathlib import Path
from typing import Any, Dict, List

from kinoforge.contract import ProjectStore


def probe_video(video_path: Path) -> Dict[str, Any]:
    import json as _json
    import subprocess

    info: Dict[str, Any] = {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "duration": 0.0,
    }
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        data = _json.loads(out.stdout or "{}")
        vstream: Dict[str, Any] = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        if vstream.get("width"):
            info["width"] = int(vstream["width"])
        if vstream.get("height"):
            info["height"] = int(vstream["height"])
        rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "30/1"
        num, _, den = rate.partition("/")
        if den and float(den) != 0:
            info["fps"] = round(float(num) / float(den), 3)
        duration = (data.get("format", {}) or {}).get("duration")
        if duration:
            info["duration"] = float(duration)
    except Exception:
        pass
    return info


def write_project(
    video_path: Path,
    config: Dict[str, Any],
    slug: str,
    moments: List[Dict],
    transcript: List[Dict],
    store: ProjectStore,
) -> None:
    output_dir = Path(config["output_dir"])
    try:
        rel_source = str(video_path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        rel_source = video_path.name
    probe = probe_video(video_path)

    existing = store.load_record(slug) or {}
    merged = list(existing.get("moments", []))
    seen = {
        (round(float(moment.get("start", 0)), 2), round(float(moment.get("end", 0)), 2))
        for moment in merged
    }
    for moment in moments:
        key = (
            round(float(moment.get("start", 0)), 2),
            round(float(moment.get("end", 0)), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "start": round(float(moment.get("start", 0)), 3),
                "end": round(float(moment.get("end", 0)), 3),
                "text": moment.get("text", ""),
                "score": round(
                    float(moment.get("score", moment.get("ai_score", 0)) or 0),
                    2,
                ),
            }
        )
    for index, moment in enumerate(merged, 1):
        moment["id"] = index

    store.save_record(
        slug,
        {
            "slug": slug,
            "title": config.get("title") or slug,
            "source": rel_source,
            **probe,
            "formats": config.get("formats", []),
            "moments": merged,
            "transcript": [
                {
                    "start": round(float(segment.get("start", 0)), 3),
                    "end": round(float(segment.get("end", 0)), 3),
                    "text": segment.get("text", ""),
                }
                for segment in (transcript or [])
            ],
        },
    )
