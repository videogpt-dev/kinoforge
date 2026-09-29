from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from kinoforge.observ import active

_FALLBACK = {"width": 1920, "height": 1080, "duration": 0, "aspect_ratio": 16 / 9, "fps": 30.0}


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Width, height, duration, aspect ratio and fps from ffprobe; 1080p/30 when unreadable."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
           str(video_path)]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, check=True, timeout=30).stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        active().warning(f"    Could not read video info: {exc}")
        return dict(_FALLBACK)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    width, height = int(video.get("width") or 1920), int(video.get("height") or 1080)
    return {
        "width": width,
        "height": height,
        "duration": float((data.get("format") or {}).get("duration") or 0),
        "aspect_ratio": width / max(height, 1),
        # avg_frame_rate is the real rate for variable-frame-rate sources; r_frame_rate is the
        # container's nominal one.
        "fps": _fps(video.get("avg_frame_rate")) or _fps(video.get("r_frame_rate")) or 30.0,
    }


def _fps(value: object) -> float:
    """ffprobe rate ("30000/1001" or "25") as fps rounded to 3 places; 0 when unusable."""
    try:
        num, _, den = str(value or "").partition("/")
        return round(float(num) / float(den) if den else float(num), 3)
    except (ValueError, ZeroDivisionError):
        return 0.0
