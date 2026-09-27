from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

_CRF = {"max": "14", "high": "18", "standard": "23"}


def _even(n: float) -> int:
    return 2 * int(round(n / 2))


def render_base_clip(
    source: Path,
    out: Path,
    in_sec: float,
    out_sec: float,
    crop: Optional[Dict],
    src_w: int,
    src_h: int,
    quality: str = "high",
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> bool:
    crop = crop or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    cx, cy = _even(crop["x"] * src_w), _even(crop["y"] * src_h)
    cw = min(_even(crop["w"] * src_w), src_w - cx)
    ch = min(_even(crop["h"] * src_h), src_h - cy)
    cw, ch = max(2, cw - cw % 2), max(2, ch - ch % 2)

    crf = _CRF.get(quality, _CRF["high"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(in_sec), "-to", str(out_sec),
        "-i", str(source),
        "-vf", f"crop={cw}:{ch}:{cx}:{cy}",
        "-c:v", "libx264", "-preset", "medium", "-crf", crf, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        while True:
            try:
                return process.wait(timeout=0.25) == 0 and out.exists()
            except subprocess.TimeoutExpired:
                if is_cancelled and is_cancelled():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    out.unlink(missing_ok=True)
                    return False
    except OSError:
        if process and process.poll() is None:
            process.kill()
        return False
