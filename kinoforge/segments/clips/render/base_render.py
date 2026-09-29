"""The editor's base render: one cropped, trimmed master from the source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from kinoforge.segments.clips.render import ffmpeg
from kinoforge.segments.clips.render.media import MasterQuality


def _even(n: float) -> int:
    return 2 * int(round(n / 2))


@dataclass(frozen=True)
class BaseCut:
    """[start, end] of `source`, cropped to a normalized box ({x, y, w, h} in 0..1, None =
    full frame), encoded at `quality` into `output`."""

    source: Path
    output: Path
    start: float
    end: float
    source_size: Tuple[int, int]
    crop: Optional[Dict[str, float]] = None
    quality: MasterQuality = MasterQuality.HIGH

    def crop_filter(self) -> str:
        """Pixel crop, snapped to even sizes (x264 needs them) and kept inside the frame."""
        box = self.crop or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        width, height = self.source_size
        x, y = _even(box["x"] * width), _even(box["y"] * height)
        w = min(_even(box["w"] * width), width - x)
        h = min(_even(box["h"] * height), height - y)
        w, h = max(2, w - w % 2), max(2, h - h % 2)
        return f"crop={w}:{h}:{x}:{y}"

    def render(self, is_cancelled: Optional[Callable[[], bool]] = None) -> bool:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "-ss", str(self.start), "-to", str(self.end), "-i", str(self.source),
            "-vf", self.crop_filter(),
            "-c:v", "libx264", "-preset", "medium", "-crf", MasterQuality(self.quality).crf,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(self.output),
        ]
        return ffmpeg.run_cancellable(args, self.output, is_cancelled)
