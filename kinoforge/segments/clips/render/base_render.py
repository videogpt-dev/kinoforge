from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from kinoforge.segments.clips.render.captions import AssCaptions, CaptionWindow
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg
from kinoforge.segments.clips.render.media import MasterQuality


@dataclass(frozen=True)
class BaseCut:
    """[start, end] of `source`, cropped to a normalized box ({x, y, w, h} in 0..1, None =
    full frame), encoded at `quality` into `output`, with `captions` (transcript lines in
    source seconds) burned in when given."""

    source: Path
    output: Path
    start: float
    end: float
    source_size: Tuple[int, int]
    crop: Optional[Dict[str, float]] = None
    quality: MasterQuality = MasterQuality.HIGH
    captions: Tuple[Dict[str, Any], ...] = ()

    def crop_box(self) -> Tuple[int, int, int, int]:
        """Pixel crop (w, h, x, y), snapped to even sizes (x264 needs them) and kept inside
        the frame."""
        box = self.crop or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        width, height = self.source_size
        x, y = self._even(box["x"] * width), self._even(box["y"] * height)
        w = min(self._even(box["w"] * width), width - x)
        h = min(self._even(box["h"] * height), height - y)
        return max(2, w - w % 2), max(2, h - h % 2), x, y

    def crop_filter(self) -> str:
        w, h, x, y = self.crop_box()
        return f"crop={w}:{h}:{x}:{y}"

    def render(self, is_cancelled: Optional[Callable[[], bool]] = None) -> bool:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        video_filter = self.crop_filter()
        ass = self.output.with_suffix(".ass")
        window = CaptionWindow(list(self.captions), self.start, self.end)
        if self.captions and AssCaptions(*self.crop_box()[:2]).write(window, ass):
            video_filter = f"{video_filter},{AssCaptions.filter(ass)}"
        args = [
            "-ss", str(self.start), "-to", str(self.end), "-i", str(self.source),
            "-vf", video_filter,
            "-c:v", "libx264", "-preset", "medium", "-crf", MasterQuality(self.quality).crf,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(self.output),
        ]
        try:
            return Ffmpeg.run_cancellable(args, self.output, is_cancelled)
        finally:
            ass.unlink(missing_ok=True)

    @staticmethod
    def _even(n: float) -> int:
        return 2 * int(round(n / 2))
