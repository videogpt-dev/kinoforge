"""Cut the chosen moments out of the source as raw clips."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from kinoforge.contract import Meter, MeterAction
from kinoforge.observ import active
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.render import ffmpeg
from kinoforge.segments.clips.render.media import ClipQuality
from kinoforge.segments.clips.render.parallel import ordered_map
from kinoforge.segments.clips.render.probe import get_video_metadata

# A stream copy can only start on a keyframe; a copied clip whose length drifts further than
# this from the requested span started early (pre-roll) and is re-encoded instead.
_COPY_TOLERANCE_SECONDS = 0.5


class ClipCutter:
    """Cuts one moment: stream copy when it lands on the requested span (fast), else a
    frame-accurate re-encode."""

    def __init__(self, source: Path, quality: ClipQuality) -> None:
        self._source = source
        self._quality = quality

    def cut(self, output: Path, moment: Moment) -> Optional[Path]:
        if self._copy(output, moment) or self._reencode(output, moment):
            active().info(f"  extracted {output.name}")
            return output
        active().warning(f"  failed {output.name}")
        return None

    def _span_args(self, moment: Moment) -> List[str]:
        return ["-ss", str(moment.start), "-i", str(self._source), "-t", str(moment.span)]

    def _copy(self, output: Path, moment: Moment) -> bool:
        try:
            ffmpeg.run([*self._span_args(moment), "-c", "copy", "-avoid_negative_ts", "1",
                        str(output)], timeout=60)
        except ffmpeg.FfmpegError:
            return False
        if not output.exists() or output.stat().st_size == 0:
            return False
        drift = abs(get_video_metadata(output)["duration"] - moment.span)
        return drift <= _COPY_TOLERANCE_SECONDS

    def _reencode(self, output: Path, moment: Moment) -> bool:
        crf, preset = self._quality.encode
        try:
            ffmpeg.run([*self._span_args(moment), "-c:v", "libx264", "-preset", preset,
                        "-crf", crf, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                        str(output)], timeout=120)
        except ffmpeg.FfmpegError as exc:
            active().error(f"      ffmpeg: {exc}")
            return False
        return output.exists() and output.stat().st_size > 0


def extract_clips(
    video_path: Path,
    moments: List[Dict],
    output_dir: Path,
    quality: str = ClipQuality.HIGH,
    *,
    max_workers: int = 1,
    meter: Optional[Meter] = None,
) -> List[Path]:
    """Raw clip_NN_raw.mp4 per moment (in order); failed cuts are skipped and not metered."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cutter = ClipCutter(video_path, ClipQuality(quality))
    items = [(output_dir / f"clip_{i:02d}_raw.mp4", Moment(m)) for i, m in enumerate(moments, 1)]
    clips = [clip for clip in ordered_map(cutter.cut, items, max_workers) if clip is not None]
    if meter:
        meter(MeterAction.CLIP_RENDER, len(clips))
    return clips
