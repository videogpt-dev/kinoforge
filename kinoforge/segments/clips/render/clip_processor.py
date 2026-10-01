from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from kinoforge.contract import Meter, MeterAction
from kinoforge.observ import active
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg, FfmpegError
from kinoforge.segments.clips.render.media import ClipQuality

_COPY_TOLERANCE_SECONDS = 0.5


class ClipCutter:
    """Cuts one moment: stream copy when it lands on the requested span (fast), else a
    frame-accurate re-encode."""

    def __init__(self, source: Path, quality: str = ClipQuality.HIGH) -> None:
        self._source = source
        self._quality = ClipQuality(quality)

    def cut_all(
        self, moments: List[Dict], output_dir: Path, *, max_workers: int = 1,
        meter: Optional[Meter] = None,
    ) -> List[Path]:
        """Raw clip_NN_raw.mp4 per moment (in order); failed cuts are skipped and not metered."""
        output_dir.mkdir(parents=True, exist_ok=True)
        items = [(output_dir / f"clip_{i:02d}_raw.mp4", Moment(m))
                 for i, m in enumerate(moments, 1)]
        clips = [clip for clip in Ffmpeg.parallel(self.cut, items, max_workers) if clip]
        if meter:
            meter(MeterAction.CLIP_RENDER, len(clips))
        return clips

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
            Ffmpeg.run([*self._span_args(moment), "-c", "copy", "-avoid_negative_ts", "1",
                        str(output)], timeout=60)
        except FfmpegError:
            return False
        if not output.exists() or output.stat().st_size == 0:
            return False
        drift = abs(Ffmpeg.probe(output)["duration"] - moment.span)
        return drift <= _COPY_TOLERANCE_SECONDS

    def _reencode(self, output: Path, moment: Moment) -> bool:
        crf, preset = self._quality.encode
        try:
            Ffmpeg.run([*self._span_args(moment), "-c:v", "libx264", "-preset", preset,
                        "-crf", crf, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                        str(output)], timeout=120)
        except FfmpegError as exc:
            active().error(f"      ffmpeg: {exc}")
            return False
        return output.exists() and output.stat().st_size > 0

