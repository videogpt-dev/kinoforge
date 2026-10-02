from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kinoforge.contract import Meter, MeterAction
from kinoforge.observ import active
from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.render.captions import AssCaptions, CaptionWindow
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg, FfmpegError
from kinoforge.segments.clips.render.media import AspectRatio, FillStyle

_AUDIO = ["-c:a", "aac", "-b:a", "128k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]


@dataclass(frozen=True)
class ClipFormatter:
    """Formats one clip to one aspect ratio."""

    mute: bool = False
    use_gpu: bool = False
    fill: Optional[FillStyle] = None

    @classmethod
    def fit_filter(cls, size: Tuple[int, int], source_ar: float,
                   fill: Optional[FillStyle]) -> str:
        """A forced fill wins; else same shape scales, wider letterboxes, taller gets a blur pad."""
        width, height = size
        if fill is FillStyle.BLUR:
            return cls._blurred_pad(width, height)
        if fill is FillStyle.BARS:
            return cls._letterbox(width, height)
        target_ar = width / height
        if abs(source_ar - target_ar) < 0.01:
            return f"scale={width}:{height},{Ffmpeg.GRADE}"
        if source_ar > target_ar:
            return cls._letterbox(width, height)
        return cls._blurred_pad(width, height)

    @staticmethod
    def _letterbox(width: int, height: int) -> str:
        return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,{Ffmpeg.GRADE}")

    @staticmethod
    def _blurred_pad(width: int, height: int) -> str:
        return (f"split=2[main][blur];[blur]scale={width}:{height},boxblur=20:1[bg];"
                f"[main]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{Ffmpeg.GRADE}")

    def format(
        self, source: Path, output: Path, aspect: str, source_ar: float,
        captions: Optional[CaptionWindow] = None,
    ) -> bool:
        size = AspectRatio(aspect).size
        video_filter = self.fit_filter(size, source_ar, self.fill)
        if captions is not None:
            ass_path = output.with_suffix(".ass")
            if AssCaptions(*size).write(captions, ass_path):
                escaped = (str(ass_path).replace("\\", "\\\\").replace(":", "\\:")
                           .replace("'", "\\'"))
                video_filter = f"{video_filter},ass='{escaped}'"
        return self._encode(source, output, video_filter)

    def _encode(self, source: Path, output: Path, video_filter: str) -> bool:
        audio = ["-an"] if self.mute else _AUDIO
        attempts = [True, False] if self.use_gpu else [False]
        for gpu in attempts:
            try:
                Ffmpeg.run(["-i", str(source), "-vf", video_filter, *Ffmpeg.encode_args(gpu),
                            *audio, str(output)], timeout=300)
                return output.exists()
            except FfmpegError as exc:
                if gpu:
                    active().warning("      GPU encode failed, falling back to libx264")
                else:
                    active().error(f"      ffmpeg: {exc}")
        return False


class VariantFormatter:
    """Formats every raw clip into every requested aspect ratio (parallel across clips), then
    meters captions and the extra encodes beyond the first format."""

    def __init__(
        self,
        formats: Sequence[str],
        transcript: Optional[List[Dict[str, Any]]],
        *,
        rendering: Dict[str, Any],
        processing: Dict[str, Any],
        meter: Optional[Meter] = None,
    ) -> None:
        self._formats = [AspectRatio(f) for f in formats]
        self._burn = bool(rendering["burn_subtitles"]) and bool(transcript)
        self._transcript = transcript or []
        self._formatter = ClipFormatter(
            mute=bool(rendering["mute_output"]), use_gpu=bool(processing["use_gpu"])
        )
        self._workers = int(processing["max_workers"])
        self._meter = meter

    def format_all(
        self, clip_paths: List[Optional[Path]], moments: List[Dict[str, Any]], output_dir: Path
    ) -> Dict[str, List[Path]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        items = [(i, clip, moment, output_dir)
                 for i, (clip, moment) in enumerate(zip(clip_paths, moments), 1) if clip]
        batches = Ffmpeg.parallel(self._format_one, items, self._workers)
        formatted: Dict[str, List[Path]] = {str(f): [] for f in self._formats}
        for produced in batches:
            for aspect, path, ok in produced:
                if ok:
                    formatted[str(aspect)].append(path)
        self._bill(batches, formatted, len(items))
        return formatted

    def _format_one(
        self, index: int, clip: Path, moment: Dict[str, Any], output_dir: Path
    ) -> List[Tuple[AspectRatio, Path, bool]]:
        active().info(f"  Processing clip {index}...")
        source_ar = Ffmpeg.probe(clip)["aspect_ratio"]
        view = Moment(moment)
        captions = (CaptionWindow(self._transcript, view.start, view.end)
                    if self._burn else None)
        produced = []
        for aspect in self._formats:
            output = output_dir / f"clip_{index:02d}_{aspect.slug}.mp4"
            ok = self._formatter.format(clip, output, aspect, source_ar, captions)
            active().info(f"    {'ok' if ok else 'failed'} {aspect}: {output.name}")
            produced.append((aspect, output, ok))
        return produced

    def _bill(self, batches: List[List[Tuple]], formatted: Dict[str, List[Path]],
              clips: int) -> None:
        if not self._meter:
            return
        captioned = sum(1 for produced in batches if self._burn and any(ok for *_, ok in produced))
        rendered = sum(len(paths) for paths in formatted.values())
        self._meter(MeterAction.CLIP_CAPTIONS, captioned)
        self._meter(MeterAction.CLIP_VARIANT, max(0, rendered - clips))
