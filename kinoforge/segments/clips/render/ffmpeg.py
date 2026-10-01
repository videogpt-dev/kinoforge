from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from kinoforge.observ import active, with_context

_T = TypeVar("_T")
_PROBE_FALLBACK = {"width": 1920, "height": 1080, "duration": 0, "aspect_ratio": 16 / 9,
                   "fps": 30.0}


class FfmpegError(RuntimeError):
    pass


class Ffmpeg:
    GRADE = "eq=contrast=1.05:saturation=1.08"

    @staticmethod
    def encode_args(use_gpu: bool) -> List[str]:
        """Video encode args: NVENC when a GPU is requested, else libx264."""
        if use_gpu:
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]

    @staticmethod
    def run(args: Sequence[str], *, timeout: float) -> None:
        """`ffmpeg -y <args>`; raises FfmpegError carrying the stderr tail on failure."""
        try:
            subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True,
                           timeout=timeout)
        except subprocess.CalledProcessError as exc:
            raise FfmpegError((exc.stderr or b"").decode(errors="replace")[-300:].strip()) from exc
        except subprocess.TimeoutExpired as exc:
            raise FfmpegError(f"ffmpeg timed out after {timeout:.0f}s") from exc
        except OSError as exc:
            raise FfmpegError(f"ffmpeg could not start: {exc}") from exc

    @staticmethod
    def run_cancellable(
        args: Sequence[str], output: Path, is_cancelled: Optional[Callable[[], bool]] = None
    ) -> bool:
        """`ffmpeg -y <args>` polling for cancellation; a cancelled run is killed and its
        partial output removed. True when ffmpeg succeeded and wrote `output`."""
        try:
            process = subprocess.Popen(
                ["ffmpeg", "-y", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError:
            return False
        while True:
            try:
                return process.wait(timeout=0.25) == 0 and output.exists()
            except subprocess.TimeoutExpired:
                if is_cancelled and is_cancelled():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    output.unlink(missing_ok=True)
                    return False

    @classmethod
    def probe(cls, path: Path) -> Dict[str, Any]:
        """Width, height, duration, aspect ratio and fps from ffprobe; 1080p/30 when unreadable."""
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
               "-show_streams", str(path)]
        try:
            data = json.loads(subprocess.run(cmd, capture_output=True, check=True,
                                             timeout=30).stdout)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            active().warning(f"    Could not read video info: {exc}")
            return dict(_PROBE_FALLBACK)
        video: Dict[str, Any] = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        width, height = int(video.get("width") or 1920), int(video.get("height") or 1080)
        return {
            "width": width,
            "height": height,
            "duration": float((data.get("format") or {}).get("duration") or 0),
            "aspect_ratio": width / max(height, 1),
            "fps": cls._fps(video.get("avg_frame_rate")) or cls._fps(video.get("r_frame_rate"))
            or 30.0,
        }

    @staticmethod
    def _fps(value: object) -> float:
        try:
            num, _, den = str(value or "").partition("/")
            return round(float(num) / float(den) if den else float(num), 3)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def parallel(fn: Callable[..., _T], items: Sequence[Tuple], max_workers: int) -> List[_T]:
        """fn(*item) for each item, results in input order, on a thread pool when there is more
        than one worker and item. Each task keeps the bound request logger."""
        if max_workers <= 1 or len(items) <= 1:
            return [fn(*item) for item in items]
        tasks = [(with_context(fn), item) for item in items]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda task: task[0](*task[1]), tasks))
