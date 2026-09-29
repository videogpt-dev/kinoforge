"""The one place clips rendering shells out to ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# Mild contrast/saturation lift applied to every formatted output.
GRADE = "eq=contrast=1.05:saturation=1.08"


class FfmpegError(RuntimeError):
    pass


def encode_args(use_gpu: bool) -> List[str]:
    """Video encode args: NVENC when a GPU is requested, else libx264."""
    if use_gpu:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]


def run(args: Sequence[str], *, timeout: float) -> None:
    """Run `ffmpeg -y <args>`; raise FfmpegError carrying the stderr tail on failure."""
    try:
        subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise FfmpegError((exc.stderr or b"").decode(errors="replace")[-300:].strip()) from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"ffmpeg timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise FfmpegError(f"ffmpeg could not start: {exc}") from exc


def run_cancellable(
    args: Sequence[str], output: Path, is_cancelled: Optional[Callable[[], bool]] = None
) -> bool:
    """Run `ffmpeg -y <args>` polling for cancellation; a cancelled run is killed and its
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
                _stop(process)
                output.unlink(missing_ok=True)
                return False


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
