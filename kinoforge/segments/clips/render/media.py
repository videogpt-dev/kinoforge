from __future__ import annotations

from enum import StrEnum
from typing import Tuple


class AspectRatio(StrEnum):
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"
    SQUARE = "1:1"
    FEED = "4:5"

    @property
    def size(self) -> Tuple[int, int]:
        return _SIZES[self]

    @property
    def slug(self) -> str:
        """File-name form, e.g. 9x16."""
        return self.value.replace(":", "x")


_SIZES = {
    AspectRatio.PORTRAIT: (1080, 1920),
    AspectRatio.LANDSCAPE: (1920, 1080),
    AspectRatio.SQUARE: (1080, 1080),
    AspectRatio.FEED: (1080, 1350),
}


class ClipQuality(StrEnum):
    """Quality of the raw clip cut (re-encode path)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def encode(self) -> Tuple[str, str]:
        """(crf, x264 preset)."""
        return _CLIP_ENCODE[self]


_CLIP_ENCODE = {
    ClipQuality.HIGH: ("20", "medium"),
    ClipQuality.MEDIUM: ("23", "fast"),
    ClipQuality.LOW: ("28", "veryfast"),
}


class MasterQuality(StrEnum):
    """Quality of the editor's base render (crop + trim master)."""

    MAX = "max"
    HIGH = "high"
    STANDARD = "standard"

    @property
    def crf(self) -> str:
        return _MASTER_CRF[self]


_MASTER_CRF = {MasterQuality.MAX: "14", MasterQuality.HIGH: "18", MasterQuality.STANDARD: "23"}


class FillStyle(StrEnum):
    """How a clip fills a frame of a different shape: blurred copy behind it, or black bars."""

    BLUR = "blur"
    BARS = "bars"
