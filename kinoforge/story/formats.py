"""Supported generated-video canvases. Stored once on project input."""

from enum import StrEnum
from typing import Dict


class VideoAspectRatio(StrEnum):
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"


DIMENSIONS = {
    VideoAspectRatio.PORTRAIT: (1080, 1920),
    VideoAspectRatio.LANDSCAPE: (1920, 1080),
}


def normalize(value: object) -> str:
    """Persist only supported aspect ratios; legacy projects remain portrait."""
    try:
        return VideoAspectRatio(str(value or VideoAspectRatio.PORTRAIT)).value
    except ValueError:
        return VideoAspectRatio.PORTRAIT.value


def project_aspect(record: Dict) -> str:
    return normalize((record.get("input") or {}).get("aspect_ratio"))


def dimensions(value: object) -> tuple[int, int]:
    return DIMENSIONS[VideoAspectRatio(normalize(value))]


def prompt_label(value: object) -> str:
    return (
        "widescreen 16:9 standard video"
        if normalize(value) == VideoAspectRatio.LANDSCAPE
        else "vertical 9:16 short-form video"
    )
