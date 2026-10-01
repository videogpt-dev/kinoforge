"""Render-layer units that need no ffmpeg: enums, fit-filter choice, captions, crop math,
parallel ordering, the execution-lease context manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from kinoforge.segments.clips.render.base_render import BaseCut
from kinoforge.segments.clips.render.captions import AssCaptions, CaptionWindow
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg
from kinoforge.segments.clips.render.formatter import ClipFormatter
from kinoforge.segments.clips.render.media import (
    AspectRatio,
    ClipQuality,
    FillStyle,
    MasterQuality,
)
from kinoforge.service.executions import ExecutionConflict, ExecutionRegistry


def test_aspect_ratio_sizes_and_slugs():
    assert AspectRatio("9:16").size == (1080, 1920)
    assert AspectRatio.FEED.size == (1080, 1350)
    assert AspectRatio.LANDSCAPE.slug == "16x9"
    with pytest.raises(ValueError):
        AspectRatio("4:3")


def test_low_quality_is_really_low():
    # Regression: "low" used to fall through to the medium preset.
    assert ClipQuality("low").encode == ("28", "veryfast")
    assert ClipQuality.HIGH.encode == ("20", "medium")
    assert MasterQuality("max").crf == "14"


@pytest.mark.parametrize("source_ar, fill, marker", [
    (16 / 9, None, "force_original_aspect_ratio=decrease,pad"),  # wider -> letterbox
    (9 / 21, None, "boxblur"),                                    # taller -> blur pad
    (1080 / 1920, None, "scale=1080:1920,eq"),                    # same shape -> scale
    (9 / 16, FillStyle.BARS, "pad="),                             # forced bars
    (16 / 9, FillStyle.BLUR, "boxblur"),                          # forced blur
])
def test_fit_filter_choice(source_ar, fill, marker):
    assert marker in ClipFormatter.fit_filter((1080, 1920), source_ar, fill)


def test_captions_clip_relative_and_skip_outside_window(tmp_path):
    path = tmp_path / "c.ass"
    window = CaptionWindow(
        [{"start": 9, "end": 11, "text": "inside"}, {"start": 20, "end": 21, "text": "out"}],
        10.0, 15.0,
    )
    assert AssCaptions(1080, 1920).write(window, path)
    text = path.read_text()
    assert "PlayResX: 1080" in text and "0:00:00.00" in text
    assert "inside" in text and "out\n" not in text
    assert AssCaptions(1080, 1920).write(CaptionWindow([], 0, 5), tmp_path / "n.ass") is False


def test_caption_helpers():
    assert AssCaptions.timestamp(3725.5) == "1:02:05.50"
    assert AssCaptions.escape("a{b}\n") == "a(b)\\N"
    assert all(len(c) <= 42 for c in AssCaptions.chunks("word " * 30))


def test_base_cut_crop_stays_even_and_inside_frame():
    cut = BaseCut(Path("s"), Path("o"), 0, 1, (641, 361), crop={"x": .9, "y": .9, "w": .5, "h": .5})
    w, h, x, y = map(int, cut.crop_filter().removeprefix("crop=").split(":"))
    assert w % 2 == 0 and h % 2 == 0 and x + w <= 641 and y + h <= 361
    assert BaseCut(Path("s"), Path("o"), 0, 1, (640, 360)).crop_filter() == "crop=640:360:0:0"


def test_parallel_keeps_input_order_in_parallel():
    items = [(n,) for n in range(20)]
    assert Ffmpeg.parallel(lambda n: n * n, items, max_workers=4) == [n * n for n in range(20)]
    assert Ffmpeg.parallel(lambda n: n, [(1,)], max_workers=4) == [1]


def test_running_holds_the_lease_and_releases_it():
    registry = ExecutionRegistry()
    with registry.running("job") as control:
        assert control is not None and not control.is_cancelled()
        with pytest.raises(ExecutionConflict):
            with registry.running("job"):
                pass
        registry.cancel("job")
        assert control.is_cancelled()
    with registry.running("job"):  # released: can run again
        pass
    with registry.running(None) as untracked:
        assert untracked is None
