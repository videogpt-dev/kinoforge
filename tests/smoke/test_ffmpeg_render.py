"""Real-ffmpeg smoke tests for the clips render layer and the whole clips pipeline: probe, raw
cuts (incl. the keyframe-drift fix), aspect formatting, captions, variants, the cancellable
base render, and an end-to-end ClipsPipeline run. Skipped when ffmpeg is not installed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)

from kinoforge.contract import MeterAction  # noqa: E402
from kinoforge.segments.clips.moments.moment import Moment  # noqa: E402
from kinoforge.segments.clips.pipeline import ClipsPipeline  # noqa: E402
from kinoforge.segments.clips.render.base_render import BaseCut  # noqa: E402
from kinoforge.segments.clips.render.captions import CaptionWindow  # noqa: E402
from kinoforge.segments.clips.render.clip_processor import extract_clips  # noqa: E402
from kinoforge.segments.clips.render.formatter import (  # noqa: E402
    ClipFormatter,
    VariantFormatter,
)
from kinoforge.segments.clips.render.media import FillStyle, MasterQuality  # noqa: E402
from kinoforge.segments.clips.render.probe import get_video_metadata  # noqa: E402
from kinoforge.segments.clips.run import ClipRun  # noqa: E402
from tests.support import capturing_logger  # noqa: E402

_TRANSCRIPT = [
    {"start": 0.0, "end": 2.0, "text": "hello there this is a caption"},
    {"start": 2.0, "end": 4.0, "text": "second line of speech"},
]
_SOURCE_AR = 640 / 360


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    """6s 640x360 @25fps with a sine track. libx264 ultrafast leaves a single keyframe at 0,
    so any cut starting later cannot be stream-copied exactly."""
    path = tmp_path_factory.mktemp("src") / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True, timeout=60,
    )
    return path


def _streams(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format",
         str(path)], check=True, capture_output=True, timeout=30,
    )
    data = json.loads(out.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        "width": int(video["width"]), "height": int(video["height"]),
        "duration": float(data["format"]["duration"]),
        "audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def _meter(log: list):
    return lambda action, qty, variant="": log.append((action, qty))


# --- probe ----------------------------------------------------------------

def test_probe_reads_real_metadata(source):
    info = get_video_metadata(source)
    assert (info["width"], info["height"]) == (640, 360)
    assert info["fps"] == pytest.approx(25.0)
    assert info["duration"] == pytest.approx(6.0, abs=0.2)
    assert info["aspect_ratio"] == pytest.approx(_SOURCE_AR)


def test_probe_falls_back_on_missing_file(tmp_path):
    info = get_video_metadata(tmp_path / "missing.mp4")
    assert (info["width"], info["height"], info["duration"]) == (1920, 1080, 0)


# --- raw cuts -------------------------------------------------------------

def test_extract_clips_cuts_exact_spans_even_off_keyframe(source, tmp_path):
    """3.0-5.0 starts off the only keyframe: a stream copy would carry 3s of pre-roll (5s
    clip). The cutter must detect the drift and re-encode to the exact 2s."""
    metered: list = []
    clips = extract_clips(
        source, [Moment.new(0, 2), Moment.new(3, 5)], tmp_path / "raw", "low",
        max_workers=2, meter=_meter(metered),
    )
    assert [c.name for c in clips] == ["clip_01_raw.mp4", "clip_02_raw.mp4"]
    for clip in clips:
        assert _streams(clip)["duration"] == pytest.approx(2.0, abs=0.3)
    assert metered == [(MeterAction.CLIP_RENDER, 2)]


def test_extract_clips_skips_a_failed_cut(tmp_path):
    assert extract_clips(tmp_path / "missing.mp4", [Moment.new(0, 1)], tmp_path / "raw") == []


# --- formatting -----------------------------------------------------------

def test_letterboxes_wide_source_into_portrait_with_audio(source, tmp_path):
    out = tmp_path / "portrait.mp4"
    assert ClipFormatter().format(source, out, "9:16", _SOURCE_AR)
    info = _streams(out)
    assert (info["width"], info["height"], info["audio"]) == (1080, 1920, True)


def test_blur_fill_square_and_mute(source, tmp_path):
    out = tmp_path / "square.mp4"
    assert ClipFormatter(mute=True, fill=FillStyle.BLUR).format(source, out, "1:1", _SOURCE_AR)
    info = _streams(out)
    assert (info["width"], info["height"], info["audio"]) == (1080, 1080, False)


def test_feed_ratio_with_bars(source, tmp_path):
    out = tmp_path / "feed.mp4"
    assert ClipFormatter(fill=FillStyle.BARS).format(source, out, "4:5", _SOURCE_AR)
    assert (_streams(out)["width"], _streams(out)["height"]) == (1080, 1350)


def test_burns_captions_for_the_moment_window(source, tmp_path):
    out = tmp_path / "captioned.mp4"
    window = CaptionWindow(_TRANSCRIPT, 0.0, 4.0)
    assert ClipFormatter().format(source, out, "16:9", _SOURCE_AR, window)
    ass = out.with_suffix(".ass").read_text()
    assert "PlayResY: 1080" in ass and "hello there" in ass
    assert (_streams(out)["width"], _streams(out)["height"]) == (1920, 1080)


def test_gpu_request_falls_back_to_cpu(source, tmp_path):
    out = tmp_path / "gpu.mp4"
    assert ClipFormatter(use_gpu=True).format(source, out, "1:1", _SOURCE_AR)
    assert out.exists()


def test_unknown_aspect_ratio_is_rejected(source, tmp_path):
    with pytest.raises(ValueError):
        ClipFormatter().format(source, tmp_path / "x.mp4", "4:3", _SOURCE_AR)


def test_variants_every_format_parallel_and_metered(source, tmp_path):
    moments = [Moment.new(0, 2), Moment.new(2, 4)]
    raw = extract_clips(source, moments, tmp_path / "raw")
    metered: list = []
    out = VariantFormatter(
        ["9:16", "1:1"], _TRANSCRIPT,
        rendering={"burn_subtitles": True, "mute_output": False},
        processing={"use_gpu": False, "max_workers": 2}, meter=_meter(metered),
    ).format_all(raw, moments, tmp_path / "fmt")
    assert {k: [p.name for p in v] for k, v in out.items()} == {
        "9:16": ["clip_01_9x16.mp4", "clip_02_9x16.mp4"],
        "1:1": ["clip_01_1x1.mp4", "clip_02_1x1.mp4"],
    }
    assert metered == [(MeterAction.CLIP_CAPTIONS, 2), (MeterAction.CLIP_VARIANT, 2)]


# --- base render ----------------------------------------------------------

def test_base_render_crops_and_trims(source, tmp_path):
    cut = BaseCut(source, tmp_path / "nested" / "base.mp4", 1.0, 4.0, (640, 360),
                  crop={"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0},
                  quality=MasterQuality.STANDARD)
    assert cut.crop_filter() == "crop=320:360:160:0"
    assert cut.render()
    info = _streams(cut.output)
    assert (info["width"], info["height"]) == (320, 360)
    assert info["duration"] == pytest.approx(3.0, abs=0.3)


def test_base_render_cancel_removes_partial_output(source, tmp_path):
    cut = BaseCut(source, tmp_path / "cancelled.mp4", 0.0, 6.0, (640, 360))
    assert cut.render(is_cancelled=lambda: True) is False
    assert not cut.output.exists()


# --- whole pipeline -------------------------------------------------------

def test_clips_pipeline_end_to_end(source, tmp_path):
    """Preset moments keep it deterministic: transcribe (stub) -> find -> rank -> limits ->
    project record + clip ids -> cut -> format -> place, all on real ffmpeg."""
    metered: list = []
    config = {
        "slug": "proj1", "output_dir": str(tmp_path), "clip_count": 2, "min_length": 1,
        "max_length": 10, "min_interest_score": 0, "formats": ["9:16", "16:9"],
        "quality": "medium", "generate_captions": True, "verbose": False,
        "preset_moments": [{"start": 0.5, "end": 2.5, "score": 9}, {"start": 3, "end": 5}],
        "processing": {"max_workers": 2, "use_gpu": False},
        "rendering": {"burn_subtitles": True, "mute_output": False},
    }
    run = ClipRun(
        job_id="proj1", config=config, workdir=tmp_path / "ws", logger=capturing_logger()[0],
        video_path=source, meter=_meter(metered),
    )
    ClipsPipeline(
        transcribe_video=lambda *_a, **_k: list(_TRANSCRIPT), moment_engine=None,
    ).run(run)
    assert run.status == "ok", run.error
    assert len(run.artifacts) == 2
    for artifact in run.artifacts:
        info = _streams(Path(artifact["path"]))
        assert (info["width"], info["height"]) == (1080, 1920)  # primary format placed
        assert info["duration"] == pytest.approx(2.0, abs=0.3)
    assert [m["start"] for m in run.record["moments"]] == [0.5, 3.0]
    assert run.record["width"] == 640 and run.record["fps"] == pytest.approx(25.0)
    assert not (tmp_path / "ws" / "_work").exists()  # scratch cleaned
    assert (MeterAction.CLIP_RENDER, 2) in metered
    assert (MeterAction.CLIP_VARIANT, 2) in metered


def test_extract_audio_makes_16k_mp3_and_passes_audio_through(source, tmp_path):
    from kinoforge.segments.clips.transcription.engine import extract_audio

    copy = tmp_path / "v.mp4"
    shutil.copy(source, copy)
    audio = extract_audio(copy)
    assert audio.suffix == ".mp3" and audio.exists()
    rate = json.loads(subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(audio)],
        check=True, capture_output=True).stdout)["streams"][0]["sample_rate"]
    assert rate == "16000"
    assert extract_audio(audio) == audio  # already audio: untouched
