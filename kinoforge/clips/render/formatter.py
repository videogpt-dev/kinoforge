"""
Enhanced formatter.py - FIXED VERSION
Multiple aspect ratios and advanced features with all errors corrected
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from kinoforge.contract import Meter, MeterAction


def _encode_args(use_gpu: bool) -> List[str]:
    """ffmpeg video-encode args: NVENC when GPU is requested, else libx264."""
    if use_gpu:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]


def _write_window_srt(transcript: List[Dict], start: float, end: float, srt_path: Path) -> bool:
    """Write a clip-relative SRT for transcript segments overlapping [start, end].
    Returns False (no file written) when the window has no caption text."""
    blocks: List[str] = []
    for seg in transcript or []:
        s, e = float(seg.get("start", 0) or 0), float(seg.get("end", 0) or 0)
        if e <= start or s >= end:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        rel_start = max(0.0, s - start)
        rel_end = max(rel_start + 0.1, min(e, end) - start)
        n = len(blocks) + 1
        blocks.append(
            f"{n}\n{format_srt_timestamp(rel_start)} --> {format_srt_timestamp(rel_end)}\n{text}\n"
        )
    if not blocks:
        return False
    srt_path.write_text("\n".join(blocks), encoding="utf-8")
    return True


def format_clips_multi_platform(
    clip_paths: List[Path],
    moments: List[Dict],
    output_dir: Path,
    formats: List[str] | None = None,
    transcript: Optional[List[Dict]] = None,
    *,
    rendering: Dict[str, Any],
    processing: Dict[str, Any],
    meter: Optional[Meter] = None,
) -> Dict[str, List[Path]]:
    """Format clips for multiple platforms with different aspect ratios.

    Honours the injected render/processing settings: burn_subtitles +
    subtitle_font_size (burn the transcript window as captions), mute_output (drop
    audio), use_gpu (NVENC), max_workers (parallel clips).
    """
    if formats is None:
        formats = ["9:16", "16:9"]

    burn = rendering["burn_subtitles"]
    mute = rendering["mute_output"]
    use_gpu = processing["use_gpu"]
    font_size = rendering["subtitle_font_size"]
    max_workers = processing["max_workers"]

    formatted_clips: Dict[str, List] = {fmt: [] for fmt in formats}

    def process_one(i: int, clip_path: Path, moment: Dict) -> List[tuple]:
        print(f"\n  Processing clip {i}/{len(clip_paths)}...")
        video_info = get_video_metadata(clip_path)

        srt_path = None
        if burn and transcript:
            candidate = output_dir / f"clip_{i:02d}.srt"
            if _write_window_srt(transcript, moment["start"], moment["end"], candidate):
                srt_path = candidate

        produced = []
        for aspect_ratio in formats:
            output_name = f"clip_{i:02d}_{aspect_ratio.replace(':', 'x')}.mp4"
            output_path = output_dir / output_name
            ok = apply_format_with_aspect_ratio(
                clip_path, output_path, srt_path, aspect_ratio, video_info, moment,
                font_size=font_size, mute=mute, use_gpu=use_gpu,
            )
            print(f"    {'✓' if ok else '✗'} {aspect_ratio}: {output_name}")
            produced.append((aspect_ratio, output_path, ok))
        return produced

    pairs = list(enumerate(zip(clip_paths, moments), 1))
    if max_workers > 1 and len(pairs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            batches = list(pool.map(lambda p: process_one(p[0], p[1][0], p[1][1]), pairs))
    else:
        batches = [process_one(i, cp, m) for i, (cp, m) in pairs]

    captioned = 0
    for produced in batches:
        for aspect_ratio, output_path, ok in produced:
            if ok:
                formatted_clips[aspect_ratio].append(output_path)
        if burn and transcript and any(ok for _, _, ok in produced):
            captioned += 1

    # Billed on what was produced. Every clip is rendered in one format as part of the
    # export; each additional aspect ratio is a second encode, so it is its own line.
    rendered = sum(len(paths) for paths in formatted_clips.values())
    if meter:
        meter(MeterAction.CLIP_CAPTIONS, captioned)
        meter(MeterAction.CLIP_VARIANT, max(0, rendered - len(clip_paths)))

    return formatted_clips


def get_video_metadata(video_path: Path) -> Dict:
    """
    Get detailed video metadata using ffprobe
    ✓ FIXED: Safe FPS parsing, added timeout
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]

    try:
        # ✓ FIXED: Added timeout
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        data = json.loads(result.stdout)

        video_stream: Dict[str, Any] = next(
            (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
            {}
        )

        # ✓ FIXED: Safe FPS parsing without eval()
        fps = 30.0
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if fps_str and '/' in fps_str:
            try:
                num, den = fps_str.split('/')
                if float(den) != 0:
                    fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps = 30
        elif fps_str:
            try:
                fps = float(fps_str)
            except ValueError:
                fps = 30

        return {
            'width': video_stream.get('width', 1920),
            'height': video_stream.get('height', 1080),
            'duration': float(data.get('format', {}).get('duration', 0)),
            'aspect_ratio': int(video_stream.get('width', 16)) / max(int(video_stream.get('height', 9)), 1),
            'fps': fps
        }
    except Exception as e:
        print(f"    Warning: Could not get video info: {e}")
        return {
            'width': 1920,
            'height': 1080,
            'duration': 0,
            'aspect_ratio': 16/9,
            'fps': 30
        }


def apply_format_with_aspect_ratio(
    input_path: Path,
    output_path: Path,
    srt_path: Optional[Path],
    aspect_ratio: str,
    video_info: Dict,
    moment: Dict,
    font_size: int = 48,
    mute: bool = False,
    use_gpu: bool = False,
    fill: Optional[str] = None,
) -> bool:
    """Format a clip to an aspect ratio (letterbox/pad, never crop). When srt_path is
    given the captions are burned in; mute drops audio; use_gpu encodes via NVENC.
    `fill` forces the fit style regardless of source orientation: "blur" (blurred
    background) or "bars" (solid black letterbox); the default picks per source."""
    dimensions = {
        "9:16": (1080, 1920),
        "16:9": (1920, 1080),
        "1:1": (1080, 1080),
        "4:5": (1080, 1350)
    }

    target_width, target_height = dimensions[aspect_ratio]
    source_ar = video_info['aspect_ratio']
    target_ar = target_width / target_height

    if fill == "blur":
        video_filter = build_pad_filter_clean(target_width, target_height, aspect_ratio, video_info)
    elif fill == "bars":
        video_filter = build_letterbox_filter(target_width, target_height, aspect_ratio, video_info)
    elif abs(source_ar - target_ar) < 0.01:
        # Already correct aspect ratio - just scale
        video_filter = build_scale_filter_clean(target_width, target_height, aspect_ratio)
    elif source_ar > target_ar:
        # Source wider than target (e.g., 16:9 to 9:16) - use LETTERBOX, never crop
        video_filter = build_letterbox_filter(target_width, target_height, aspect_ratio, video_info)
    else:
        # Source taller than target - use PAD with blurred background
        video_filter = build_pad_filter_clean(target_width, target_height, aspect_ratio, video_info)

    if srt_path is not None:
        # Escape for the subtitles filter (':' and "'" are filtergraph separators).
        esc = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        video_filter = f"{video_filter},subtitles='{esc}':force_style='FontSize={font_size}'"

    audio_args = ["-an"] if mute else [
        "-c:a", "aac", "-b:a", "128k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
    ]

    def _run(gpu: bool) -> bool:
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vf", video_filter]
        cmd += _encode_args(gpu) + audio_args + [str(output_path)]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=300)
        return output_path.exists()

    try:
        return _run(use_gpu)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as first:
        # Bound here, not only in the fallback below: a CPU-path failure used to reach
        # the stderr print with `e` unbound, raising NameError over the real ffmpeg error.
        e: Exception = first
        if use_gpu:
            # NVENC unavailable (e.g. no GPU) — fall back to libx264.
            print("      ⚠️  GPU encode failed, falling back to libx264...")
            try:
                return _run(False)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                e = e2
        if hasattr(e, 'stderr') and e.stderr:
            print(f"      Error: {e.stderr.decode()[:200]}")
        return False


def build_scale_filter_clean(width: int, height: int, aspect_ratio: str) -> str:
    """Simple scale with subtle enhancement - NO CAPTIONS"""
    return (
        f"scale={width}:{height},"
        f"eq=contrast=1.05:saturation=1.08"
    )


def build_letterbox_filter(width: int, height: int, aspect_ratio: str, video_info: Dict) -> str:
    """Scale then add black letterbox bars - PRESERVES ALL CONTENT"""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"eq=contrast=1.05:saturation=1.08"
    )


def build_pad_filter_clean(width: int, height: int, aspect_ratio: str, video_info: Dict) -> str:
    """Scale then add blurred background padding - ELEGANT, NO CROP.

    A single-input simple filtergraph: `split` the source into a scaled-to-fit
    foreground and a blurred fill, then overlay. (Referencing `[0:v]` twice would
    make it a complex graph that -vf rejects — split keeps it -vf compatible.)"""
    return (
        f"split=2[main][blur];"
        f"[blur]scale={width}:{height},boxblur=20:1[bg];"
        f"[main]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"eq=contrast=1.05:saturation=1.08"
    )


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    # ✓ FIXED: Comma not period for SRT format
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def add_intro_outro(
    clip_path: Path,
    output_path: Path,
    intro_path: Path | None = None,
    outro_path: Path | None = None
) -> bool:
    """Add intro and/or outro to clip"""
    if not intro_path and not outro_path:
        return False

    concat_file = clip_path.parent / f"concat_{clip_path.stem}.txt"

    # ✓ FIXED: Added encoding
    with open(concat_file, 'w', encoding='utf-8') as f:
        if intro_path:
            f.write(f"file '{intro_path}'\n")
        f.write(f"file '{clip_path}'\n")
        if outro_path:
            f.write(f"file '{outro_path}'\n")

    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',
        str(output_path)
    ]

    try:
        # ✓ FIXED: Added timeout
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, timeout=300)
        concat_file.unlink()
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if concat_file.exists():
            concat_file.unlink()
        return False


def add_zoom_effect(
    input_path: Path,
    output_path: Path,
    zoom_factor: float = 1.1
) -> bool:
    """Add subtle zoom effect for engagement"""
    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_path),
        '-vf', f"zoompan=z='min(zoom+0.0015,{zoom_factor})':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920",
        '-c:a', 'copy',
        str(output_path)
    ]

    try:
        # ✓ FIXED: Added timeout
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, timeout=300)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def add_progress_bar(
    input_path: Path,
    output_path: Path,
    bar_height: int = 8,
    bar_color: str = "white"
) -> bool:
    """Add progress bar at top of video"""
    color_map = {
        'white': '0xFFFFFF',
        'red': '0xFF0000',
        'blue': '0x0000FF',
        'green': '0x00FF00'
    }

    color_hex = color_map.get(bar_color, '0xFFFFFF')

    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_path),
        '-vf', f"drawbox=x=0:y=0:w='iw*t/duration':h={bar_height}:color={color_hex}:t=fill",
        '-c:a', 'copy',
        str(output_path)
    ]

    try:
        # ✓ FIXED: Added timeout
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, timeout=300)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
