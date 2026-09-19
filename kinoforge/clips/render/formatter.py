"""Format clips to platform aspect ratios, letterbox or blurred pad, optional burned captions."""

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


def _ass_timestamp(seconds: float) -> str:
    """ASS timestamp: H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    """Escape caption text for an ASS Dialogue line."""
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def _chunk_caption(text: str, max_chars: int = 42) -> List[str]:
    """Break caption text into <= max_chars chunks on word boundaries (about two lines)."""
    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def _write_window_ass(
    transcript: List[Dict], start: float, end: float, ass_path: Path, width: int, height: int
) -> bool:
    """Write a clip-relative ASS for transcript segments overlapping [start, end].

    PlayResX/Y are pinned to the output dimensions so Fontsize is real pixels (an SRT
    with force_style scales against libass' 288px default, blowing captions up ~6x).
    Font size, margins and outline scale with height; captions sit bottom-centre.
    Returns False when the window has no caption text."""
    font_size = max(24, round(height * 0.040))
    outline = max(2, round(height * 0.003))
    margin_v = round(height * 0.10)
    margin_h = round(width * 0.07)
    events: List[str] = []
    for seg in transcript or []:
        s, e = float(seg.get("start", 0) or 0), float(seg.get("end", 0) or 0)
        if e <= start or s >= end:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        rel_start = max(0.0, s - start)
        rel_end = max(rel_start + 0.1, min(e, end) - start)
        # Split a long segment into short chunks (~2 lines) so captions never blanket the
        # frame; each chunk gets a time slice proportional to its length.
        chunks = _chunk_caption(text)
        total = sum(len(c) for c in chunks) or 1
        cursor = rel_start
        for chunk in chunks:
            span = (rel_end - rel_start) * (len(chunk) / total)
            c_end = min(rel_end, cursor + max(0.4, span))
            events.append(
                f"Dialogue: 0,{_ass_timestamp(cursor)},{_ass_timestamp(c_end)},"
                f"Default,,0,0,0,,{_ass_escape(chunk)}"
            )
            cursor = c_end
    if not events:
        return False
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},1,2,{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
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
    meter: Optional[Meter] = None) -> Dict[str, List[Path]]:
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
    max_workers = processing["max_workers"]

    formatted_clips: Dict[str, List] = {fmt: [] for fmt in formats}

    def process_one(i: int, clip_path: Path, moment: Dict) -> List[tuple]:
        print(f"\n  Processing clip {i}/{len(clip_paths)}...")
        video_info = get_video_metadata(clip_path)

        produced = []
        for aspect_ratio in formats:
            output_name = f"clip_{i:02d}_{aspect_ratio.replace(':', 'x')}.mp4"
            output_path = output_dir / output_name
            ok = apply_format_with_aspect_ratio(
                clip_path, output_path, aspect_ratio, video_info, moment,
                transcript=transcript if burn else None, mute=mute, use_gpu=use_gpu)
            print(f"    {'ok' if ok else 'failed'} {aspect_ratio}: {output_name}")
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
    """Width, height, duration, aspect ratio and fps from ffprobe. Falls back to 1080p/30 on error."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        data = json.loads(result.stdout)

        video_stream: Dict[str, Any] = next(
            (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
            {}
        )

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
    aspect_ratio: str,
    video_info: Dict,
    moment: Dict,
    transcript: Optional[List[Dict]] = None,
    mute: bool = False,
    use_gpu: bool = False,
    fill: Optional[str] = None) -> bool:
    """Format a clip to an aspect ratio (letterbox/pad, never crop). When a transcript is
    given the moment window is burned in as captions sized to the output; mute drops
    audio; use_gpu encodes via NVENC. `fill` forces the fit style regardless of source
    orientation: "blur" (blurred background) or "bars" (solid black letterbox); the
    default picks per source."""
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
        video_filter = build_scale_filter_clean(target_width, target_height, aspect_ratio)
    elif source_ar > target_ar:
        # Wider than target: letterbox, never crop.
        video_filter = build_letterbox_filter(target_width, target_height, aspect_ratio, video_info)
    else:
        # Taller than target: pad with a blurred background.
        video_filter = build_pad_filter_clean(target_width, target_height, aspect_ratio, video_info)

    if transcript:
        ass_path = output_path.with_suffix(".ass")
        if _write_window_ass(
            transcript, moment.get("start", 0), moment.get("end", 0),
            ass_path, target_width, target_height):
            # Escape for the ass filter (':' and "'" are filtergraph separators).
            esc = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            video_filter = f"{video_filter},ass='{esc}'"

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
            # NVENC unavailable (e.g. no GPU): fall back to libx264.
            print("      GPU encode failed, falling back to libx264")
            try:
                return _run(False)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                e = e2
        if hasattr(e, 'stderr') and e.stderr:
            print(f"      Error: {e.stderr.decode()[:200]}")
        return False


def build_scale_filter_clean(width: int, height: int, aspect_ratio: str) -> str:
    return (
        f"scale={width}:{height},"
        f"eq=contrast=1.05:saturation=1.08"
    )


def build_letterbox_filter(width: int, height: int, aspect_ratio: str, video_info: Dict) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"eq=contrast=1.05:saturation=1.08"
    )


def build_pad_filter_clean(width: int, height: int, aspect_ratio: str, video_info: Dict) -> str:
    """Scale to fit over a blurred copy of itself, no crop.

    A single-input simple filtergraph: `split` the source into a scaled-to-fit
    foreground and a blurred fill, then overlay. (Referencing `[0:v]` twice would
    make it a complex graph that -vf rejects, split keeps it -vf compatible.)"""
    return (
        f"split=2[main][blur];"
        f"[blur]scale={width}:{height},boxblur=20:1[bg];"
        f"[main]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"eq=contrast=1.05:saturation=1.08"
    )
