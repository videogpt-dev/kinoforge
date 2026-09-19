from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from kinoforge.contract import Meter, MeterAction


def extract_clips(
    video_path: Path,
    moments: List[Dict],
    output_dir: Path,
    quality: str = 'high',
    *,
    max_workers: int = 1,
    meter: Optional[Meter] = None,
) -> List[Path]:
    def extract_one(i: int, moment: Dict) -> Optional[Path]:
        clip_name = f"clip_{i:02d}_raw.mp4"
        clip_path = output_dir / clip_name
        start_time = moment['start']
        duration = moment['end'] - moment['start']

        ok = extract_clip_fast(video_path, clip_path, start_time, duration)
        if not ok:
            ok = extract_clip_reencode(video_path, clip_path, start_time, duration, quality)

        if ok and clip_path.exists():
            print(f"  extracted {clip_name}")
            return clip_path
        print(f"  failed {clip_name}")
        return None

    indexed = list(enumerate(moments, 1))
    if max_workers > 1 and len(indexed) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda im: extract_one(im[0], im[1]), indexed))
    else:
        results = [extract_one(i, m) for i, m in indexed]

    clips = [c for c in results if c is not None]
    if meter:
        meter(MeterAction.CLIP_RENDER, len(clips))
    return clips


def extract_clip_fast(
    video_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> bool:
    """Stream copy, no re-encode. Fails when the cut points don't land on keyframes."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', str(video_path),
        '-t', str(duration),
        '-c', 'copy',
        '-avoid_negative_ts', '1',
        str(output_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
        return output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def extract_clip_reencode(
    video_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    quality: str = 'high',
) -> bool:
    """Fallback when stream copy can't cut cleanly on a keyframe."""
    presets = {
        'high': {'crf': '20', 'preset': 'medium'},
        'medium': {'crf': '23', 'preset': 'fast'},
        'fast': {'crf': '28', 'preset': 'veryfast'},
    }
    settings = presets.get(quality, presets['medium'])

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', str(video_path),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', settings['preset'],
        '-crf', settings['crf'],
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        str(output_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)
        return output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if getattr(e, 'stderr', None):
            print(f"      ffmpeg: {e.stderr.decode()[:200]}")
        return False
