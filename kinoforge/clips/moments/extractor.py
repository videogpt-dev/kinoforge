"""
Moment Extraction - energy spikes and viral keywords, with a transcript-window fallback.

Scoring the candidates is the AI provider's job (see app/ai/providers); this module
only finds them.
"""

import re
from pathlib import Path
from typing import Any, Dict, List


def extract_auto_moments(
        video_path: Path,
        transcript: List[Dict],
        min_length: int = 30,
        max_length: int = 60,
        target_clips: int = 10,
        verbose: bool = False
) -> List[Dict]:
    """
    Auto-generate 5-10 clips using energy spikes + viral keywords
    
    This is the primary auto-generation method that combines:
    1. Audio energy spike detection
    2. Viral keyword identification
    3. Hook detection
    4. Multi-factor scoring
    
    Args:
        video_path: Path to video file
        transcript: Full transcript with timestamps
        min_length: Minimum clip duration (seconds)
        max_length: Maximum clip duration (seconds)
        target_clips: Target number of clips (5-10 recommended)
        verbose: Print detailed analysis
    
    Returns:
        List of auto-selected moments with scores
    
    Example:
        >>> moments = extract_auto_moments(
        ...     Path("video.mp4"),
        ...     transcript,
        ...     target_clips=8
        ... )
        >>> for m in moments:
        ...     print(f"{m['start']:.1f}s-{m['end']:.1f}s: {m['score']:.1f}/10")
    """

    try:
        from kinoforge.clips.moments.energy_analyzer import (
            combine_energy_and_keywords,
            detect_energy_spikes,
            get_top_viral_moments,
        )
        from kinoforge.clips.signals.hooks import HookDetector
    except ImportError:
        if verbose:
            print("  Energy analyzer not available, using traditional extraction")
        return extract_candidate_moments(transcript, min_length, max_length)

    if verbose:
        print(f"  Auto-generating {target_clips} clips using energy + keywords...")

    try:
        # Step 1: Detect energy spikes
        if verbose:
            print("  Step 1: Analyzing audio energy...")

        energy_spikes = detect_energy_spikes(
            video_path,
            segment_size=0.5,
            threshold_multiplier=1.5,
            verbose=verbose
        )

        if not energy_spikes:
            if verbose:
                print("  No energy spikes detected, falling back to traditional extraction")
            return extract_candidate_moments(transcript, min_length, max_length)

        # Step 2: Combine with keyword detection
        if verbose:
            print("  Step 2: Detecting viral keywords...")

        combined_spikes = combine_energy_and_keywords(energy_spikes, transcript)

        # Step 3: Get top viral moments by score
        if verbose:
            print(f"  Step 3: Selecting top {target_clips} moments...")

        viral_moments = get_top_viral_moments(
            combined_spikes,
            count=target_clips,
            min_duration=min_length,
            max_duration=max_length
        )

        # Step 4: Enhance with hook detection and scoring
        final_moments: List[Dict[str, Any]] = []

        for spike in viral_moments:
            # Analyze hook strength
            hook_signal = HookDetector().analyze(transcript, spike.start, spike.end)

            # Get full text
            text = get_text_between_times(transcript, spike.start, spike.end)

            # Combine scores: 50% energy+keywords, 30% hook, 20% baseline
            combined_score = (
                spike.viral_score * 0.5 +
                (hook_signal.strength / 10) * 3 * 0.3 +
                7.0 * 0.2  # Baseline score
            )

            moment = {
                'start': spike.start,
                'end': spike.end,
                'duration': spike.duration,
                'text': text,
                'score': min(10.0, combined_score),
                'energy_level': spike.energy_level,
                'viral_keywords': spike.keywords,
                'hook_type': hook_signal.hook_type,
                'hook_strength': hook_signal.strength,
                'reason': f"Energy: {spike.energy_level:.0f}/100, Keywords: {', '.join(spike.keywords or ['none'])}",
                'language': detect_language(text),
                'source': 'energy_analysis'
            }
            final_moments.append(moment)

        # Sort by score (highest first)
        final_moments.sort(key=lambda m: m['score'], reverse=True)

        # If the energy path produced nothing usable (e.g. spikes too short to
        # meet the duration window), fall back to the transcript sliding window.
        if not final_moments:
            if verbose:
                print("  Energy path yielded no valid-length moments, "
                      "falling back to transcript extraction")
            return extract_candidate_moments(transcript, min_length, max_length)

        if verbose:
            print(f"  Generated {len(final_moments)} moments")
            for i, m in enumerate(final_moments, 1):
                print(f"    {i}. {m['start']:.1f}s-{m['end']:.1f}s (score: {m['score']:.1f}/10)")

        return final_moments

    except Exception as e:
        if verbose:
            print(f"  Auto-generation failed: {e}")
            print("  Falling back to traditional extraction...")
        return extract_candidate_moments(transcript, min_length, max_length)


def extract_candidate_moments(
        transcript: List[Dict],
        min_length: int = 30,
        max_length: int = 60
) -> List[Dict]:
    """
    Traditional rule-based moment extraction (no AI)
    Used as fallback when AI extraction fails.

    Produces NON-OVERLAPPING windows that span the whole transcript, each
    between min_length and max_length seconds. This gives clean, well-spread
    clips instead of hundreds of near-duplicate overlapping windows.
    """
    candidates = []
    n = len(transcript)
    i = 0

    while i < n:
        start_time = transcript[i]['start']
        text_parts = []
        end_time = start_time
        j = i

        # Grow the window until it reaches at least min_length, without
        # exceeding max_length.
        while j < n:
            seg = transcript[j]
            if seg['end'] - start_time > max_length:
                break
            text_parts.append(seg['text'])
            end_time = seg['end']
            j += 1
            if end_time - start_time >= min_length:
                break

        duration = end_time - start_time
        text = ' '.join(text_parts).strip()

        # Keep only valid-length windows with some actual content
        if min_length <= duration <= max_length and len(text.split()) >= 5:
            candidates.append({
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'text': text,
                'language': detect_language(text)
            })
            i = j  # advance past this window (non-overlapping)
        else:
            i += 1

    return candidates


def snap_to_transcript(
    transcript: List[Dict], start: float, end: float,
    min_len: float = 0.0, max_len: float = 0.0
) -> tuple:
    """Snap a rough [start, end] onto whole-segment boundaries, then size it into the
    [min_len, max_len] window: extend a short pick (forward first, then back) and trim
    a long one from the tail. Returns (start, end, text).

    Boundaries always land on a segment edge, so a clip opens and closes on a complete
    sentence instead of the arbitrary second an LLM happened to name.
    """
    segs = sorted((s for s in transcript if s.get("text")),
                  key=lambda s: float(s.get("start") or 0))
    n = len(segs)
    if n == 0:
        return start, end, ""

    def s_start(i: int) -> float:
        return float(segs[i].get("start") or 0)

    def s_end(i: int) -> float:
        return float(segs[i].get("end") or 0)

    # First segment at/before the requested start; first segment ending at/after the end.
    si = 0
    for i in range(n):
        if s_start(i) <= start:
            si = i
        else:
            break
    ei = si
    for i in range(si, n):
        if s_end(i) >= end:
            ei = i
            break
    else:
        ei = n - 1
    if ei < si:
        ei = si

    if min_len > 0:
        while (s_end(ei) - s_start(si)) < min_len and (ei < n - 1 or si > 0):
            if ei < n - 1:
                ei += 1
            elif si > 0:
                si -= 1
            else:
                break
    if max_len > 0:
        while (s_end(ei) - s_start(si)) > max_len and ei > si:
            ei -= 1

    text = " ".join((segs[i].get("text") or "").strip() for i in range(si, ei + 1)).strip()
    return s_start(si), s_end(ei), text


def get_text_between_times(transcript: List[Dict], start: float, end: float) -> str:
    """Extract text between timestamps"""
    text_parts = []
    for segment in transcript:
        if segment['start'] >= start and segment['end'] <= end:
            text_parts.append(segment['text'])
    return ' '.join(text_parts).strip()


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def detect_language(text: str) -> str:
    """Simple language detection"""
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    elif re.search(r'[\u4e00-\u9fff]', text):
        return 'chinese'
    elif re.search(r'[\u0600-\u06FF]', text):
        return 'arabic'
    return 'english'
