"""
Energy Spike Detection & Viral Moment Identification
Combines audio energy analysis with keyword detection to find viral moments
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def _numpy():
    """numpy, or an ImportError that says what to install.

    It ships in requirements-ml.txt, so a slim install genuinely has none and energy
    analysis is unavailable there. Bound per call site rather than kept as a
    module-level `np = None`, which reads as a valid module everywhere below it.
    """
    try:
        import numpy
    except ImportError as e:
        raise ImportError("numpy is required for energy analysis. "
                          "Install with: pip install numpy") from e
    return numpy


@dataclass
class EnergySpike:
    """Represents an audio energy spike (potential viral moment)"""
    start: float       # seconds
    end: float         # seconds
    duration: float    # seconds
    energy_level: float  # 0-100 (normalized)
    energy_delta: float  # Change from baseline
    keywords: List[str]  # Detected keywords in this segment
    keyword_score: float  # 0-10 based on keywords
    viral_score: float   # Combined viral potential score
    confidence: float    # 0-1 confidence level


@dataclass(frozen=True)
class KeywordSet:
    """One category of viral signal: literal words, or a regex when the signal is a
    shape rather than a vocabulary (numbers)."""
    weight: float
    words: Tuple[str, ...] = ()
    pattern: str = ""


VIRAL_KEYWORDS: Dict[str, KeywordSet] = {
    'emotional': KeywordSet(0.8, (
        'amazing', 'incredible', 'shocking', 'wow', 'unbelievable', 'crazy',
        'insane', 'mind-blowing', 'genius', 'brilliant', 'stupid', 'ridiculous')),
    'action': KeywordSet(0.9, (
        'happened', 'crashed', 'exploded', 'collapsed', 'shattered', 'destroyed',
        'broke', 'failed', 'succeeded', 'won', 'lost', 'killed', 'beaten')),
    'revelation': KeywordSet(0.85, (
        'secret', 'truth', 'never knew', "didn't know", 'find out', 'discover',
        'reveal', 'exposed', 'turns out', 'actually', 'wait', 'hold on')),
    'data': KeywordSet(0.7, pattern=r'\d+(?:%|k|m|billion|million|thousand|x|times)?'),
    'hook': KeywordSet(0.75, (
        'what if', 'imagine', 'picture this', 'think about', 'consider this',
        'would you', 'could you', 'have you ever')),
}


def detect_energy_spikes(
    video_path: Path,
    segment_size: float = 0.5,  # seconds
    threshold_multiplier: float = 1.5,  # 1.5x above baseline = spike
    window_size: int = 10,  # Rolling window for baseline (in segments)
    verbose: bool = False
) -> List[EnergySpike]:
    """
    Detect audio energy spikes in video
    
    Args:
        video_path: Path to video file
        segment_size: Analyze audio in this size chunks
        threshold_multiplier: Energy x this amount above baseline = spike
        window_size: Rolling baseline window
        verbose: Print detailed output
    
    Returns:
        List of detected energy spikes sorted by energy level
    
    Example:
        >>> spikes = detect_energy_spikes(Path("video.mp4"))
        >>> for spike in spikes:
        >>>     print(f"Spike: {spike.start:.2f}s, energy: {spike.energy_level:.1f}/100")
    """

    np = _numpy()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Extract audio energy using ffmpeg
    energy_values = _extract_audio_energy(video_path, segment_size, verbose)

    if not energy_values:
        return []

    # Calculate rolling baseline (moving average)
    baseline = []
    for i in range(len(energy_values)):
        start = max(0, i - window_size // 2)
        end = min(len(energy_values), i + window_size // 2)
        baseline.append(np.mean(energy_values[start:end]))

    # Detect spikes
    spikes = []
    in_spike = False
    spike_start = 0
    spike_energy = []

    for i, energy in enumerate(energy_values):
        is_spike = energy > (baseline[i] * threshold_multiplier)

        if is_spike and not in_spike:
            # Start of new spike
            in_spike = True
            spike_start = i
            spike_energy = [energy]

        elif is_spike and in_spike:
            # Continue spike
            spike_energy.append(energy)

        elif not is_spike and in_spike:
            # End of spike
            in_spike = False

            avg_energy = np.mean(spike_energy)
            max_energy = np.max(spike_energy)

            spike = EnergySpike(
                start=spike_start * segment_size,
                end=i * segment_size,
                duration=i * segment_size - spike_start * segment_size,
                energy_level=min(100, (max_energy / np.max(energy_values)) * 100),
                energy_delta=avg_energy - baseline[spike_start],
                keywords=[],  # Will be filled later
                keyword_score=0.0,
                viral_score=0.0,
                confidence=min(1.0, avg_energy / np.max(energy_values))
            )
            spikes.append(spike)

    # Handle spike at end
    if in_spike:
        avg_energy = np.mean(spike_energy)
        max_energy = np.max(spike_energy)

        spike = EnergySpike(
            start=spike_start * segment_size,
            end=len(energy_values) * segment_size,
            duration=len(energy_values) * segment_size - spike_start * segment_size,
            energy_level=min(100, (max_energy / np.max(energy_values)) * 100),
            energy_delta=avg_energy - baseline[spike_start],
            keywords=[],
            keyword_score=0.0,
            viral_score=0.0,
            confidence=min(1.0, avg_energy / np.max(energy_values))
        )
        spikes.append(spike)

    # Sort by energy level (highest first)
    spikes.sort(key=lambda s: s.energy_level, reverse=True)

    if verbose:
        print(f"  Detected {len(spikes)} energy spikes")
        for spike in spikes[:5]:
            print(f"    {spike.start:.2f}s-{spike.end:.2f}s: {spike.energy_level:.1f}/100")

    return spikes


def _extract_audio_energy(
    video_path: Path,
    segment_size: float = 0.5,
    verbose: bool = False
) -> List[float]:
    """Extract audio energy values using ffmpeg"""

    # Use ffmpeg's volume filter to get RMS energy
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-af', f'aformat=s16:44100,volumedetect=r=10:nb_samples={int(44100 * segment_size)}',
        '-f', 'null',
        '-'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Energy extraction timed out on {video_path.name}")

    # Parse volumedetect output
    energy_values = []
    for line in result.stderr.split('\n'):
        if 'mean_volume:' in line:
            try:
                # Extract dB value and convert to linear scale
                db_value = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                # Convert dB to linear (roughly 0-100 scale)
                linear_value = max(0, min(100, (db_value + 40) * 2.5))  # -40dB to 0dB range
                energy_values.append(linear_value)
            except (IndexError, ValueError):
                continue

    if not energy_values:
        # Fallback: use simple peak detection
        energy_values = _extract_audio_energy_fallback(video_path, segment_size)

    return energy_values


def _extract_audio_energy_fallback(
    video_path: Path,
    segment_size: float
) -> List[float]:
    """Fallback energy extraction if volumedetect fails"""
    np = _numpy()

    # Extract raw audio and analyze
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-f', 's16le',
        '-'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600
        )
        audio_data = np.frombuffer(result.stdout, dtype=np.int16)
    except Exception:
        return []

    # Calculate energy in chunks
    chunk_size = int(44100 * segment_size)
    energy_values = []

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if len(chunk) > 0:
            # RMS energy (cast to float64 first: int16 ** 2 overflows -> nan)
            chunk = chunk.astype(np.float64)
            rms = np.sqrt(np.mean(chunk ** 2))
            # Normalize to 0-100
            normalized = min(100.0, float(rms) / 32768 * 100)
            energy_values.append(normalized)

    return energy_values


def detect_viral_keywords(
    transcript: List[Dict],
    moment_start: float,
    moment_end: float
) -> Tuple[List[str], float]:
    """
    Detect viral keywords in transcript segment
    
    Args:
        transcript: Full transcript with timestamps
        moment_start: Start time of moment
        moment_end: End time of moment
    
    Returns:
        Tuple of (keywords_found, keyword_score 0-10)
    """

    # Extract text from moment
    moment_text = ''
    for segment in transcript:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)

        if seg_start < moment_end and seg_end > moment_start:
            moment_text += ' ' + segment.get('text', '')

    moment_text = moment_text.strip().lower()

    if not moment_text:
        return [], 0.0

    found_keywords = []
    keyword_score = 0.0
    weights_applied = 0.0

    # Check each keyword category
    for category, config in VIRAL_KEYWORDS.items():
        if config.pattern:
            if re.search(config.pattern, moment_text):
                found_keywords.append(category)
                keyword_score += 7.0 * config.weight
                weights_applied += config.weight
        else:
            for word in config.words:
                if word in moment_text:
                    found_keywords.append(word)
                    keyword_score += 7.0 * config.weight
                    weights_applied += config.weight
                    break  # Only count once per category

    # Normalize score (0-10)
    if weights_applied > 0:
        keyword_score = min(10.0, keyword_score / weights_applied * 1.5)

    return list(set(found_keywords)), keyword_score


def combine_energy_and_keywords(
    energy_spikes: List[EnergySpike],
    transcript: List[Dict]
) -> List[EnergySpike]:
    """
    Combine energy analysis with keyword detection
    Updates spikes with keyword info and computes viral score
    
    Args:
        energy_spikes: List of detected energy spikes
        transcript: Full transcript
    
    Returns:
        Updated spikes with keywords and viral scores
    """

    updated_spikes = []

    for spike in energy_spikes:
        # Detect keywords in this spike
        keywords, keyword_score = detect_viral_keywords(
            transcript,
            spike.start,
            spike.end
        )

        spike.keywords = keywords
        spike.keyword_score = keyword_score

        # Combine energy level (0-100) and keyword score (0-10)
        # Weight: 60% energy, 40% keywords
        viral_score = (spike.energy_level / 10) * 0.6 + keyword_score * 0.4
        spike.viral_score = min(10.0, viral_score)

        updated_spikes.append(spike)

    # Sort by viral score (highest first)
    updated_spikes.sort(key=lambda s: s.viral_score, reverse=True)

    return updated_spikes


def get_top_viral_moments(
    energy_spikes: List[EnergySpike],
    count: int = 10,
    min_duration: float = 30.0,
    max_duration: float = 60.0
) -> List[EnergySpike]:
    """
    Select top viral moments by score, filtered by duration
    
    Args:
        energy_spikes: List of combined energy+keyword spikes
        count: Number of moments to return
        min_duration: Minimum moment duration
        max_duration: Maximum moment duration
    
    Returns:
        Top viral moments (up to count), sorted by viral_score
    """

    # Filter by duration
    filtered = [
        s for s in energy_spikes
        if min_duration <= s.duration <= max_duration
    ]

    # Take top by score
    return filtered[:count]
