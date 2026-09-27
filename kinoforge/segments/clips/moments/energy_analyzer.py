"""Energy-spike detection and viral-moment identification.

Combines audio energy analysis (EnergyAnalyzer) with keyword detection
(ViralKeywordDetector), then ranks/selects candidates (ViralRanker).
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from kinoforge.observ import active


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


VIRAL_KEYWORDS = {
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


def _numpy():
    """numpy, or an ImportError that says what to install.

    It ships in requirements-ml.txt, so a slim install genuinely has none and energy
    analysis is unavailable there. Bound per call rather than kept as a module-level
    `np = None`, which reads as a valid module everywhere below it."""
    try:
        import numpy
    except ImportError as e:
        raise ImportError("numpy is required for energy analysis. "
                          "Install with: pip install numpy") from e
    return numpy


class EnergyAnalyzer:
    """Detects audio energy spikes in a video via ffmpeg volume analysis."""

    def __init__(
        self,
        *,
        segment_size: float = 0.5,
        threshold_multiplier: float = 1.5,
        window_size: int = 10,
        verbose: bool = False,
    ) -> None:
        self.segment_size = segment_size
        self.threshold_multiplier = threshold_multiplier
        self.window_size = window_size
        self.verbose = verbose

    def detect(self, video_path: Path) -> List[EnergySpike]:
        """Energy spikes in the video, sorted by energy level (highest first)."""
        np = _numpy()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        energy_values = self._extract_audio_energy(video_path)
        if not energy_values:
            return []

        baseline = self._rolling_baseline(np, energy_values)
        spikes = self._scan_spikes(np, energy_values, baseline)
        spikes.sort(key=lambda s: s.energy_level, reverse=True)

        if self.verbose:
            active().info(f"  Detected {len(spikes)} energy spikes")
            for spike in spikes[:5]:
                active().info(
                    f"    {spike.start:.2f}s-{spike.end:.2f}s: {spike.energy_level:.1f}/100"
                )
        return spikes

    def _rolling_baseline(self, np, energy_values: List[float]) -> List[float]:
        """Moving average of energy, centered ±window_size/2 on each segment."""
        baseline = []
        for i in range(len(energy_values)):
            start = max(0, i - self.window_size // 2)
            end = min(len(energy_values), i + self.window_size // 2)
            baseline.append(np.mean(energy_values[start:end]))
        return baseline

    def _scan_spikes(
        self, np, energy_values: List[float], baseline: List[float]
    ) -> List[EnergySpike]:
        """Walk the energy series, grouping above-threshold runs into spikes."""
        spikes: List[EnergySpike] = []
        in_spike = False
        spike_start = 0
        spike_energy: List[float] = []

        for i, energy in enumerate(energy_values):
            if energy > (baseline[i] * self.threshold_multiplier):
                if not in_spike:
                    in_spike = True
                    spike_start = i
                    spike_energy = [energy]
                else:
                    spike_energy.append(energy)
            elif in_spike:
                in_spike = False
                spikes.append(
                    self._make_spike(np, spike_start, i, spike_energy, energy_values, baseline)
                )

        if in_spike:
            spikes.append(
                self._make_spike(
                    np, spike_start, len(energy_values), spike_energy, energy_values, baseline
                )
            )
        return spikes

    def _make_spike(
        self,
        np,
        spike_start: int,
        end_index: int,
        spike_energy: List[float],
        energy_values: List[float],
        baseline: List[float],
    ) -> EnergySpike:
        """Build one spike from its energy window. Keyword/viral scores filled in later."""
        avg_energy = np.mean(spike_energy)
        max_energy = np.max(spike_energy)
        peak = np.max(energy_values)
        return EnergySpike(
            start=spike_start * self.segment_size,
            end=end_index * self.segment_size,
            duration=end_index * self.segment_size - spike_start * self.segment_size,
            energy_level=min(100, (max_energy / peak) * 100),
            energy_delta=avg_energy - baseline[spike_start],
            keywords=[],
            keyword_score=0.0,
            viral_score=0.0,
            confidence=min(1.0, avg_energy / peak),
        )

    def _extract_audio_energy(self, video_path: Path) -> List[float]:
        """Extract audio energy values using ffmpeg's volumedetect filter."""
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-af',
            f'aformat=s16:44100,volumedetect=r=10:nb_samples={int(44100 * self.segment_size)}',
            '-f', 'null', '-',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Energy extraction timed out on {video_path.name}")

        energy_values = []
        for line in result.stderr.split('\n'):
            if 'mean_volume:' in line:
                try:
                    db_value = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                    # Convert dB to a rough 0-100 linear scale (-40dB to 0dB range).
                    linear_value = max(0, min(100, (db_value + 40) * 2.5))
                    energy_values.append(linear_value)
                except (IndexError, ValueError):
                    continue

        if not energy_values:
            energy_values = self._extract_audio_energy_fallback(video_path)
        return energy_values

    def _extract_audio_energy_fallback(self, video_path: Path) -> List[float]:
        """Fallback energy extraction if volumedetect fails: RMS over raw PCM chunks."""
        np = _numpy()
        cmd = ['ffmpeg', '-i', str(video_path), '-f', 's16le', '-']
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
        except Exception:
            return []

        chunk_size = int(44100 * self.segment_size)
        energy_values = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) > 0:
                # cast to float64 first: int16 ** 2 overflows -> nan
                chunk = chunk.astype(np.float64)
                rms = np.sqrt(np.mean(chunk ** 2))
                energy_values.append(min(100.0, float(rms) / 32768 * 100))
        return energy_values


class ViralKeywordDetector:
    """Scores a transcript window against the viral-keyword categories (0-10)."""

    def __init__(self, keywords=VIRAL_KEYWORDS) -> None:
        self._keywords = keywords

    def detect(
        self, transcript: List[dict], moment_start: float, moment_end: float
    ) -> Tuple[List[str], float]:
        """Return (keywords_found, keyword_score 0-10) for the moment's transcript window."""
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
        for category, config in self._keywords.items():
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

        if weights_applied > 0:
            keyword_score = min(10.0, keyword_score / weights_applied * 1.5)
        return list(set(found_keywords)), keyword_score


class ViralRanker:
    """Fuses energy spikes with keyword signals and selects the top viral moments."""

    def __init__(self, keyword_detector: Optional[ViralKeywordDetector] = None) -> None:
        self._keywords = keyword_detector or ViralKeywordDetector()

    def combine(
        self, energy_spikes: List[EnergySpike], transcript: List[dict]
    ) -> List[EnergySpike]:
        """Attach keywords to each spike and compute a viral score (60% energy, 40% keywords),
        returning the spikes sorted by that score (highest first)."""
        updated_spikes = []
        for spike in energy_spikes:
            keywords, keyword_score = self._keywords.detect(transcript, spike.start, spike.end)
            spike.keywords = keywords
            spike.keyword_score = keyword_score
            viral_score = (spike.energy_level / 10) * 0.6 + keyword_score * 0.4
            spike.viral_score = min(10.0, viral_score)
            updated_spikes.append(spike)
        updated_spikes.sort(key=lambda s: s.viral_score, reverse=True)
        return updated_spikes

    @staticmethod
    def top(
        energy_spikes: List[EnergySpike],
        count: int = 10,
        min_duration: float = 30.0,
        max_duration: float = 60.0,
    ) -> List[EnergySpike]:
        """Top `count` spikes whose duration is within [min_duration, max_duration]."""
        filtered = [s for s in energy_spikes if min_duration <= s.duration <= max_duration]
        return filtered[:count]
