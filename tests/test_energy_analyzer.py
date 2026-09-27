"""Viral-signal scoring: keyword detection, energy+keyword combine, top selection."""

from __future__ import annotations

import numpy as np

from kinoforge.segments.clips.moments.energy_analyzer import (
    EnergyAnalyzer,
    EnergySpike,
    ViralKeywordDetector,
    ViralRanker,
)


def _spike(start, end, energy_level=80.0):
    return EnergySpike(
        start=start,
        end=end,
        duration=end - start,
        energy_level=energy_level,
        energy_delta=5.0,
        keywords=[],
        keyword_score=0.0,
        viral_score=0.0,
        confidence=0.9,
    )


# --- ViralKeywordDetector -------------------------------------------------

def test_detect_keywords_empty_window():
    assert ViralKeywordDetector().detect([], 0.0, 10.0) == ([], 0.0)


def test_detect_keywords_emotional_word():
    transcript = [{"start": 0, "end": 10, "text": "this is absolutely amazing"}]
    keywords, score = ViralKeywordDetector().detect(transcript, 0.0, 10.0)
    assert "amazing" in keywords
    assert score == 10.0  # single category normalizes and caps


def test_detect_keywords_numeric_pattern():
    transcript = [{"start": 0, "end": 10, "text": "about 50 percent agree"}]
    keywords, score = ViralKeywordDetector().detect(transcript, 0.0, 10.0)
    assert "data" in keywords
    assert score > 0


# --- ViralRanker.combine --------------------------------------------------

def test_combine_sets_keywords_and_weighted_viral_score():
    transcript = [{"start": 0, "end": 40, "text": "this is amazing"}]
    [spike] = ViralRanker().combine([_spike(0, 40, energy_level=80.0)], transcript)
    assert "amazing" in spike.keywords
    # 60% energy (80/10=8) + 40% keyword (10) = 4.8 + 4.0 = 8.8
    assert abs(spike.viral_score - 8.8) < 1e-9


def test_combine_sorts_by_viral_score_desc():
    transcript = [
        {"start": 0, "end": 40, "text": "amazing shocking"},
        {"start": 100, "end": 140, "text": "and then things"},
    ]
    spikes = ViralRanker().combine(
        [_spike(100, 140, energy_level=10.0), _spike(0, 40, energy_level=90.0)], transcript
    )
    assert spikes[0].start == 0  # higher viral score first


# --- ViralRanker.top ------------------------------------------------------

def test_top_moments_filter_by_duration_and_count():
    spikes = [_spike(0, 40), _spike(0, 5), _spike(0, 45), _spike(0, 50)]
    top = ViralRanker.top(spikes, count=2, min_duration=30.0, max_duration=60.0)
    assert len(top) == 2
    assert all(30.0 <= s.duration <= 60.0 for s in top)


# --- EnergyAnalyzer._rolling_baseline -------------------------------------

def test_rolling_baseline_centered_mean():
    analyzer = EnergyAnalyzer(window_size=2)
    baseline = [float(v) for v in analyzer._rolling_baseline(np, [1.0, 2.0, 3.0, 4.0])]
    assert baseline == [1.0, 1.5, 2.5, 3.5]
