from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from kinoforge.contract import ModelRef
from kinoforge.segments.clips.moments.discovery import MomentDiscoverer
from kinoforge.segments.clips.moments.llm_client import Completer, LlmClient, Renderer
from kinoforge.segments.clips.moments.scoring import MomentScorer, MomentTuning


class AiMomentEngine:
    """Facade over the transcript-discovery and candidate-scoring strategies, backed by one
    LlmClient. discover_moments / filter_moments / score_moments / name is the moment-provider
    contract the clips runner duck-types."""

    def __init__(
        self,
        route: ModelRef,
        *,
        min_clip_length: float,
        max_clip_length: float,
        tuning: Optional[Mapping[str, Any]],
        complete: Completer,
        render: Renderer,
        context_window: int = 1_000_000,
    ):
        self.name = str(route)
        knobs = MomentTuning.from_mapping(tuning)
        llm = LlmClient(complete, render, knobs.max_workers)
        self._discoverer = MomentDiscoverer(llm, route, context_window=context_window)
        self._scorer = MomentScorer(
            llm, route, float(min_clip_length), float(max_clip_length), knobs
        )

    def discover_moments(
        self, transcript: List[Dict], min_len: float, max_len: float, target_clips: int
    ) -> List[Dict]:
        return self._discoverer.discover_moments(transcript, min_len, max_len, target_clips)

    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        return self._scorer.filter_moments(candidates, transcript)

    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        return self._scorer.score_moments(moments, transcript)
