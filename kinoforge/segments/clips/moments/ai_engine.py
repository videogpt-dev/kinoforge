from __future__ import annotations

from typing import Dict, List

from kinoforge.segments.clips.moments.discovery import MomentDiscoverer
from kinoforge.segments.clips.moments.llm_client import Completer, LlmClient, Renderer
from kinoforge.segments.clips.moments.scoring import MomentScorer

# Engine-owned defaults for the internal scoring knobs, so a caller need not know them.
_TUNING_DEFAULTS: Dict[str, float] = {
    "max_workers": 4,
    "batch_size": 8,
    "max_candidates": 40,
    "context_pad": 2.0,
}


class AiMomentEngine:
    """Facade over the transcript-discovery and candidate-scoring strategies, backed by one
    LlmClient. Its public surface (discover_moments / filter_moments / score_moments /
    set_clip_window / health_check / name / provider) is the moment-provider contract the clips
    runner duck-types."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        name: str = "",
        min_clip_length: float,
        max_clip_length: float,
        tuning: Dict[str, float],
        complete: Completer,
        render: Renderer,
        context_window: int = 1_000_000,
    ):
        self.provider = provider
        self.model = model
        self.name = name or f"{provider}:{model}"

        def tune(key: str) -> float:
            value = tuning.get(key)
            return _TUNING_DEFAULTS[key] if value in (None, 0, "") else value

        llm = LlmClient(complete, render, int(tune("max_workers")))
        self._discoverer = MomentDiscoverer(llm, provider, self.name, context_window=context_window)
        self._scorer = MomentScorer(
            llm, provider, self.name,
            float(min_clip_length), float(max_clip_length),
            int(tune("batch_size")), int(tune("max_candidates")), float(tune("context_pad")),
        )

    def set_clip_window(self, min_length: float, max_length: float) -> None:
        self._scorer.set_clip_window(min_length, max_length)

    def health_check(self) -> bool:
        return True

    def discover_moments(
        self, transcript: List[Dict], min_len: float, max_len: float, target_clips: int
    ) -> List[Dict]:
        return self._discoverer.discover_moments(transcript, min_len, max_len, target_clips)

    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        return self._scorer.filter_moments(candidates, transcript)

    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        return self._scorer.score_moments(moments, transcript)
