"""Moment extraction: energy spikes and viral keywords, with a transcript-window fallback.

Scoring the candidates is the AI provider's job (see the moments providers); this module
only finds them. TranscriptText holds the stateless transcript utilities.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from kinoforge.observ import active
from kinoforge.segments.clips.moments.moment import Moment


def _seg_start(seg: Dict) -> float:
    return float(seg.get("start") or 0)


def _seg_end(seg: Dict) -> float:
    return float(seg.get("end") or 0)


def _span(segs: List[Dict], first: int, last: int) -> float:
    return _seg_end(segs[last]) - _seg_start(segs[first])


class TranscriptText:
    """Stateless transcript utilities: boundary snapping, window text, timing, language."""

    @staticmethod
    def snap(
        transcript: List[Dict], start: float, end: float,
        min_len: float = 0.0, max_len: float = 0.0,
    ) -> tuple:
        """Snap a rough [start, end] onto whole-segment boundaries, then size it into the
        [min_len, max_len] window: extend a short pick (forward first, then back) and trim a
        long one from the tail. Returns (start, end, text). Boundaries always land on a
        segment edge, so a clip opens and closes on a complete sentence."""
        segs = sorted((s for s in transcript if s.get("text")), key=_seg_start)
        if not segs:
            return start, end, ""
        first = TranscriptText._first_index(segs, start)
        last = TranscriptText._last_index(segs, first, end)
        first, last = TranscriptText._fit(segs, first, last, min_len, max_len)
        text = " ".join((seg.get("text") or "").strip() for seg in segs[first:last + 1]).strip()
        return _seg_start(segs[first]), _seg_end(segs[last]), text

    @staticmethod
    def _first_index(segs: List[Dict], start: float) -> int:
        """The last segment starting at or before `start` (0 when none does)."""
        first = 0
        for i, seg in enumerate(segs):
            if _seg_start(seg) > start:
                break
            first = i
        return first

    @staticmethod
    def _last_index(segs: List[Dict], first: int, end: float) -> int:
        """The first segment from `first` on that reaches `end` (the last one when none does)."""
        for i in range(first, len(segs)):
            if _seg_end(segs[i]) >= end:
                return i
        return len(segs) - 1

    @staticmethod
    def _fit(segs: List[Dict], first: int, last: int, min_len: float, max_len: float
             ) -> tuple:
        """Grow a too-short span (forward first, then back) and trim a too-long one's tail."""
        n = len(segs)
        while min_len > 0 and _span(segs, first, last) < min_len and (last < n - 1 or first > 0):
            if last < n - 1:
                last += 1
            else:
                first -= 1
        while max_len > 0 and _span(segs, first, last) > max_len and last > first:
            last -= 1
        return first, last

    @staticmethod
    def text_between(transcript: List[Dict], start: float, end: float) -> str:
        """Text of segments fully contained in [start, end]."""
        text_parts = [
            segment["text"]
            for segment in transcript
            if segment["start"] >= start and segment["end"] <= end
        ]
        return " ".join(text_parts).strip()

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    @staticmethod
    def detect_language(text: str) -> str:
        """Dominant-script language guess: pick the script with the most characters, so a stray
        glyph cannot flip the label (e.g. one Arabic character in an otherwise Hindi clip). Falls
        back to english when no non-Latin script is present."""
        counts = {
            "hindi": len(re.findall(r"[ऀ-ॿ]", text)),
            "chinese": len(re.findall(r"[一-鿿]", text)),
            "arabic": len(re.findall(r"[؀-ۿ]", text)),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else "english"


class MomentExtractor:
    """Finds candidate clip moments: the energy+keyword+hook path when the analyzer is
    available, else a rule-based transcript-window fallback."""

    def __init__(
        self,
        min_length: int = 30,
        max_length: int = 60,
        target_clips: int = 10,
        verbose: bool = False,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.target_clips = target_clips
        self.verbose = verbose

    def auto(self, video_path: Path, transcript: List[Dict]) -> List[Dict]:
        """Auto-generate clips from audio energy + viral keywords + hook scoring, falling back
        to transcript-window extraction when the energy analyzer is missing, fails, or yields
        no valid-length moments."""
        try:
            import kinoforge.segments.clips.moments.energy_analyzer  # noqa: F401  probe
        except ImportError:
            if self.verbose:
                active().warning("  Energy analyzer not available, using traditional extraction")
            return self.candidates(transcript)

        try:
            moments = self._energy_moments(video_path, transcript)
        except Exception as e:
            if self.verbose:
                active().warning(f"  Auto-generation failed: {e}")
                active().warning("  Falling back to traditional extraction...")
            return self.candidates(transcript)

        if not moments:
            return self.candidates(transcript)

        if self.verbose:
            active().success(f"  Generated {len(moments)} moments")
            for i, m in enumerate(moments, 1):
                active().info(
                    f"    {i}. {m['start']:.1f}s-{m['end']:.1f}s (score: {m['score']:.1f}/10)"
                )
        return moments

    def _energy_moments(self, video_path: Path, transcript: List[Dict]) -> List[Dict[str, Any]]:
        """Run the energy+keyword+hook pipeline. Returns [] to signal a fallback is needed."""
        from kinoforge.segments.clips.moments.energy_analyzer import EnergyAnalyzer, ViralRanker
        from kinoforge.segments.clips.signals.hooks import HookDetector

        if self.verbose:
            active().info(f"  Auto-generating {self.target_clips} clips using energy + keywords...")
            active().info("  Step 1: Analyzing audio energy...")
        analyzer = EnergyAnalyzer(
            segment_size=0.5, threshold_multiplier=1.5, verbose=self.verbose
        )
        energy_spikes = analyzer.detect(video_path)
        if not energy_spikes:
            if self.verbose:
                active().warning(
                    "  No energy spikes detected, falling back to traditional extraction"
                )
            return []

        ranker = ViralRanker()
        if self.verbose:
            active().info("  Step 2: Detecting viral keywords...")
        combined_spikes = ranker.combine(energy_spikes, transcript)

        if self.verbose:
            active().info(f"  Step 3: Selecting top {self.target_clips} moments...")
        viral_moments = ranker.top(
            combined_spikes, count=self.target_clips,
            min_duration=self.min_length, max_duration=self.max_length,
        )

        detector = HookDetector()
        final_moments = [self._build_moment(spike, transcript, detector) for spike in viral_moments]
        final_moments.sort(key=lambda m: m["score"], reverse=True)
        if not final_moments and self.verbose:
            active().warning("  Energy path yielded no valid-length moments, "
                             "falling back to transcript extraction")
        return final_moments

    @staticmethod
    def _build_moment(spike: Any, transcript: List[Dict], detector: Any) -> Dict[str, Any]:
        """Score one viral spike (50% energy+keywords, 30% hook, 20% baseline) into a moment."""
        hook_signal = detector.analyze(transcript, spike.start, spike.end)
        text = TranscriptText.text_between(transcript, spike.start, spike.end)
        combined_score = spike.viral_score * 0.5 + (hook_signal.strength / 10) * 3 * 0.3 + 7.0 * 0.2
        return Moment.new(
            spike.start, spike.end,
            text=text,
            score=min(10.0, combined_score),
            energy_level=spike.energy_level,
            viral_keywords=spike.keywords,
            hook_type=hook_signal.hook_type,
            hook_strength=hook_signal.strength,
            reason=(
                f"Energy: {spike.energy_level:.0f}/100, "
                f"Keywords: {', '.join(spike.keywords or ['none'])}"
            ),
            language=TranscriptText.detect_language(text),
            source="energy_analysis",
        )

    def candidates(self, transcript: List[Dict]) -> List[Dict]:
        """Rule-based fallback (no AI): NON-OVERLAPPING windows spanning the whole transcript,
        each between min_length and max_length seconds. Clean, well-spread clips instead of
        hundreds of near-duplicate overlapping windows."""
        candidates = []
        n = len(transcript)
        i = 0
        while i < n:
            start_time = transcript[i]["start"]
            text_parts = []
            end_time = start_time
            j = i
            # Grow the window to at least min_length, without exceeding max_length.
            while j < n:
                seg = transcript[j]
                if seg["end"] - start_time > self.max_length:
                    break
                text_parts.append(seg["text"])
                end_time = seg["end"]
                j += 1
                if end_time - start_time >= self.min_length:
                    break

            duration = end_time - start_time
            text = " ".join(text_parts).strip()
            if self.min_length <= duration <= self.max_length and len(text.split()) >= 5:
                candidates.append(Moment.new(
                    start_time, end_time, text=text,
                    language=TranscriptText.detect_language(text),
                ))
                i = j  # advance past this window (non-overlapping)
            else:
                i += 1
        return candidates
