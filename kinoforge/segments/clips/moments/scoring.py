from __future__ import annotations

from typing import Dict, List, Optional

from kinoforge.segments.clips.moments.analysis import MomentAnalysis
from kinoforge.segments.clips.moments.llm_client import LlmClient
from kinoforge.observ import active, logged


class MomentScorer:
    """Candidate filter + score: pre-filter, then batched LLM analysis that judges, scores, and
    picks natural clip boundaries within each candidate's context window."""

    def __init__(
        self,
        llm: LlmClient,
        provider: str,
        name: str,
        min_clip_length: float,
        max_clip_length: float,
        batch_size: int,
        max_candidates: int,
        context_pad: float,
    ) -> None:
        self._llm = llm
        self.provider = provider
        self.name = name
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
        self._batch_size = batch_size
        self._max_candidates = max_candidates
        self._context_pad = context_pad

    def set_clip_window(self, min_length: float, max_length: float) -> None:
        if min_length:
            self.min_clip_length = float(min_length)
        if max_length:
            self.max_clip_length = float(max_length)

    @logged
    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Filter + score candidate moments in one combined batched pass, attaching
        score/reason/hook/worthy so a later score_moments() makes zero extra calls."""
        if not candidates:
            return []

        active().info(f"  Analyzing moments ({self.name})...")
        from kinoforge.segments.clips.moments.filter import MomentFilter

        pre_filtered = MomentFilter.run(candidates, transcript)
        if not pre_filtered:
            return []
        top = pre_filtered[: self._max_candidates]
        analyses = self._analyze_moments_batched(top, transcript)

        kept: List[Dict] = []
        adjusted = 0
        for moment, a in zip(top, analyses):
            moment["score"] = a.get("score", 60.0)
            moment["ai_reason"] = a.get("reason", "")
            moment["ai_hook"] = a.get("hook", "")
            moment["ai_scored"] = True
            moment["provider"] = self.provider
            ns, ne = a.get("start"), a.get("end")
            if ns is not None and ne is not None and ne > ns:
                if abs(ns - moment.get("start", ns)) > 0.05 or abs(ne - moment.get("end", ne)) > 0.05:
                    adjusted += 1
                moment["start"] = ns
                moment["end"] = ne
                moment["duration"] = ne - ns
            if a.get("worthy", True):
                kept.append(moment)

        n_batches = (len(top) + self._batch_size - 1) // self._batch_size
        active().info(f"  Analyzed {len(top)} moments in {n_batches} request(s), "
              f"kept {len(kept)}, adjusted {adjusted} boundaries")
        return kept if kept else top[:10]

    @logged
    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Rank moments. Reuses scores from filter_moments; scores any stragglers."""
        unscored = [m for m in moments if not m.get("ai_scored")]
        if unscored:
            active().info(f"  Scoring {len(unscored)} moments ({self.name})...")
            analyses = self._analyze_moments_batched(unscored, transcript)
            for moment, a in zip(unscored, analyses):
                moment["score"] = a.get("score", 60.0)
                moment["ai_reason"] = a.get("reason", "")
                moment["ai_hook"] = a.get("hook", "")
                moment["ai_scored"] = True
                moment["provider"] = self.provider
                ns, ne = a.get("start"), a.get("end")
                if ns is not None and ne is not None and ne > ns:
                    moment["start"] = ns
                    moment["end"] = ne
                    moment["duration"] = ne - ns
        else:
            active().info(f"  Ranking {len(moments)} moments (scores from analysis pass, no extra calls)")

        ranked = sorted(moments, key=lambda m: m.get("score", 0), reverse=True)
        if ranked:
            active().success(f"  Ranked {len(ranked)} moments "
                  f"(top: {ranked[0].get('score', 0):.0f}, low: {ranked[-1].get('score', 0):.0f})")
        return ranked

    def _analyze_moments_batched(
        self, moments: List[Dict], transcript: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Analyze moments in batches (one request per batch), parallel across batches.
        Returns a list aligned to `moments`, each {worthy, score, reason, hook, start, end}."""
        transcript = transcript or []
        size = self._batch_size
        batches = [moments[i : i + size] for i in range(0, len(moments), size)]
        batch_results = self._llm.parallel_map(
            lambda b: self._analyze_batch(b, transcript), batches, desc="  Analyzing"
        )
        out: List[Dict] = []
        for batch, result in zip(batches, batch_results):
            if result and len(result) == len(batch):
                out.extend(result)
            else:
                out.extend(MomentAnalysis.fallback(batch))
        return out

    def _analyze_batch(self, batch: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Single request: judge + score a batch and pick natural clip boundaries, each clip
        shown with its surrounding timestamped transcript so cuts land on segment edges."""
        min_len = self.min_clip_length
        max_len = self.max_clip_length
        pad = self._context_pad

        windows: List[tuple] = []
        blocks: List[str] = []
        for i, m in enumerate(batch):
            cs = float(m.get("start", 0.0))
            ce = float(m.get("end", cs + max_len))
            win_start = max(0.0, cs - pad)
            win_end = ce + pad
            ctx = [
                s for s in transcript if s.get("end", 0) > win_start and s.get("start", 0) < win_end
            ]
            if ctx:
                win_end = max(win_end, ctx[-1].get("end", win_end))
                seg_lines = "\n".join(
                    f"[{s.get('start', 0):.1f}-{s.get('end', 0):.1f}] {s.get('text', '')[:160]}"
                    for s in ctx
                )
            else:
                seg_lines = f"[{cs:.1f}-{ce:.1f}] {m.get('text', '')[:280]}"
            windows.append((win_start, win_end))
            blocks.append(
                f"CLIP {i + 1}: rough candidate {cs:.1f}-{ce:.1f}s\n"
                f"Transcript segments (use these exact timestamps for boundaries):\n{seg_lines}"
            )

        prompt = self._llm.render(
            "prompts.agents.clip_analysis", min_len=f"{min_len:.0f}", max_len=f"{max_len:.0f}",
            clips="\n\n".join(blocks),
        )
        content = self._llm.complete(prompt, max_tokens=160 * len(batch) + 200, temperature=0.3)
        if content:
            parsed = self._llm.parse_json_array(content)
            if parsed:
                by_id: Dict[int, Dict] = {}
                for obj in parsed:
                    if isinstance(obj, dict) and "id" in obj:
                        try:
                            by_id[int(obj["id"])] = obj
                        except (ValueError, TypeError):
                            pass
                results = []
                for i in range(len(batch)):
                    obj = by_id.get(i + 1) or (parsed[i] if i < len(parsed) else {})
                    results.append(MomentAnalysis.normalize(obj, batch[i], windows[i]))
                return results

        return MomentAnalysis.fallback(batch)
