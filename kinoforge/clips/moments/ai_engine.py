from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

_TRANSCRIPT_CHARS_BUDGET = 9000

Completer = Callable[..., str]
Renderer = Callable[..., str]


class AiMomentEngine:
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
    ):
        self.provider = provider
        self.model = model
        self.name = name or f"{provider}:{model}"
        self._complete_fn = complete
        self._render = render
        self.min_clip_length = float(min_clip_length)
        self.max_clip_length = float(max_clip_length)
        self._max_workers = int(tuning["max_workers"])
        self._batch_size = int(tuning["batch_size"])
        self._max_candidates = int(tuning["max_candidates"])
        self._context_pad = float(tuning["context_pad"])

    def set_clip_window(self, min_length: float, max_length: float) -> None:
        if min_length:
            self.min_clip_length = float(min_length)
        if max_length:
            self.max_clip_length = float(max_length)

    def health_check(self) -> bool:
        return True

    def _complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        return self._complete_fn(prompt, max_tokens=max_tokens, temperature=temperature)

    def _parallel_map(self, fn, items: List, desc: str) -> List:
        """Run fn over items concurrently (network-bound), preserving order, with a tqdm bar."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List = [None] * len(items)
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            future_to_idx = {ex.submit(fn, item): i for i, item in enumerate(items)}
            completed = as_completed(future_to_idx)
            if tqdm is not None:
                completed = tqdm(
                    completed, total=len(items), desc=desc, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}]",
                )
            for fut in completed:
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None
        return results

    # --- discovery (transcript-first) ---------------------------------------------------

    def discover_moments(
        self, transcript: List[Dict], min_len: float, max_len: float, target_clips: int
    ) -> List[Dict]:
        """Choose self-contained moments by reading the WHOLE transcript (chunked when long),
        returning real start/end for each, snapped to segment edges and sized into
        [min_len, max_len]. Moments come back already scored (ai_scored=True)."""
        segs = [s for s in (transcript or []) if s.get("text")]
        if not segs:
            return []

        from kinoforge.clips.moments.extractor import detect_language, snap_to_transcript

        print(f"  Reading the full transcript to find moments ({self.name})...")
        chunks = self._chunk_transcript(segs)
        want = max(target_clips, 5)
        raw_lists = self._parallel_map(
            lambda c: self._discover_in_chunk(c, min_len, max_len, want), chunks, desc="  Reading"
        )
        raw = [m for lst in raw_lists if lst for m in lst]

        moments: List[Dict] = []
        for r in raw:
            try:
                rs, re_ = float(r["start"]), float(r["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if re_ <= rs:
                continue
            s, e, text = snap_to_transcript(segs, rs, re_, min_len, max_len)
            if e - s < 2.0:
                continue
            moments.append({
                "start": s, "end": e, "duration": e - s, "text": text,
                "score": min(max(float(r.get("score") or 60), 0), 100),
                "ai_reason": str(r.get("reason", ""))[:200],
                "ai_hook": str(r.get("hook", ""))[:120],
                "title": str(r.get("title", ""))[:120],
                "language": detect_language(text),
                "ai_scored": True, "provider": self.provider,
                "source": "transcript_discovery",
            })

        deduped = self._dedupe_overlaps(moments)
        print(f"  Chose {len(deduped)} moments from the full transcript "
              f"({len(chunks)} pass(es), {len(raw)} raw picks)")
        return deduped

    def _chunk_transcript(self, segs: List[Dict]) -> List[List[Dict]]:
        """Split into windows that fit one request, overlapping a few segments so a moment
        straddling a boundary is still seen whole. Short transcripts return one window."""
        overlap = 3
        chunks: List[List[Dict]] = []
        cur: List[Dict] = []
        size = 0
        for s in segs:
            text = str(s.get("text") or "")
            cur.append(s)
            size += len(text) + 12  # + room for the "[123.4] " prefix
            if size >= _TRANSCRIPT_CHARS_BUDGET:
                chunks.append(cur)
                cur = cur[-overlap:]
                size = sum(len(str(x.get("text") or "")) + 12 for x in cur)
        if cur and (not chunks or len(cur) > overlap):
            chunks.append(cur)
        return chunks or [segs]

    def _discover_in_chunk(
        self, chunk: List[Dict], min_len: float, max_len: float, want: int
    ) -> List[Dict]:
        """One request: hand the model a transcript window, get back its best moments."""
        lines = "\n".join(
            f"[{float(s.get('start') or 0):.1f}] {str(s.get('text') or '').strip()[:200]}"
            for s in chunk
        )
        prompt = self._render(
            "prompts.agents.moment_discovery", count=str(want), min_len=f"{min_len:.0f}",
            max_len=f"{max_len:.0f}", transcript=lines,
        )
        if not prompt.strip():
            return []
        content = self._complete(prompt, max_tokens=120 * want + 300, temperature=0.4)
        parsed = self._parse_json_array(content)
        return [p for p in parsed if isinstance(p, dict)] if parsed else []

    @staticmethod
    def _dedupe_overlaps(moments: List[Dict]) -> List[Dict]:
        """Keep the highest-scoring moment out of any overlapping set. Greedy, best first."""
        ordered = sorted(moments, key=lambda m: float(m.get("score") or 0), reverse=True)
        kept: List[Dict] = []
        for m in ordered:
            ms, me = float(m["start"]), float(m["end"])
            clash = False
            for k in kept:
                ks, ke = float(k["start"]), float(k["end"])
                overlap = max(0.0, min(me, ke) - max(ms, ks))
                shorter = min(me - ms, ke - ks) or 1.0
                if overlap / shorter > 0.5:
                    clash = True
                    break
            if not clash:
                kept.append(m)
        return kept

    # --- candidate filter + score (window-based) ----------------------------------------

    def filter_moments(self, candidates: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Filter + score candidate moments in one combined batched pass, attaching
        score/reason/hook/worthy so a later score_moments() makes zero extra calls."""
        if not candidates:
            return []

        print(f"  Analyzing moments ({self.name})...")
        from kinoforge.clips.moments.filter import filter_moments_aggressively

        pre_filtered = filter_moments_aggressively(candidates, transcript)
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
        print(f"  Analyzed {len(top)} moments in {n_batches} request(s), "
              f"kept {len(kept)}, adjusted {adjusted} boundaries")
        return kept if kept else top[:10]

    def score_moments(self, moments: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """Rank moments. Reuses scores from filter_moments; scores any stragglers."""
        unscored = [m for m in moments if not m.get("ai_scored")]
        if unscored:
            print(f"  Scoring {len(unscored)} moments ({self.name})...")
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
            print(f"  Ranking {len(moments)} moments (scores from analysis pass, no extra calls)")

        ranked = sorted(moments, key=lambda m: m.get("score", 0), reverse=True)
        if ranked:
            print(f"  Ranked {len(ranked)} moments "
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
        batch_results = self._parallel_map(
            lambda b: self._analyze_batch(b, transcript), batches, desc="  Analyzing"
        )
        out: List[Dict] = []
        for batch, result in zip(batches, batch_results):
            if result and len(result) == len(batch):
                out.extend(result)
            else:
                out.extend(
                    {"worthy": True, "score": 60.0, "reason": "fallback", "hook": "",
                     "start": m.get("start"), "end": m.get("end")}
                    for m in batch
                )
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

        prompt = self._render(
            "prompts.agents.clip_analysis", min_len=f"{min_len:.0f}", max_len=f"{max_len:.0f}",
            clips="\n\n".join(blocks),
        )
        content = self._complete(prompt, max_tokens=160 * len(batch) + 200, temperature=0.3)
        if content:
            parsed = self._parse_json_array(content)
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
                    results.append(self._normalize_analysis(obj, batch[i], windows[i]))
                return results

        return [
            {"worthy": True, "score": 60.0, "reason": "fallback", "hook": "",
             "start": m.get("start"), "end": m.get("end")}
            for m in batch
        ]

    @staticmethod
    def _normalize_analysis(obj: Dict, moment: Dict, window: tuple) -> Dict:
        """Coerce a parsed object into the standard analysis shape; clamp start/end to the
        clip's context window, keeping the moment's original boundaries when missing/invalid."""
        orig_start, orig_end = moment.get("start"), moment.get("end")
        try:
            score = float(obj.get("score", 60))
        except (ValueError, TypeError):
            score = 60.0
        worthy = obj.get("worthy", True)
        if isinstance(worthy, str):
            worthy = worthy.strip().lower() in ("true", "yes", "1")

        win_start, win_end = window
        start, end = orig_start, orig_end
        try:
            ns = float(obj["start"])
            ne = float(obj["end"])
            if ne > ns:
                ns = max(win_start, min(ns, win_end))
                ne = max(win_start, min(ne, win_end))
                if ne - ns >= 2.0:
                    start, end = ns, ne
        except (KeyError, ValueError, TypeError):
            pass

        return {
            "worthy": bool(worthy),
            "score": min(max(score, 0), 100),
            "reason": str(obj.get("reason", ""))[:200],
            "hook": str(obj.get("hook", ""))[:120],
            "start": start, "end": end,
        }

    @staticmethod
    def _parse_json_array(content: str):
        """Extract a JSON array from a model response (handles code fences/extra text)."""
        import json

        if "```" in content:
            for chunk in re.split(r"```(?:json)?", content):
                chunk = chunk.strip()
                if chunk.startswith("["):
                    content = chunk
                    break
        start, end = content.find("["), content.rfind("]")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
