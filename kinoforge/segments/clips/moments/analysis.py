from __future__ import annotations

from typing import Dict, List


class MomentAnalysis:
    """Coerces a scoring model's raw output into the standard analysis shape, and supplies a
    neutral fallback when the model returns nothing usable."""

    @staticmethod
    def fallback(batch: List[Dict]) -> List[Dict]:
        """Neutral analysis for a batch when the model returned nothing usable: keep every
        moment at its own boundaries with a middling score."""
        return [
            {"worthy": True, "score": 60.0, "reason": "fallback", "hook": "",
             "start": m.get("start"), "end": m.get("end")}
            for m in batch
        ]

    @staticmethod
    def normalize(obj: Dict, moment: Dict, window: tuple) -> Dict:
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
