from __future__ import annotations

from typing import Any, Dict, Iterable


class Moment:
    """Typed view over one moment dict. The dict stays the wire/storage shape (extra keys a
    caller or provider added ride through untouched); this owns the span arithmetic and
    invariants (duration == end - start) that stages used to redo inline."""

    __slots__ = ("data",)

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    @classmethod
    def new(cls, start: float, end: float, **fields: Any) -> Dict[str, Any]:
        """A fresh moment dict with a consistent span."""
        moment = cls(dict(fields))
        moment.set_span(start, end)
        return moment.data

    @property
    def start(self) -> float:
        return float(self.data.get("start") or 0)

    @property
    def end(self) -> float:
        return float(self.data.get("end") or 0)

    @property
    def span(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def score(self) -> float:
        return float(self.data.get("score") or 0)

    def set_span(self, start: float, end: float) -> None:
        self.data["start"] = float(start)
        self.data["end"] = float(end)
        self.data["duration"] = float(end) - float(start)

    def clamp_to(self, max_len: float) -> None:
        if max_len and self.span > max_len:
            self.set_span(self.start, self.start + max_len)

    def overlap_ratio(self, other: "Moment") -> float:
        """Shared seconds as a fraction of the shorter moment."""
        shared = max(0.0, min(self.end, other.end) - max(self.start, other.start))
        return shared / (min(self.span, other.span) or 1.0)

    @staticmethod
    def score_scale(moments: Iterable[Dict[str, Any]]) -> float:
        """Scores arrive on 0-10 or 0-100; the top score tells which."""
        return 100.0 if max((Moment(m).score for m in moments), default=0) > 10 else 10.0
