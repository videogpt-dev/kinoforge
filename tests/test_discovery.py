"""Transcript-first MomentDiscoverer: the context-window budget and tail truncation."""

from __future__ import annotations

from kinoforge.observ import bind, build_logger, reset
from kinoforge.segments.clips.moments.discovery import MomentDiscoverer


def _disc(context_window: int) -> MomentDiscoverer:
    # llm is unused by the budget path, so a bare object stands in for it.
    return MomentDiscoverer(object(), "op", "op:m", context_window=context_window)


def test_fit_budget_keeps_all_when_under_window():
    d = _disc(1_000_000)
    segs = [{"text": "hello world"} for _ in range(5)]
    assert d._fit_budget(segs) == segs


def test_fit_budget_truncates_tail_when_over_window():
    d = _disc(1)  # tiny window clamps to the floor budget
    segs = [{"text": "x" * 1000} for _ in range(10)]
    kept = d._fit_budget(segs)
    assert 0 < len(kept) < len(segs)


def test_fit_budget_warns_when_truncating():
    logger = build_logger(segment="clips")
    token = bind(logger)
    try:
        _disc(1)._fit_budget([{"text": "x" * 1000} for _ in range(10)])
    finally:
        reset(token)
    assert any(entry["level"] == "warning" for entry in logger.entries)


def test_budget_scales_with_context_window():
    assert _disc(1_000_000)._budget_chars > _disc(32_000)._budget_chars
