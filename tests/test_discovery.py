"""AiMomentEngine discovery: the context-window budget and tail truncation."""

from __future__ import annotations

from kinoforge.contract import ModelRef
from kinoforge.observ import bind, build_logger, reset
from kinoforge.segments.clips.moments.ai_engine import AiMomentEngine


def _disc(context_window: int) -> AiMomentEngine:
    return AiMomentEngine(
        ModelRef("op", "m"), min_clip_length=5, max_clip_length=30, tuning=None,
        complete=lambda *a, **k: "", render=lambda *a, **k: "", context_window=context_window,
    )


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


_SEGS = [{"start": float(i), "end": float(i + 1), "text": f"line {i}"} for i in range(40)]


def _engine(reply: str) -> AiMomentEngine:
    return AiMomentEngine(
        ModelRef("op", "m"), min_clip_length=5, max_clip_length=30, tuning=None,
        complete=lambda *a, **k: reply, render=lambda *a, **k: "prompt",
    )


def test_discovery_parses_fenced_json_and_snaps_to_segments():
    reply = 'sure:\n```json\n[{"start": 2.4, "end": 12.2, "score": 80, "title": "t"}]\n```'
    moments = _engine(reply).discover_moments(_SEGS, 5, 30, 3)
    assert [(m["start"], m["end"], m["score"]) for m in moments] == [(2.0, 13.0, 80)]
    assert moments[0]["ai_scored"] is True


def test_discovery_survives_unparseable_reply():
    assert _engine("no json here").discover_moments(_SEGS, 5, 30, 3) == []


def test_scoring_applies_batch_verdicts_by_id():
    reply = '[{"id": 2, "score": 90, "worthy": true}, {"id": 1, "score": 10, "worthy": true}]'
    moments = [{"start": 0.0, "end": 6.0, "text": "a"}, {"start": 10.0, "end": 16.0, "text": "b"}]
    ranked = _engine(reply).score_moments(moments, _SEGS)
    assert [(m["text"], m["score"]) for m in ranked] == [("b", 90), ("a", 10)]
