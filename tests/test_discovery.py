"""AiMomentFinder: time-split transcript parts, reply parsing, and the offline fallback."""

from __future__ import annotations

from typing import List

from kinoforge.contract import ModelRef
from kinoforge.observ import bind, build_logger, reset
from kinoforge.segments.clips.moments.finders import AiMomentFinder, MomentSpec

_SEGS = [{"start": float(i), "end": float(i + 1), "text": f"line {i}"} for i in range(40)]
_SPEC = MomentSpec(min_len=5, max_len=30, count=3)


def _finder(reply: str = "[]", context_window: int = 1_000_000,
            prompts: List[dict] | None = None) -> AiMomentFinder:
    def render(name, **params):
        if prompts is not None:
            prompts.append(params)
        return name

    return AiMomentFinder(
        ModelRef("op", "m"), complete=lambda prompt, **k: reply, render=render,
        context_window=context_window,
    )


def test_small_transcript_is_sent_whole_in_one_part():
    prompts: list = []
    _finder(prompts=prompts).find(_SEGS, _SPEC)
    assert len(prompts) == 1
    assert prompts[0]["part"] == "This is the whole transcript."
    assert prompts[0]["transcript"].splitlines()[-1] == "[39.0] line 39"
    assert prompts[0]["count"] == "3"


def test_long_transcript_is_split_by_time_with_overlap_and_no_tail_lost():
    segs = [{"start": float(i), "end": float(i + 1), "text": "x" * 200} for i in range(60)]
    prompts: list = []
    _finder(context_window=1, prompts=prompts).find(segs, MomentSpec(5, 3, 1))
    assert len(prompts) > 1
    assert "part 1 of" in prompts[0]["part"] and "0:00-" in prompts[0]["part"]
    assert prompts[-1]["transcript"].splitlines()[-1].startswith("[59.0]")
    first = prompts[0]["transcript"].splitlines()
    second = prompts[1]["transcript"].splitlines()
    assert second[0] in first


def test_parses_fenced_json_and_snaps_to_segments():
    reply = 'sure:\n```json\n[{"start": 2.4, "end": 12.2, "score": 80, "title": "t"}]\n```'
    moments = _finder(reply).find(_SEGS, _SPEC)
    assert [(m["start"], m["end"], m["score"]) for m in moments] == [(2.0, 13.0, 80)]
    assert moments[0]["ai_scored"] is True


def test_reads_the_json_after_a_reasoning_preamble():
    reply = ('Let me list lines:\n[0.0] So, you have an idea\n[4.9] and more\n'
             'Final: [{"start": 2.0, "end": 12.0, "score": 70}]')
    moments = _finder(reply).find(_SEGS, _SPEC)
    assert [(m["start"], m["score"]) for m in moments] == [(2.0, 70)]


def test_returns_at_most_count_moments():
    reply = str([{"start": s, "end": s + 6, "score": 90 - s} for s in (0, 10, 20, 30)])
    moments = _finder(reply.replace("'", '"')).find(_SEGS, MomentSpec(5, 30, 2))
    assert len(moments) == 2


def test_no_picks_falls_back_to_offline_ranking_with_a_warning():
    segs = [{"start": 0.0, "end": 6.0, "text": "one two three four five six"}]
    logger = build_logger(segment="clips")
    token = bind(logger)
    try:
        moments = _finder("no json here").find(segs, _SPEC)
    finally:
        reset(token)
    assert [m["start"] for m in moments] == [0.0]
    assert "ai_scored" not in moments[0]
    assert any("ranking the transcript offline" in e["text"] for e in logger.entries)
