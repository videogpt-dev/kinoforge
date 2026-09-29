"""HeuristicScorer: the keyless fallback ranking."""

from __future__ import annotations

from kinoforge.segments.clips.moments.moment import Moment
from kinoforge.segments.clips.moments.scorer import HeuristicScorer


def test_vague_references_match_whole_words_only():
    # "with" / "history" / "items" contain it/this/those-like substrings but are not references.
    assert HeuristicScorer.context_clarity("With history, items matter", "english") == 10.0
    assert HeuristicScorer.context_clarity("This is what they said", "english") == 7.0


def test_rank_sorts_best_first_and_keeps_fields():
    moments = [
        Moment.new(0, 10, text="plain words", language="english", clip="a"),
        Moment.new(0, 35, text="Why is the truth hidden? You never knew. First. Second. Done.",
                   language="english", clip="b"),
    ]
    ranked = HeuristicScorer.rank(moments)
    assert [m["clip"] for m in ranked] == ["b", "a"]
    assert set(ranked[0]["scores"]) == {"context_clarity", "hook_strength", "standalone",
                                        "retention"}
    assert "scores" not in moments[0]  # scored copies, inputs untouched


def test_retention_prefers_30_to_45_seconds():
    assert HeuristicScorer.retention("x", 35) > HeuristicScorer.retention("x", 10)
    assert HeuristicScorer.retention("x", 70) < HeuristicScorer.retention("x", 50)


def test_hook_counts_one_strong_hook_per_language():
    assert HeuristicScorer.hook_strength("never the secret", "english") == 8.0
    assert HeuristicScorer.hook_strength("क्यों सच", "hindi") == 8.0
    assert HeuristicScorer.rank([]) == []
