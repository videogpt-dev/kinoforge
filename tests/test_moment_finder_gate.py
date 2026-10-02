"""Proof for the moment_finder gate: it deterministically decides offline vs AI moment finding.
offline (True) means no LLM is constructed or called for moment selection."""

from __future__ import annotations

from kinoforge.segments.clips.moments.finders import MomentFinders

prefers_offline = MomentFinders.prefers_offline
_ROUTE = {"provider": "openrouter", "model": "deepseek/deepseek-chat"}


def test_offline_forces_offline_even_with_a_valid_route():
    # The whole point: a real AI route is present, but 'offline' still bypasses the LLM.
    assert prefers_offline({"moment_finder": "offline", "moment_route": _ROUTE}) is True


def test_ai_and_auto_use_the_llm_when_a_route_is_present():
    assert prefers_offline({"moment_finder": "ai", "moment_route": _ROUTE}) is False
    assert prefers_offline({"moment_finder": "auto", "moment_route": _ROUTE}) is False
    assert prefers_offline({"moment_route": _ROUTE}) is False  # auto is the default


def test_no_route_is_always_offline():
    # ai/auto with no route cannot build the LLM engine, so they fall back to offline.
    assert prefers_offline({"moment_finder": "ai"}) is True
    assert prefers_offline({"moment_finder": "auto", "moment_route": {}}) is True
    assert prefers_offline({}) is True
