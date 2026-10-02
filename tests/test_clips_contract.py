"""Contract guards for the clips request: moment routing lives in `config` only, and the
runner's {**config, **options} merge must not let an options default clobber it. This is the
regression guard for the bug where an empty options.moment_route silently forced offline."""

from __future__ import annotations

from kinoforge.schemas import ClipsOptionsRequest
from kinoforge.segments.clips.moments.finders import MomentFinders

_ROUTE = {"provider": "cloud", "model": "auto"}


def _merge(config: dict) -> dict:
    """Exactly what ClipsRuntime.execute does: options over config."""
    return {**config, **ClipsOptionsRequest().model_dump()}


def test_options_model_does_not_declare_moment_fields():
    # If these come back onto options, the merge can clobber config again.
    fields = set(ClipsOptionsRequest.model_fields)
    assert "moment_route" not in fields
    assert "moment_finder" not in fields


def test_config_route_survives_the_options_merge():
    merged = _merge({"moment_finder": "auto", "moment_route": _ROUTE})
    assert merged["moment_route"] == _ROUTE  # not clobbered by an options default
    assert MomentFinders.prefers_offline(merged) is False  # AI selected


def test_offline_in_config_forces_offline_through_merge():
    merged = _merge({"moment_finder": "offline", "moment_route": _ROUTE})
    assert MomentFinders.prefers_offline(merged) is True
