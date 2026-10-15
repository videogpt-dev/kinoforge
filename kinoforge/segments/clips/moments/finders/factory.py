from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict

from kinoforge.contract import ModelRef
from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.observ import active
from kinoforge.segments.clips.moments.finders.ai import AiMomentFinder
from kinoforge.segments.clips.moments.finders.base import MomentFinder
from kinoforge.segments.clips.moments.finders.offline import OfflineMomentFinder

RouteCompleter = Callable[..., str]


class MomentFinders:
    """Picks the offline or AI finder from config.moment_finder (auto|ai|offline). The AI one
    runs config.moment_route (blank: the gateway's route for the task) through
    `complete(route, prompt, *, max_tokens, temperature)`, and ranks offline when it finds
    nothing."""

    @staticmethod
    def prefers_offline(config: Dict[str, Any]) -> bool:
        return str(config.get("moment_finder") or "auto") == "offline"

    @classmethod
    def pick(cls, config: Dict[str, Any], definitions: DefinitionBundle,
             complete: RouteCompleter) -> MomentFinder:
        route = ModelRef.from_mapping(config.get("moment_route"))
        if cls.prefers_offline(config):
            active().info("Moment finder: offline")
            return OfflineMomentFinder()
        active().info(f"Moment finder: AI {route}", context_window=config.get("context_window"))
        return AiMomentFinder(
            route,
            context_window=int(config.get("context_window") or 1_000_000),
            complete=partial(complete, route),
            render=DefinitionRenderer(definitions),
            max_workers=int((config.get("scoring") or {}).get("max_workers") or 4),
        )
