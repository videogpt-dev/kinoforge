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
    """Picks the offline or AI finder from config.moment_finder (auto|ai|offline) and
    config.moment_route. `complete(route, prompt, *, max_tokens, temperature)` serves the AI."""

    @staticmethod
    def prefers_offline(config: Dict[str, Any]) -> bool:
        finder = str(config.get("moment_finder") or "auto")
        return finder == "offline" or not (config.get("moment_route") or {})

    @classmethod
    def pick(cls, config: Dict[str, Any], definitions: DefinitionBundle,
             complete: RouteCompleter) -> MomentFinder:
        finder = str(config.get("moment_finder") or "auto")
        route = ModelRef.from_mapping(config.get("moment_route"))
        if cls.prefers_offline(config):
            if finder == "ai":
                active().warning(
                    "moment_finder=ai but no moment_route resolved; using offline. "
                    "The caller must send config.moment_route={provider, model}."
                )
            active().info("Moment finder: offline", moment_finder=finder,
                          has_route=bool(route.provider))
            return OfflineMomentFinder()
        active().info(f"Moment finder: AI {route}", context_window=config.get("context_window"))
        return AiMomentFinder(
            route,
            context_window=int(config.get("context_window") or 1_000_000),
            complete=partial(complete, route),
            render=DefinitionRenderer(definitions),
            max_workers=int((config.get("scoring") or {}).get("max_workers") or 4),
        )
