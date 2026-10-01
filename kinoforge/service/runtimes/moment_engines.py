from __future__ import annotations

from functools import partial
from typing import Any, Dict, Union

from kinoforge.contract import ModelRef
from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.inference import InfrelayClient
from kinoforge.observ import active
from kinoforge.segments.clips.moments.ai_engine import AiMomentEngine
from kinoforge.segments.clips.moments.offline_engine import OfflineMomentEngine


class MomentEngines:
    """Picks the offline or AI moment engine per run from
    config.moment_finder (auto|ai|offline) and config.moment_route."""

    def __init__(self, infrelay: InfrelayClient) -> None:
        self._infrelay = infrelay

    @staticmethod
    def prefers_offline(config: Dict[str, Any]) -> bool:
        """Offline when moment_finder is 'offline' or no moment_route is set; else the AI finder."""
        finder = str(config.get("moment_finder") or "auto")
        return finder == "offline" or not (config.get("moment_route") or {})

    def __call__(
        self, config: Dict[str, Any], definitions: DefinitionBundle
    ) -> Union[AiMomentEngine, OfflineMomentEngine]:
        finder = str(config.get("moment_finder") or "auto")
        route = ModelRef.from_mapping(config.get("moment_route"))
        if self.prefers_offline(config):
            if finder == "ai":
                active().warning(
                    "moment_finder=ai but no moment_route resolved; using offline. "
                    "The caller must send config.moment_route={provider, model}."
                )
            active().info("Moment engine: offline", moment_finder=finder,
                          has_route=bool(route.provider))
            return OfflineMomentEngine()
        active().info(f"Moment engine: AI {route}", context_window=config.get("context_window"))
        return AiMomentEngine(
            route,
            min_clip_length=float(config["min_length"]),
            max_clip_length=float(config["max_length"]),
            tuning=config.get("scoring"),
            context_window=int(config.get("context_window") or 1_000_000),
            complete=partial(self._infrelay.complete, route),
            render=DefinitionRenderer(definitions),
        )
