from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple

from kinoforge.definitions import DefinitionBundle, DefinitionRenderer
from kinoforge.definitions.models import DefinitionKind
from kinoforge.series.showrunner import Showrunner
from kinoforge.service.events import EventLogger, EventMeter
from kinoforge.service.infrelay import InfrelayClient
from kinoforge.service.models import SeriesPlanRequest
from kinoforge.story.ports import StoryPorts


def _parse_json(text: str) -> Dict[str, Any]:
    """The JSON object out of a completion, whatever the model wrapped it in."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start : end + 1]
    return json.loads(s)


def _unsupported(*_args: object, **_kwargs: object) -> Any:
    raise RuntimeError("operation is not supported in series planning")


class SeriesRuntime:
    def __init__(self, infrelay_url: str, infrelay_token: str) -> None:
        self._infrelay_url = infrelay_url
        self._infrelay_token = infrelay_token

    def plan(self, request: SeriesPlanRequest) -> Dict[str, Any]:
        bundle = DefinitionBundle.from_mapping(
            request.definitions.model_dump(exclude_none=True),
            engine_version="0.1.0",
        )
        renderer = DefinitionRenderer(bundle)
        infrelay = InfrelayClient(self._infrelay_url, self._infrelay_token, request.owner)
        logger = EventLogger()
        meter = EventMeter()
        budgets = dict(request.config.get("budgets") or {})
        default_pick = dict(request.pick or {})

        def complete(
            purpose: str,
            system: str,
            user: str,
            *,
            pick: Dict[str, Any],
            project: str,
            temperature: float,
            max_tokens: int,
            bill: bool,
        ) -> Tuple[str, Dict[str, Any]]:
            over = {**default_pick, **(pick or {})}
            provider = str(over.get("provider") or "")
            model = str(over.get("model") or "")
            if not provider:
                raise RuntimeError("series text route is missing a provider")
            return infrelay.text(
                provider, model, system, user, temperature=temperature, max_tokens=max_tokens
            )

        def complete_json(
            purpose: str,
            system: str,
            user: str,
            *,
            pick: Dict[str, Any],
            temperature: float,
            max_tokens: int,
        ) -> Dict[str, Any]:
            text, _ = complete(
                purpose,
                system,
                user,
                pick=pick,
                project="",
                temperature=temperature,
                max_tokens=max_tokens,
                bill=True,
            )
            return _parse_json(text)

        def prompt(key: str, **params: object) -> str:
            return renderer.render(DefinitionKind.AGENT, key, **params)

        def story_budget(stage_override: object = None) -> int:
            return int(budgets.get("story") or 0)

        ports = StoryPorts(
            complete=complete,
            complete_json=complete_json,
            parse_json=_parse_json,
            prompt=prompt,
            contract=_unsupported,
            language_rule=_unsupported,
            story_budget=story_budget,
            meter=meter,
        )
        result = Showrunner(ports).plan(request.series.model_dump(), request.count, default_pick)
        return {"result": result, "meter_events": meter.events, "logs": logger.entries}


def series_runtime() -> SeriesRuntime:
    return SeriesRuntime(
        os.getenv("INFRELAY_URL") or "",
        os.getenv("INFRELAY_SERVICE_TOKEN") or "",
    )
