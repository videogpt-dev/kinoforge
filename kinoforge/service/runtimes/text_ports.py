"""Adapts an Infrelay text client + resolved definitions into the StoryPorts a story or series
stage consumes. Series passes no fragment keys, so its contract/language_rule raise."""

from __future__ import annotations

import json
import re
from typing import Any

from kinoforge.contract import Meter, ModelRef, no_meter
from kinoforge.definitions import DefinitionRenderer
from kinoforge.definitions.models import DefinitionKind
from kinoforge.inference import InfrelayClient
from kinoforge.observ import active
from kinoforge.segments.story.ports import StoryPorts

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class InfrelayTextPorts:
    def __init__(
        self,
        infrelay: InfrelayClient,
        renderer: DefinitionRenderer,
        *,
        budgets: dict[str, Any],
        label: str,
        meter: Meter = no_meter,
        contract_key: str | None = None,
        language_key: str | None = None,
    ) -> None:
        self._infrelay = infrelay
        self._renderer = renderer
        self._budgets = budgets
        self._label = label  # "story" / "series", for the missing-route error
        self._meter = meter
        self._contract_key = contract_key
        self._language_key = language_key

    def to_ports(self) -> StoryPorts:
        return StoryPorts(
            complete=self.complete, complete_json=self.complete_json,
            parse_json=self.parse_json, prompt=self.prompt, contract=self.contract,
            language_rule=self.language_rule, story_budget=self.story_budget,
            meter=self._meter,
        )

    def complete(
        self, purpose: str, system: str, user: str, *,
        route: ModelRef, temperature: float, max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        if not route.provider:
            raise RuntimeError(f"{self._label} text route is missing a provider")
        active().debug(f"{self._label} text call: {purpose}", route=str(route),
                       max_tokens=max_tokens, temperature=temperature)
        return self._infrelay.text(
            route, system, user, temperature=temperature, max_tokens=max_tokens
        )

    def complete_json(
        self, purpose: str, system: str, user: str, *,
        route: ModelRef, temperature: float, max_tokens: int,
    ) -> dict[str, Any]:
        text, _ = self.complete(
            purpose, system, user, route=route, temperature=temperature, max_tokens=max_tokens
        )
        return self.parse_json(text)

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        """The JSON object out of a completion, whatever the model wrapped it in. Raw line
        breaks inside strings and trailing commas, two common model slips, are tolerated."""
        s = (text or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            s = s[start : end + 1]
        try:
            return json.loads(s, strict=False)
        except json.JSONDecodeError:
            return json.loads(_TRAILING_COMMA.sub(r"\1", s), strict=False)

    def prompt(self, key: str, **params: object) -> str:
        return self._renderer.render(DefinitionKind.AGENT, key, **params)

    def contract(self, scene_count: int) -> str:
        if not self._contract_key:
            raise RuntimeError(f"contract fragment is not available in {self._label}")
        return self._renderer.render(
            DefinitionKind.FRAGMENT, self._contract_key, scene_count=scene_count
        )

    def language_rule(self, subject: str, language: str) -> str:
        if not self._language_key:
            raise RuntimeError(f"language fragment is not available in {self._label}")
        return self._renderer.render(
            DefinitionKind.FRAGMENT, self._language_key, subject=subject, language=language
        )

    def story_budget(self, stage_override: object = None) -> int:
        """A stage's own token budget, capped by the platform ceiling; 0 means no ceiling."""
        limit = int(self._budgets.get("story") or 0)
        try:
            requested = int(str(stage_override)) if stage_override not in (None, "") else 0
        except (TypeError, ValueError):
            requested = 0
        if requested <= 0:
            return limit
        return min(requested, limit) if limit > 0 else requested
