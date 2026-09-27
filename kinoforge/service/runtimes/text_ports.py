"""Adapts an Infrelay text client + resolved definitions into the StoryPorts a story or series
stage consumes.

Story and series runtimes used to build StoryPorts from a stack of per-request nested closures
(complete / complete_json / prompt / contract / language_rule / story_budget). Those are bound
methods here instead: one object holds the client, renderer, default route, and budgets, and
`to_ports()` hands the stage a StoryPorts backed by it. Series passes no fragment keys, so its
contract/language_rule raise (planning does not use them)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from kinoforge.contract import Meter, _no_meter
from kinoforge.definitions import DefinitionRenderer
from kinoforge.definitions.models import DefinitionKind
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.inference import InfrelayClient


class InfrelayTextPorts:
    def __init__(
        self,
        infrelay: InfrelayClient,
        renderer: DefinitionRenderer,
        *,
        default_pick: Dict[str, Any],
        budgets: Dict[str, Any],
        label: str,
        meter: Meter = _no_meter,
        contract_key: Optional[str] = None,
        language_key: Optional[str] = None,
    ) -> None:
        self._infrelay = infrelay
        self._renderer = renderer
        self._default_pick = default_pick
        self._budgets = budgets
        self._label = label  # "story" / "series", for the missing-route error
        self._meter = meter
        self._contract_key = contract_key
        self._language_key = language_key

    def to_ports(self) -> StoryPorts:
        return StoryPorts(
            complete=self.complete,
            complete_json=self.complete_json,
            parse_json=self.parse_json,
            prompt=self.prompt,
            contract=self.contract,
            language_rule=self.language_rule,
            story_budget=self.story_budget,
            meter=self._meter,
        )

    def complete(
        self,
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
        route = {**self._default_pick, **(pick or {})}
        provider = str(route.get("provider") or "")
        model = str(route.get("model") or "")
        if not provider:
            raise RuntimeError(f"{self._label} text route is missing a provider")
        return self._infrelay.text(
            provider, model, system, user, temperature=temperature, max_tokens=max_tokens
        )

    def complete_json(
        self,
        purpose: str,
        system: str,
        user: str,
        *,
        pick: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        text, _ = self.complete(
            purpose, system, user, pick=pick, project="",
            temperature=temperature, max_tokens=max_tokens, bill=True,
        )
        return self.parse_json(text)

    @staticmethod
    def parse_json(text: str) -> Dict[str, Any]:
        """The JSON object out of a completion, whatever the model wrapped it in."""
        s = (text or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            s = s[start : end + 1]
        return json.loads(s)

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
        limit = int(self._budgets.get("story") or 0)
        if stage_override in (None, ""):
            return limit
        try:
            requested = int(str(stage_override))
        except (TypeError, ValueError):
            return limit
        return min(requested, limit) if requested > 0 else limit
