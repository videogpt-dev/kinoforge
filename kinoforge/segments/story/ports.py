from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

from kinoforge.contract import Meter, ModelRef, _no_meter


class Complete(Protocol):
    def __call__(
        self, purpose: str, system: str, user: str, *,
        route: ModelRef, temperature: float, max_tokens: int,
    ) -> Tuple[str, Dict]: ...


class CompleteJson(Protocol):
    def __call__(
        self, purpose: str, system: str, user: str, *,
        route: ModelRef, temperature: float, max_tokens: int,
    ) -> Dict: ...


class ParseJson(Protocol):
    def __call__(self, text: str) -> Dict: ...


class Prompt(Protocol):
    def __call__(self, key: str, **params: object) -> str: ...


class Contract(Protocol):
    def __call__(self, scene_count: int) -> str: ...


class LanguageRule(Protocol):
    def __call__(self, subject: str, language: str) -> str: ...


class StoryBudget(Protocol):
    def __call__(self, stage_override: object = None) -> int: ...


@dataclass(frozen=True)
class StoryPorts:
    complete: Complete
    complete_json: CompleteJson
    parse_json: ParseJson
    prompt: Prompt
    contract: Contract
    language_rule: LanguageRule
    story_budget: StoryBudget
    meter: Meter = _no_meter
