from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kinoforge.contract import AITask, Meter, ModelRef, no_meter


class Complete(Protocol):
    def __call__(
        self, task: AITask, system: str, user: str, *,
        route: ModelRef, temperature: float, max_tokens: int,
    ) -> tuple[str, dict]: ...


class CompleteJson(Protocol):
    def __call__(
        self, task: AITask, system: str, user: str, *,
        route: ModelRef, temperature: float, max_tokens: int,
    ) -> dict: ...


class ParseJson(Protocol):
    def __call__(self, text: str) -> dict: ...


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
    meter: Meter = no_meter
