"""Injected capabilities the story writer needs but does not own.

The caller (cloud core today, self-host thin runtime later) resolves definitions and
text/billing wiring, then hands the writer these ports. The engine imports no `app.*`
and reaches inference only through them, so it runs the same under either control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

from kinoforge.contract import Meter, _no_meter


class Complete(Protocol):
    def __call__(
        self,
        purpose: str,
        system: str,
        user: str,
        *,
        pick: Dict,
        project: str,
        temperature: float,
        max_tokens: int,
        bill: bool,
    ) -> Tuple[str, Dict]: ...


class CompleteJson(Protocol):
    def __call__(
        self,
        purpose: str,
        system: str,
        user: str,
        *,
        pick: Dict,
        temperature: float,
        max_tokens: int,
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
