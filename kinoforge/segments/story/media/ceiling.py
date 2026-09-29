from __future__ import annotations

import re
from typing import Callable, TypeVar

_T = TypeVar("_T")

_VALIDATION_LIMIT = re.compile(
    r"input\.prompt.*?(?:less than or equal to|maximum|max(?:imum)? length)\D+(?P<limit>\d+)",
    re.IGNORECASE | re.DOTALL,
)


def limit_from_error(error: Exception) -> int:
    """The prompt-length ceiling a provider declared in its validation error, else 0."""
    match = _VALIDATION_LIMIT.search(str(error))
    if not match:
        return 0
    try:
        value = int(match.group("limit"))
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


class PromptCeiling:
    """Runs a generation under a provider prompt-length ceiling: try at the configured limit,
    and if the provider rejects the prompt with a smaller declared limit, learn it and retry
    once. `learned` holds the ceiling observed on this run (0 when none)."""

    def __init__(
        self, configured: int = 0, from_error: Callable[[Exception], int] = limit_from_error
    ) -> None:
        self.configured = configured
        self._from_error = from_error
        self.learned = 0

    def run(self, attempt: Callable[[int], _T]) -> _T:
        try:
            return attempt(self.configured)
        except Exception as error:
            observed = self._from_error(error)
            if not observed or (self.configured and observed >= self.configured):
                raise
            self.learned = observed
            return attempt(observed)
