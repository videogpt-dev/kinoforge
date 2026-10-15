"""What a request's inference calls cost, collected wherever they run (an ambient list, like the
bound logger: pool workers started with `with_context` add to it too), for the caller to bill.
Each entry is one answered call: {task, kind, provider, model, credits, byok, units}."""

from __future__ import annotations

import contextvars
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

_T = TypeVar("_T")


class InferenceCharges:
    _current: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
        "kinoforge_inference_charges", default=None
    )

    @classmethod
    @contextmanager
    def collect(cls) -> Iterator[List[Dict[str, Any]]]:
        charges: List[Dict[str, Any]] = []
        token = cls._current.set(charges)
        try:
            yield charges
        finally:
            cls._current.reset(token)

    @classmethod
    def add(cls, charge: Dict[str, Any]) -> None:
        charges = cls._current.get()
        if charges is not None:
            charges.append(charge)

    @classmethod
    def reported(cls, fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        """Runs a runtime call collecting its charges, returned under `inference`."""

        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            with cls.collect() as charges:
                out = fn(*args, **kwargs)
            return {**out, "inference": charges}

        return run
