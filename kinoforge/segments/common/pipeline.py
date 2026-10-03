from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from kinoforge.segments.common.worker import Worker

W = TypeVar("W", bound=Worker)


class Stage(ABC, Generic[W]):
    """One pipeline step. Returning False stops the pipeline."""

    @abstractmethod
    def __call__(self, worker: W) -> bool: ...


class Pipeline(ABC, Generic[W]):
    """Checks `ready`, then runs each stage in order until one returns False. An unexpected
    exception fails the worker instead of escaping."""

    def __init__(self, *stages: Stage[W]) -> None:
        self._stages = stages

    @abstractmethod
    def ready(self, worker: W) -> bool: ...

    def run(self, worker: W) -> None:
        try:
            if worker.stopped() or not self.ready(worker):
                return
            for stage in self._stages:
                if not stage(worker):
                    return
        except Exception as exc:
            worker.logger.error(f"Unexpected error: {exc}")
            worker.fail(f"Unexpected error: {exc!s}")
