from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, Tuple, TypeVar

from kinoforge.observ import with_context

_T = TypeVar("_T")


def ordered_map(fn: Callable[..., _T], items: Sequence[Tuple], max_workers: int) -> List[_T]:
    """fn(*item) for each item, results in input order. Runs on a thread pool when more than one
    worker and item; each task carries the bound request logger (see observ.with_context)."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(*item) for item in items]
    tasks = [with_context(fn) for _ in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda pair: pair[0](*pair[1]), zip(tasks, items)))
