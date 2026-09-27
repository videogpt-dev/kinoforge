from __future__ import annotations

import json
import re
from typing import Callable, List

Completer = Callable[..., str]
Renderer = Callable[..., str]


class LlmClient:
    """Transport for the AI moment engine: text completion, prompt-template rendering, ordered
    concurrent fan-out, and lenient JSON-array parsing. Holds no moment logic."""

    def __init__(self, complete: Completer, render: Renderer, max_workers: int) -> None:
        self._complete_fn = complete
        self._render_fn = render
        self._max_workers = max_workers

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        return self._complete_fn(prompt, max_tokens=max_tokens, temperature=temperature)

    def render(self, template: str, **kwargs: str) -> str:
        return self._render_fn(template, **kwargs)

    def parallel_map(self, fn, items: List, desc: str) -> List:
        """Run fn over items concurrently (network-bound), preserving order, with a tqdm bar."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List = [None] * len(items)
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            future_to_idx = {ex.submit(fn, item): i for i, item in enumerate(items)}
            completed = as_completed(future_to_idx)
            if tqdm is not None:
                completed = tqdm(
                    completed, total=len(items), desc=desc, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}]",
                )
            for fut in completed:
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None
        return results

    @staticmethod
    def parse_json_array(content: str):
        """Extract a JSON array from a model response (handles code fences/extra text)."""
        if "```" in content:
            for chunk in re.split(r"```(?:json)?", content):
                chunk = chunk.strip()
                if chunk.startswith("["):
                    content = chunk
                    break
        start, end = content.find("["), content.rfind("]")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
