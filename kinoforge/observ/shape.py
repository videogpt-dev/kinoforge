from __future__ import annotations

from typing import Any


def shape(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Log-safe view of a value: long strings truncated, lists collapsed to a count, dicts
    expanded to max_depth then collapsed, scalars kept. Keeps prompts, media bytes, and big
    bundles out of log lines while still showing real params."""
    if isinstance(value, str):
        return f"{value[:80]}... ({len(value)} chars)" if len(value) > 80 else value
    if isinstance(value, list):
        return f"list({len(value)})"
    if isinstance(value, dict):
        if depth >= max_depth:
            return f"dict({len(value)})"
        return {key: shape(item, depth + 1, max_depth) for key, item in value.items()}
    return value
