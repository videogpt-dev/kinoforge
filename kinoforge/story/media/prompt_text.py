"""Small, deterministic helpers for provider-facing media prompts."""

import re
from typing import Any


def leading_sentences(text: str, count: int) -> str:
    """Keep complete leading sentences; useful for compact identity/style anchors."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(part for part in parts[:count] if part).strip()


def join_labeled(parts: list[tuple[str, str]]) -> str:
    """Join non-empty prompt sections without chat/persona boilerplate."""
    return (
        ". ".join(
            f"{label}: {text}".strip(". ") if label else text.strip(". ")
            for label, text in parts
            if text.strip()
        )
        + "."
    )


def fit_labeled(parts: list[tuple[str, str]], max_chars: int) -> str:
    """Fit priority-ordered sections, cutting only at a word boundary as last resort."""
    kept: list[tuple[str, str]] = []
    for label, text in parts:
        text = text.strip()
        if not text:
            continue
        candidate = join_labeled([*kept, (label, text)])
        if len(candidate) <= max_chars:
            kept.append((label, text))
            continue
        prefix = join_labeled(kept)
        separator = 0 if not kept else 2
        label_size = len(label) + 2 if label else 0
        available = max_chars - len(prefix) - separator - label_size - 1
        if available <= 0:
            break
        clipped = text[:available].rsplit(" ", 1)[0].rstrip(" ,;:.")
        if clipped:
            kept.append((label, clipped))
        break
    return join_labeled(kept)[:max_chars]


def mentioned_characters(characters: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Return cast explicitly referenced by name or role in scene direction."""
    haystack = (text or "").casefold()
    mentioned = []
    for character in characters:
        name = str(character.get("name") or "").strip()
        if not name:
            continue
        sections = [part.strip() for part in re.split(r"\s+(?:—|–|-)\s+", name) if part.strip()]
        personal = sections[0].split()
        aliases = {name, sections[0], *sections[1:], *personal}
        if any(alias.casefold() in haystack for alias in aliases if len(alias) >= 3):
            mentioned.append(character)
    return mentioned
