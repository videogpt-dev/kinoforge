"""Small, deterministic helpers for provider-facing media prompts."""

import re
from typing import Any


class PromptText:
    """Text shaping for media prompts: complete-sentence trims and cast-mention detection."""

    @staticmethod
    def leading_sentences(text: str, count: int) -> str:
        """Keep complete leading sentences; useful for compact identity/style anchors."""
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return " ".join(part for part in parts[:count] if part).strip()

    @staticmethod
    def mentioned_characters(
        characters: list[dict[str, Any]], text: str
    ) -> list[dict[str, Any]]:
        """Return cast explicitly referenced by name or role in scene direction."""
        haystack = (text or "").casefold()
        mentioned = []
        for character in characters:
            name = str(character.get("name") or "").strip()
            if not name:
                continue
            sections = [
                part.strip() for part in re.split(r"\s+(?:—|–|-)\s+", name) if part.strip()
            ]
            personal = sections[0].split()
            aliases = {name, sections[0], *sections[1:], *personal}
            if any(alias.casefold() in haystack for alias in aliases if len(alias) >= 3):
                mentioned.append(character)
        return mentioned


class LabeledPrompt:
    """A priority-ordered set of labeled prompt sections that renders to one line, either
    whole or fitted to a character budget (cutting only at a word boundary as last resort)."""

    def __init__(self, parts: list[tuple[str, str]]) -> None:
        self._parts = list(parts)

    def join(self) -> str:
        """Join non-empty sections without chat/persona boilerplate."""
        return self._join(self._parts)

    def fit(self, max_chars: int) -> str:
        """Fit priority-ordered sections within max_chars, clipping the last one if needed."""
        kept: list[tuple[str, str]] = []
        for label, text in self._parts:
            text = text.strip()
            if not text:
                continue
            candidate = self._join([*kept, (label, text)])
            if len(candidate) <= max_chars:
                kept.append((label, text))
                continue
            prefix = self._join(kept)
            separator = 0 if not kept else 2
            label_size = len(label) + 2 if label else 0
            available = max_chars - len(prefix) - separator - label_size - 1
            if available <= 0:
                break
            clipped = text[:available].rsplit(" ", 1)[0].rstrip(" ,;:.")
            if clipped:
                kept.append((label, clipped))
            break
        return self._join(kept)[:max_chars]

    @staticmethod
    def _join(parts: list[tuple[str, str]]) -> str:
        return (
            ". ".join(
                f"{label}: {text}".strip(". ") if label else text.strip(". ")
                for label, text in parts
                if text.strip()
            )
            + "."
        )
