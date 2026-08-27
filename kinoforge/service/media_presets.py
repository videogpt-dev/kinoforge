"""Image style presets rendered from a frozen DefinitionBundle.

The cloud core freezes the account's resolved image presets (default look, portrait,
negative, plus any picked style) into the request bundle; this adapter renders them with
`string.Template`, matching cloud rendering exactly. The scene/character wrapping logic is
ported verbatim from core's preset store so provider input is identical on both paths.
"""

from __future__ import annotations

from string import Template

from kinoforge.definitions import DefinitionBundle
from kinoforge.definitions.models import DefinitionKind


class BundleImagePresets:
    def __init__(
        self,
        bundle: DefinitionBundle,
        *,
        scene_key: str,
        portrait_key: str,
        negative_key: str,
    ) -> None:
        self._bundle = bundle
        self._scene_key = scene_key
        self._portrait_key = portrait_key
        self._negative_key = negative_key

    def _body(self, key: str) -> str:
        return self._bundle.require(DefinitionKind.PRESET, key).body

    def _motion(self, key: str) -> str:
        extras = self._bundle.require(DefinitionKind.PRESET, key).extras
        return str(extras.get("motion") or "")

    def _render(self, key: str, **params: object) -> str:
        values = {name: "" if value is None else str(value) for name, value in params.items()}
        return Template(self._body(key)).substitute(values)

    def image_scene(self, prompt: str, style: str, key: str | None = None) -> str:
        resolved = key or self._scene_key
        body = self._render(resolved, prompt=prompt, style=style)
        # Account-created presets historically allowed a plain look description with no
        # $prompt placeholder. That made every scene receive identical provider input.
        # Preserve those presets, but always carry scene content into generation.
        if prompt.strip() and "prompt" not in Template(self._body(resolved)).get_identifiers():
            body = f"Scene: {prompt.strip('. ')}. Visual direction: {body.strip('. ')}"
        return body.strip(". ").strip()

    def image_character(self, who: str, style: str, key: str | None = None) -> str:
        portrait = self._render(self._portrait_key, who=who, style=style)
        if key and key != self._scene_key:
            portrait = self._render(key, prompt=portrait, style=style)
        return portrait.strip()

    def image_negative(self) -> str:
        return self._render(self._negative_key).strip()

    def image_motion(self, prompt: str, style: str, key: str | None = None) -> str:
        """The clip-only cinematography line from an image preset's motion field. Blank when
        the preset defines none. Never touches the still, only how it is set in motion."""
        motion = self._motion(key or self._scene_key)
        if not motion.strip():
            return ""
        values = {"prompt": "" if prompt is None else str(prompt), "style": "" if style is None else str(style)}
        return Template(motion).substitute(values).strip()
