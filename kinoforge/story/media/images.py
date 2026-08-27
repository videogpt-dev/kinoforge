"""Scene-image generation kernel for a story video.

Builds character/style reference and per-scene prompts, and runs the image call with one
learned retry against a provider's prompt-length ceiling. The provider call, style
presets and the prompt-limit cache are injected; the project store, reference-sheet
composition, placement and progress stay with the caller.
"""

from typing import Any, Callable, Dict, List, Optional

from kinoforge.story.media.prompt_text import fit_labeled, leading_sentences, mentioned_characters

_IMAGE_KIND = "image"


class ImagePainter:
    def __init__(
        self,
        *,
        generate_image: Callable[..., bytes],
        image_scene: Callable[..., str],
        image_character: Callable[..., str],
        image_negative: Callable[[], str],
        prompt_limit_configured: Callable[[str, str, str], int],
        prompt_limit_from_error: Callable[[Exception], int],
        prompt_limit_remember: Callable[[str, str, str, int], None],
    ) -> None:
        self._generate_image = generate_image
        self._image_scene = image_scene
        self._image_character = image_character
        self._image_negative = image_negative
        self._limit_configured = prompt_limit_configured
        self._limit_from_error = prompt_limit_from_error
        self._limit_remember = prompt_limit_remember

    def character_prompt(self, story: Dict, rec: Dict, max_chars: int = 0) -> Optional[str]:
        chars = story.get("characters") or []
        if not chars:
            return None
        who = "; ".join(f"{c['name']}: {c['description']}" for c in chars if c.get("description"))
        if not who:
            return None
        style = story.get("style") or ""
        key = ((rec.get("input") or {}).get("image_preset") or "").strip()
        prompt = self._image_character(who, style, key=key or None)
        if not max_chars or len(prompt) <= max_chars:
            return prompt
        compact_who = "; ".join(
            f"{c['name']}: {leading_sentences(str(c['description']), 2)}"
            for c in chars
            if c.get("name") and c.get("description")
        )
        compact = self._image_character(
            compact_who, leading_sentences(style, 2), key=key or None
        )
        return fit_labeled([("Character sheet", compact)], max_chars)

    def with_character_traits(self, prompt: str, characters: List[Dict]) -> str:
        """Add concise appearance anchors only for cast referenced by this scene."""
        low = prompt.lower()
        traits = [
            f"{c['name']} is {leading_sentences(str(c['description']), 2)}"
            for c in mentioned_characters(characters[:6], prompt)
            if (c.get("name") or "").strip()
            and (c.get("description") or "").strip()
            and c["description"].lower() not in low
        ]
        if not traits:
            return prompt
        return (
            f"{prompt.rstrip('. ')}. Keep these characters' appearance consistent wherever "
            f"they appear: {'; '.join(traits)}"
        )

    def image_preset_key(self, rec: Dict, scene: Dict) -> str:
        """The image style this scene uses: its own pick, else the project's, else "" (the
        default Cinematic look)."""
        scene_key = (scene.get("image_preset") or "").strip()
        if scene_key:
            return scene_key
        return ((rec.get("input") or {}).get("image_preset") or "").strip()

    def scene_prompt(self, scene: Dict, story: Dict, rec: Dict, max_chars: int = 0) -> str:
        prompt = self.with_character_traits(
            scene.get("prompt") or "", story.get("characters") or []
        )
        style = story.get("style") or ""
        key = self.image_preset_key(rec, scene)
        if not style and not key:
            return fit_labeled([("Scene", prompt)], max_chars) if max_chars else prompt
        rendered = self._image_scene(prompt, style, key=key or None)
        if not max_chars or len(rendered) <= max_chars:
            return rendered
        # Under a provider ceiling, scene content wins. Render visual direction separately
        # so a long look preset cannot consume the whole request before scene detail appears.
        visual_direction = self._image_scene("", style, key=key or None)
        return fit_labeled(
            [("Scene", prompt), ("Visual direction", visual_direction)],
            max_chars,
        )

    def negative(self) -> str:
        """What the image must not contain."""
        return self._image_negative()

    def generate_for(
        self,
        prompt_factory: Callable[[int], str],
        cfg: Dict[str, str],
        **options: Any,
    ) -> bytes:
        """Generate once; learn and retry one provider-declared image prompt ceiling."""
        provider, model = cfg["provider"], cfg["model"]
        limit = self._limit_configured(provider, _IMAGE_KIND, model)
        try:
            return self._generate_image(prompt_factory(limit), provider, model, **options)
        except Exception as error:
            observed = self._limit_from_error(error)
            if not observed or (limit and observed >= limit):
                raise
            self._limit_remember(provider, _IMAGE_KIND, model, observed)
            return self._generate_image(prompt_factory(observed), provider, model, **options)
