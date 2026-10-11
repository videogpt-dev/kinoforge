"""Scene-image generation kernel for a story video.

Builds character/style reference and per-scene prompts and runs the image call under the
provider's prompt ceiling. The provider call and style presets are injected; the project
store, reference-sheet composition, placement and progress stay with the caller.
"""

from collections.abc import Callable

from kinoforge.contract import ImageSpec, ModelRef
from kinoforge.segments.story.media.ceiling import PromptCeiling
from kinoforge.segments.story.media.prompt_text import LabeledPrompt, PromptText

GenerateImage = Callable[[ModelRef, str, ImageSpec], bytes]

REFERENCE_ANCHOR = (
    "Match the reference image for how the characters and products look (faces, hair, "
    "clothing, colors, logos), but compose this new shot instead of copying the reference"
)


class ImagePainter:
    def __init__(
        self,
        *,
        generate_image: GenerateImage,
        image_scene: Callable[..., str],
        image_character: Callable[..., str],
        image_negative: Callable[[], str],
        ceiling: PromptCeiling,
    ) -> None:
        self._generate_image = generate_image
        self._image_scene = image_scene
        self._image_character = image_character
        self._image_negative = image_negative
        self._ceiling = ceiling

    def character_prompt(self, story: dict, rec: dict, max_chars: int = 0) -> str:
        """The character-sheet prompt, or "" when the story has no described cast."""
        chars = [c for c in story.get("characters") or [] if c.get("description")]
        who = "; ".join(self._who(c, str(c["description"])) for c in chars)
        if not who:
            return ""
        style = story.get("style") or ""
        key = ((rec.get("input") or {}).get("image_preset") or "").strip()
        prompt = self._image_character(who, style, key=key or None)
        if not max_chars or len(prompt) <= max_chars:
            return prompt
        compact_who = "; ".join(
            self._who(c, PromptText.leading_sentences(str(c["description"]), 2)) for c in chars
        )
        compact = self._image_character(
            compact_who, PromptText.leading_sentences(style, 2), key=key or None
        )
        return LabeledPrompt([("Character sheet", compact)]).fit(max_chars)

    @staticmethod
    def _who(character: dict, description: str) -> str:
        name = str(character.get("name") or "").strip()
        return f"{name}: {description}" if name else description

    def with_character_traits(
        self, prompt: str, characters: list[dict], story_texts: list[str] | None = None
    ) -> str:
        """Add concise appearance anchors only for cast referenced by this scene."""
        low = prompt.lower()
        traits = [
            f"{c['name']} is {PromptText.leading_sentences(str(c['description']), 2)}"
            for c in PromptText.mentioned_characters(characters[:6], prompt, story_texts)
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

    def image_preset_key(self, rec: dict, scene: dict) -> str:
        """The image style this scene uses: its own pick, else the project's, else "" (the
        default Cinematic look)."""
        scene_key = (scene.get("image_preset") or "").strip()
        if scene_key:
            return scene_key
        return ((rec.get("input") or {}).get("image_preset") or "").strip()

    @staticmethod
    def with_reference(prompt: str) -> str:
        """A reference image only anchors identity when the model is told what it is for;
        unexplained, edit-style models tend to return the reference itself."""
        return f"{prompt.rstrip('. ')}. {REFERENCE_ANCHOR}." if prompt else prompt

    def scene_prompt(
        self, scene: dict, story: dict, rec: dict, max_chars: int = 0, *, referenced: bool = False
    ) -> str:
        prompt = self.with_character_traits(
            scene.get("prompt") or "",
            story.get("characters") or [],
            [str(s.get("prompt") or "") for s in story.get("scenes") or []],
        )
        if referenced:
            prompt = self.with_reference(prompt)
        style = story.get("style") or ""
        key = self.image_preset_key(rec, scene)
        if not style and not key:
            return LabeledPrompt([("Scene", prompt)]).fit(max_chars) if max_chars else prompt
        rendered = self._image_scene(prompt, style, key=key or None)
        if not max_chars or len(rendered) <= max_chars:
            return rendered
        # Under a provider ceiling, scene content wins. Render visual direction separately
        # so a long look preset cannot consume the whole request before scene detail appears.
        visual_direction = self._image_scene("", style, key=key or None)
        return LabeledPrompt(
            [("Scene", prompt), ("Visual direction", visual_direction)]
        ).fit(max_chars)

    def negative(self) -> str:
        """What the image must not contain."""
        return self._image_negative()

    def generate_for(
        self, prompt_factory: Callable[[int], str], ref: ModelRef, spec: ImageSpec
    ) -> bytes:
        return self._ceiling.run(
            lambda limit: self._generate_image(ref, prompt_factory(limit), spec)
        )
