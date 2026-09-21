"""Per-scene AI video (clip) generation kernel for a story video.

A video model sees one prompt at a time and a few seconds of film, so a clip's prompt
carries the shot direction, the cast (restated so free text-to-video providers keep faces
consistent), the cinematography line and what continues from the shot before. Builds those
prompts and runs the clip call with one learned retry against a provider's prompt ceiling.
The provider call and prompt-limit cache are injected; billing, the director pass, seeds,
store and placement stay with the caller.
"""

from typing import Any, Callable, Dict, List

from kinoforge.story.media.prompt_text import (
    fit_labeled,
    join_labeled,
    leading_sentences,
    mentioned_characters,
)

_VIDEO_KIND = "video"


class ClipMaker:
    def __init__(
        self,
        *,
        generate_video: Callable[..., bytes],
        prompt_limit_configured: Callable[[str, str, str], int],
        prompt_limit_from_error: Callable[[Exception], int],
        prompt_limit_remember: Callable[[str, str, str, int], None],
    ) -> None:
        self._generate_video = generate_video
        self._limit_configured = prompt_limit_configured
        self._limit_from_error = prompt_limit_from_error
        self._limit_remember = prompt_limit_remember

    def cast_in(self, story: Dict, scene_text: str = "", compact: bool = False) -> str:
        """The film's cast, restated for a clip so a character the prompt does not describe
        is not redrawn differently in every clip. The whole (small) cast is carried; the
        shot's action still decides who is on screen."""
        characters = [
            character
            for character in (story.get("characters") or [])[:6]
            if (character.get("name") or "").strip()
            and (character.get("description") or "").strip()
        ]
        if compact:
            characters = mentioned_characters(characters, scene_text)
        return "; ".join(
            f"{c.get('name')}: "
            f"{leading_sentences(str(c.get('description') or ''), 2) if compact else c.get('description')}"
            for c in characters
            if (c.get("name") or "").strip() and (c.get("description") or "").strip()
        )

    def preset_key(self, rec: Dict, scene: Dict) -> str:
        """The image preset this scene uses (its own pick, else the project's), so look and
        motion stay one choice."""
        return (scene.get("image_preset") or "").strip() or (
            (rec.get("input") or {}).get("image_preset") or ""
        ).strip()

    def video_prompt(
        self, scene: Dict, shot: Dict, story: Dict, motion: str = "", max_chars: int = 0
    ) -> str:
        """One clip's prompt: what moves, how the camera moves, who is in it, and what
        carries over from the shot before. `motion` is the image preset's cinematography
        line, applied film-wide over the per-shot direction."""
        style = (story.get("style") or "").strip()
        scene_text = (scene.get("prompt") or "").strip()
        cast = self.cast_in(story)
        parts = [
            ("", scene_text),
            ("Action", (shot.get("action") or "").strip()),
            ("Camera", (shot.get("camera") or "").strip()),
            ("Continuity", (shot.get("continuity") or "").strip()),
            ("Characters", cast),
            ("Cinematography", motion.strip()),
            ("Style", style),
        ]
        prompt = join_labeled(parts)
        if not max_chars or len(prompt) <= max_chars:
            return prompt
        compact_parts = [
            ("", scene_text),
            ("Action", (shot.get("action") or "").strip()),
            ("Camera", (shot.get("camera") or "").strip()),
            ("Continuity", (shot.get("continuity") or "").strip()),
            ("Characters", self.cast_in(story, scene_text, compact=True)),
            ("Cinematography", leading_sentences(motion, 2)),
            ("Style", leading_sentences(style, 2)),
        ]
        return fit_labeled(compact_parts, max_chars)

    def _spoken_line(self, scene: Dict) -> str:
        """The exact words a native-audio clip should voice: the scene's narration, kept
        verbatim so the spoken track matches the script."""
        from kinoforge.story.audio_modes import uses_native_audio

        if not uses_native_audio(scene):
            return ""
        return (scene.get("narration") or "").strip()

    def to_animate(self, scenes: List[Dict]) -> List[int]:
        """The indexes worth buying a clip for. The writer marks them (scenes[].motion);
        a story written before the field existed animates everything (as it did before),
        and only a deliberate "none" animates nothing."""
        if not any("motion" in s for s in scenes):
            return list(range(len(scenes)))
        return [i for i, s in enumerate(scenes) if s.get("motion")]

    def generate_clip(
        self,
        scene: Dict,
        story: Dict,
        motion: str,
        cfg: Dict,
        **options: Any,
    ) -> bytes:
        """Generate once; learn and retry one provider-declared prompt ceiling.

        A native-audio scene voices its own dialogue: the spoken words go to the provider as
        structured `dialogue`, and non-diegetic music is turned off for that clip so speech
        stays clear (a scene without dialogue keeps music on, for score-only shots)."""
        provider, model = cfg["provider"], cfg["model"]
        dialogue = self._spoken_line(scene)
        if dialogue:
            options["dialogue"] = dialogue
            options["music"] = False
        limit = self._limit_configured(provider, _VIDEO_KIND, model)
        prompt = self.video_prompt(scene, scene.get("shot") or {}, story, motion, limit)
        try:
            return self._generate_video(prompt, provider, model, **options)
        except Exception as error:
            observed = self._limit_from_error(error)
            if not observed or (limit and observed >= limit):
                raise
            self._limit_remember(provider, _VIDEO_KIND, model, observed)
            fitted = self.video_prompt(scene, scene.get("shot") or {}, story, motion, observed)
            return self._generate_video(fitted, provider, model, **options)
