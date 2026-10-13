"""Per-scene AI video (clip) generation kernel for a story video.

A video model sees one prompt at a time and a few seconds of film, so a clip's prompt
carries the shot direction, the cast (restated so free text-to-video providers keep faces
consistent), the cinematography line and what continues from the shot before. Builds those
prompts and runs the clip call under the provider's prompt ceiling. The provider call is
injected; billing, the director pass, seeds, store and placement stay with the caller.
"""

from collections.abc import Callable
from dataclasses import replace

from kinoforge.contract import ModelRef, VideoSpec
from kinoforge.segments.story.audio_modes import uses_native_audio
from kinoforge.segments.story.media.ceiling import PromptCeiling
from kinoforge.segments.story.media.prompt_text import LabeledPrompt, PromptText

GenerateVideo = Callable[[ModelRef, str, VideoSpec], bytes]


class ClipMaker:
    def __init__(self, *, generate_video: GenerateVideo, ceiling: PromptCeiling) -> None:
        self._generate_video = generate_video
        self._ceiling = ceiling

    def cast_in(self, story: dict, scene_text: str = "", compact: bool = False) -> str:
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
            characters = PromptText.mentioned_characters(
                characters,
                scene_text,
                [str(s.get("prompt") or "") for s in story.get("scenes") or []],
            )
        return "; ".join(
            f"{c.get('name')}: {self._describe(c, compact)}"
            for c in characters
            if (c.get("name") or "").strip() and (c.get("description") or "").strip()
        )

    @staticmethod
    def _describe(character: dict, compact: bool) -> str:
        description = str(character.get("description") or "")
        return PromptText.leading_sentences(description, 2) if compact else description

    def preset_key(self, rec: dict, scene: dict) -> str:
        """The image preset this scene uses (its own pick, else the project's), so look and
        motion stay one choice."""
        return (scene.get("image_preset") or "").strip() or (
            (rec.get("input") or {}).get("image_preset") or ""
        ).strip()

    def video_prompt(
        self, scene: dict, shot: dict, story: dict, motion: str = "", max_chars: int = 0
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
        prompt = LabeledPrompt(parts).join()
        if not max_chars or len(prompt) <= max_chars:
            return prompt
        compact_parts = [
            ("", scene_text),
            ("Action", (shot.get("action") or "").strip()),
            ("Camera", (shot.get("camera") or "").strip()),
            ("Continuity", (shot.get("continuity") or "").strip()),
            ("Characters", self.cast_in(story, scene_text, compact=True)),
            ("Cinematography", PromptText.leading_sentences(motion, 2)),
            ("Style", PromptText.leading_sentences(style, 2)),
        ]
        return LabeledPrompt(compact_parts).fit(max_chars)

    def _spoken_line(self, scene: dict) -> str:
        """The exact words a native-audio clip should voice: the scene's narration, kept
        verbatim so the spoken track matches the script."""
        if not uses_native_audio(scene):
            return ""
        return (scene.get("narration") or "").strip()

    def to_animate(self, scenes: list[dict]) -> list[int]:
        """The indexes worth buying a clip for. The writer marks them (scenes[].motion);
        a story written before the field existed animates everything (as it did before),
        and only a deliberate "none" animates nothing."""
        if not any("motion" in s for s in scenes):
            return list(range(len(scenes)))
        return [i for i, s in enumerate(scenes) if s.get("motion")]

    def generate_clip(
        self, scene: dict, story: dict, motion: str, ref: ModelRef, spec: VideoSpec
    ) -> bytes:
        return self._ceiling.run(
            lambda limit: self._generate_video(ref, *self.clip_call(scene, story, motion, spec, limit))
        )

    def clip_call(
        self, scene: dict, story: dict, motion: str, spec: VideoSpec, limit: int
    ) -> tuple[str, VideoSpec]:
        """The prompt and spec one clip runs with. A native-audio scene voices its own
        dialogue: the spoken words go to the provider as structured `dialogue`, and music is
        turned off so speech stays clear."""
        dialogue = self._spoken_line(scene)
        if dialogue:
            spec = replace(spec, dialogue=dialogue, music=False)
        return self.video_prompt(scene, scene.get("shot") or {}, story, motion, limit), spec
