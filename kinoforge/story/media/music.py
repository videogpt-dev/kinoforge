"""Background-music generation kernel for a story video.

Derives an instrumental mood from the story's style/logline and sizes one bed to the
video's length (capped so a music model doesn't run forever — assembly loops it under the
voiceover). The provider call and its billing are injected; storage and job state stay
with the caller.
"""

from typing import Dict, Protocol

from kinoforge.story.timing import scene_seconds

_MAX_MUSIC_SECONDS = 30  # music models are slow/limited; the composition loops it


class GenerateMusic(Protocol):
    def __call__(
        self, prompt: str, provider: str, model: str, *, seconds: int, project: str
    ) -> bytes: ...


class MusicPreset(Protocol):
    def __call__(self, mood: str) -> str: ...


class MusicComposer:
    def __init__(self, *, generate_music: GenerateMusic, music_preset: MusicPreset) -> None:
        self._generate_music = generate_music
        self._music_preset = music_preset

    def prompt(self, story: Dict) -> str:
        mood = (story.get("style") or story.get("logline") or "cinematic").strip()
        return self._music_preset(mood)

    def seconds(self, story: Dict) -> int:
        total = sum(scene_seconds(scene) for scene in (story.get("scenes") or []))
        return max(5, min(int(round(total)) or _MAX_MUSIC_SECONDS, _MAX_MUSIC_SECONDS))

    def compose(self, story: Dict, cfg: Dict, project: str = "") -> bytes:
        """Music bytes for the story, sized to its scenes. Raises through the injected
        provider on a bad route or gateway error."""
        return self._generate_music(
            self.prompt(story),
            cfg["provider"],
            cfg["model"],
            seconds=self.seconds(story),
            project=project,
        )
