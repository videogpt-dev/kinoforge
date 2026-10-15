import json
import re
from enum import Enum

from kinoforge.contract import AITask, ModelRef
from kinoforge.segments.story.agent import clean_text, strip_dashes
from kinoforge.segments.story.formats import normalize
from kinoforge.segments.story.languages import LANGUAGES, language_name
from kinoforge.segments.story.ports import StoryPorts


class Suggestion(Enum):
    """A field the writer can suggest: its prompt and output-token allowance."""

    TITLE = ("title", "story_suggest_title", 200)
    DESCRIPTION = ("description", "story_suggest_description", 300)

    def __init__(self, kind: str, prompt_key: str, max_tokens: int) -> None:
        self.kind = kind
        self.prompt_key = prompt_key
        self.max_tokens = max_tokens

    @classmethod
    def of(cls, kind: str) -> "Suggestion | None":
        return next((item for item in cls if item.kind == kind), None)


class StoryOperations:
    def __init__(self, ports: StoryPorts) -> None:
        self.ports = ports

    def _ask_json(
        self, task: AITask, system: str, prompt: str, route: ModelRef | None,
        temperature: float, max_tokens: int,
    ) -> dict:
        return self.ports.complete_json(
            task, system, prompt, route=route or ModelRef(),
            temperature=temperature, max_tokens=max_tokens,
        )

    def _agent_voice(self, agent: dict) -> str:
        """The same voice RunAgent writes with: the agent's persona, else the global default."""
        persona = str(agent.get("persona") or "").strip()
        return persona or self.ports.prompt("story_system")

    @staticmethod
    def _story_summary(story: dict) -> str:
        characters = "; ".join(
            f"{character.get('name')}: {character.get('description')}"
            for character in story.get("characters") or []
        )
        return (
            f"LOGLINE: {story.get('logline', '')}\n"
            f"STYLE: {story.get('style', '')}\n"
            f"CHARACTERS: {characters or 'none'}"
        )

    def rewrite_scene(
        self,
        rec: dict,
        index: int,
        agent: dict,
        instructions: str = "",
        route: ModelRef | None = None,
        max_tokens: int = 0,
    ) -> dict:
        story = rec.get("story") or {}
        scenes = story.get("scenes") or []
        if not 0 <= index < len(scenes):
            return {"ok": False, "error": "scene out of range"}
        current = scenes[index]
        language = str((rec.get("input") or {}).get("language") or "")
        prompt = self.ports.prompt(
            "story_rewrite_scene",
            summary=self._story_summary(story),
            scene_num=index + 1,
            scene_total=len(scenes),
            cur_prompt=current.get("prompt", ""),
            cur_narration=current.get("narration", ""),
            language_rule=self.ports.language_rule("narration", language_name(language)),
        )
        if instructions.strip():
            prompt = f"{prompt}\n\nExtra direction for this rewrite:\n{instructions.strip()}"
        try:
            data = self._ask_json(AITask.STORY_WRITE, self._agent_voice(agent), prompt, route, 0.9, max_tokens)
        except Exception as error:
            return {"ok": False, "error": str(error)}
        return {
            "ok": True,
            "scene": {
                "prompt": clean_text(data.get("prompt") or current.get("prompt")),
                "narration": clean_text(data.get("narration")),
            },
        }

    def rewrite_characters(
        self,
        rec: dict,
        agent: dict,
        route: ModelRef | None = None,
        max_tokens: int = 0,
    ) -> dict:
        story = rec.get("story") or {}
        scenes_text = "\n".join(
            f"- {scene.get('narration', '')}" for scene in (story.get("scenes") or [])[:12]
        )
        prompt = self.ports.prompt(
            "story_rewrite_characters",
            logline=story.get("logline", ""),
            style=story.get("style", ""),
            scenes_text=scenes_text,
        )
        try:
            data = self._ask_json(AITask.STORY_WRITE, self._agent_voice(agent), prompt, route, 0.8, max_tokens)
        except Exception as error:
            return {"ok": False, "error": str(error)}
        characters = [
            {
                "name": clean_text(character.get("name")),
                "description": clean_text(character.get("description")),
            }
            for character in data.get("characters") or []
            if isinstance(character, dict)
            and (character.get("name") or character.get("description"))
        ]
        return {"ok": True, "characters": characters}

    def translate(
        self,
        story: dict,
        language: str,
        route: ModelRef | None = None,
        max_tokens: int = 0,
    ) -> dict:
        entry = LANGUAGES.get(language.strip())
        if not entry:
            return {"ok": False, "error": "unsupported language"}
        scenes = story.get("scenes") or []
        lines = "\n".join(
            f"{index}. {scene.get('narration', '')}" for index, scene in enumerate(scenes)
        )
        prompt = self.ports.prompt(
            "story_translate",
            language=entry[0],
            logline=story.get("logline", ""),
            lines=lines,
            scene_count=len(scenes),
        )
        try:
            data = self._ask_json(
                AITask.STORY_TRANSLATE, self.ports.prompt("story_system"), prompt, route, 0.3,
                max_tokens,
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}
        translated = self._translated_lines(data.get("narration"), scenes)
        if isinstance(translated, str):
            return {"ok": False, "error": translated}
        return {
            "ok": True,
            "story": {
                "logline": clean_text(data.get("logline") or story.get("logline")),
                "style": story.get("style", ""),
                "characters": story.get("characters") or [],
                "scenes": [
                    {"prompt": scene.get("prompt", ""), "narration": line}
                    for scene, line in zip(scenes, translated)
                ],
            },
        }

    @staticmethod
    def _translated_lines(narration: object, scenes: list[dict]) -> list[str] | str:
        """The cleaned translated lines, or the reason the translation is unusable."""
        if not isinstance(narration, list) or len(narration) != len(scenes):
            count = len(narration) if isinstance(narration, list) else 0
            return f"translation returned {count} narration entries; expected {len(scenes)}"
        if any(not isinstance(line, str) for line in narration):
            return "translation returned invalid narration"
        source = [clean_text(scene.get("narration")) for scene in scenes]
        translated = [clean_text(line) for line in narration]
        if any(src and not out for src, out in zip(source, translated)):
            return "translation omitted narrated scenes"
        if any(source) and source == translated:
            return "translation returned original narration unchanged"
        return translated

    def direct_shots(
        self,
        rec: dict,
        route: ModelRef | None = None,
        max_tokens: int = 0,
    ) -> dict:
        story = rec.get("story") or {}
        scenes = story.get("scenes") or []
        if not scenes:
            return {"ok": False, "error": "no scenes to direct"}
        listing = "\n".join(
            f"{index + 1}. IMAGE: {scene.get('prompt', '')}\n"
            f"   NARRATION: {scene.get('narration', '')}"
            for index, scene in enumerate(scenes)
        )
        prompt = self.ports.prompt(
            "video_shots",
            summary=self._story_summary(story),
            scenes=listing,
            scene_total=len(scenes),
            aspect=normalize((rec.get("input") or {}).get("aspect_ratio")),
        )
        try:
            data = self._ask_json(
                AITask.STORY_DIRECT, self.ports.prompt("story_system"), prompt, route, 0.7,
                max_tokens,
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}
        raw = data.get("shots") or []
        shots = []
        for index in range(len(scenes)):
            shot = raw[index] if index < len(raw) and isinstance(raw[index], dict) else {}
            shots.append(
                {
                    "action": str(shot.get("action") or "").strip(),
                    "camera": str(shot.get("camera") or "").strip(),
                    "continuity": str(shot.get("continuity") or "").strip(),
                }
            )
        if not any(shot["action"] or shot["camera"] for shot in shots):
            return {"ok": False, "error": "director returned no shots"}
        return {"ok": True, "shots": shots}

    @staticmethod
    def _clean_suggestion(text: str, single_line: bool) -> str:
        value = (text or "").strip()
        quoted = re.findall(r'"([^\"]{3,})"', value)
        if quoted and len(value) - len(quoted[-1]) > 20:
            value = quoted[-1]
        for label in ("title:", "description:"):
            if value.lower().startswith(label):
                value = value[len(label) :].strip()
        if single_line:
            value = next((line for line in value.splitlines() if len(line.strip()) >= 3), value)
        return strip_dashes(value).strip().strip('"').strip("'").strip()

    @staticmethod
    def _plausible_suggestion(text: str, kind: str) -> bool:
        value = text.strip()
        if len(value) < 3 or re.match(r"^[\w ]{1,24}:\s*\w+\.?$", value):
            return False
        lowered = value.lower()
        refusals = ("i cannot", "i can't", "i'm sorry", "as an ai", "cannot assist", "unable to")
        if any(marker in lowered for marker in refusals):
            return False
        return len(value) <= 160 if kind == Suggestion.TITLE.kind else len(value) >= 15

    def _suggest_once(
        self, kind: str, system: str, user: str, route: ModelRef, max_tokens: int
    ) -> str:
        text, _ = self.ports.complete(
            AITask.STORY_SUGGEST, system, user, route=route, temperature=0.8,
            max_tokens=max_tokens,
        )
        return self._clean_suggestion(text, kind == Suggestion.TITLE.kind)

    def suggest_field(
        self,
        kind: str,
        title: str = "",
        description: str = "",
        idea: str = "",
        route: ModelRef | None = None,
    ) -> dict:
        suggestion = Suggestion.of(kind)
        if suggestion is None:
            return {"ok": False, "error": "unknown suggestion kind"}
        max_tokens = suggestion.max_tokens
        system = self.ports.prompt(suggestion.prompt_key)
        user = json.dumps(
            {"title": title.strip(), "description": description.strip(), "idea": idea.strip()},
            ensure_ascii=False,
        )
        try:
            cleaned = self._suggest_once(kind, system, user, route or ModelRef(), max_tokens)
        except Exception as error:
            return {"ok": False, "error": str(error)}
        if not self._plausible_suggestion(cleaned, kind):
            return {"ok": False, "error": "model returned unusable suggestion"}
        return {"ok": True, "text": cleaned}
