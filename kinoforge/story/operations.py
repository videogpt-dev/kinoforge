import json
import re
from typing import Dict, Optional

from kinoforge.story.agent import strip_dashes
from kinoforge.story.formats import normalize
from kinoforge.story.languages import LANGUAGES, language_name
from kinoforge.story.ports import StoryPorts


class StoryOperations:
    def __init__(self, ports: StoryPorts) -> None:
        self.ports = ports

    @staticmethod
    def _clean(value: object) -> str:
        return strip_dashes(str(value or "")).strip()

    def _agent_voice(self, agent: Dict) -> str:
        persona = str(agent.get("persona") or "").strip()
        if persona:
            return persona
        stages = agent.get("stages") or []
        for stage in reversed(stages):
            if stage.get("format") == "story_json" and str(stage.get("system") or "").strip():
                return str(stage["system"])
        for stage in stages:
            if str(stage.get("system") or "").strip():
                return str(stage["system"])
        return self.ports.prompt("story_system")

    @staticmethod
    def _story_summary(story: Dict) -> str:
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
        rec: Dict,
        index: int,
        agent: Dict,
        instructions: str = "",
        pick: Optional[Dict] = None,
        max_tokens: int = 0,
    ) -> Dict:
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
            data = self.ports.complete_json(
                "story",
                self._agent_voice(agent),
                prompt,
                pick=dict(pick or {}),
                temperature=0.9,
                max_tokens=max_tokens,
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}
        return {
            "ok": True,
            "scene": {
                "prompt": self._clean(data.get("prompt") or current.get("prompt")),
                "narration": self._clean(data.get("narration")),
            },
        }

    def rewrite_characters(
        self,
        rec: Dict,
        agent: Dict,
        pick: Optional[Dict] = None,
        max_tokens: int = 0,
    ) -> Dict:
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
            data = self.ports.complete_json(
                "story",
                self._agent_voice(agent),
                prompt,
                pick=dict(pick or {}),
                temperature=0.8,
                max_tokens=max_tokens,
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}
        characters = [
            {
                "name": self._clean(character.get("name")),
                "description": self._clean(character.get("description")),
            }
            for character in data.get("characters") or []
            if isinstance(character, dict)
            and (character.get("name") or character.get("description"))
        ]
        return {"ok": True, "characters": characters}

    def translate(
        self,
        story: Dict,
        language: str,
        pick: Optional[Dict] = None,
        max_tokens: int = 0,
    ) -> Dict:
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
            data = self.ports.complete_json(
                "translate",
                self.ports.prompt("story_system"),
                prompt,
                pick=dict(pick or {}),
                temperature=0.3,
                max_tokens=max_tokens,
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}
        narration = data.get("narration")
        if not isinstance(narration, list) or len(narration) != len(scenes):
            count = len(narration) if isinstance(narration, list) else 0
            return {
                "ok": False,
                "error": f"translation returned {count} narration entries; expected {len(scenes)}",
            }
        if any(not isinstance(line, str) for line in narration):
            return {"ok": False, "error": "translation returned invalid narration"}
        source_lines = [self._clean(scene.get("narration")) for scene in scenes]
        translated_lines = [self._clean(line) for line in narration]
        if any(source and not translated for source, translated in zip(source_lines, translated_lines)):
            return {"ok": False, "error": "translation omitted narrated scenes"}
        if any(source_lines) and source_lines == translated_lines:
            return {"ok": False, "error": "translation returned original narration unchanged"}
        return {
            "ok": True,
            "story": {
                "logline": self._clean(data.get("logline") or story.get("logline")),
                "style": story.get("style", ""),
                "characters": story.get("characters") or [],
                "scenes": [
                    {"prompt": scene.get("prompt", ""), "narration": translated_lines[index]}
                    for index, scene in enumerate(scenes)
                ],
            },
        }

    def direct_shots(
        self,
        rec: Dict,
        pick: Optional[Dict] = None,
        max_tokens: int = 0,
    ) -> Dict:
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
            data = self.ports.complete_json(
                "director",
                self.ports.prompt("story_system"),
                prompt,
                pick=dict(pick or {}),
                temperature=0.7,
                max_tokens=max_tokens,
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
        return len(value) <= 160 if kind == "title" else len(value) >= 15

    def suggest_field(
        self,
        kind: str,
        title: str = "",
        description: str = "",
        idea: str = "",
        pick: Optional[Dict] = None,
        fallback_model: str = "",
    ) -> Dict:
        title, description, idea = title.strip(), description.strip(), idea.strip()
        if kind == "title":
            system = self.ports.prompt("story_suggest_title")
            max_tokens = 200
        elif kind == "description":
            system = self.ports.prompt("story_suggest_description")
            max_tokens = 300
        else:
            return {"ok": False, "error": "unknown suggestion kind"}
        user = json.dumps(
            {"title": title, "description": description, "idea": idea},
            ensure_ascii=False,
        )

        def attempt(model: str = "") -> str:
            route = dict(pick or {})
            if model:
                route["model"] = model
            text, _ = self.ports.complete(
                "suggest",
                system,
                user,
                pick=route,
                project="",
                temperature=0.8,
                max_tokens=max_tokens,
                bill=True,
            )
            return self._clean_suggestion(text, kind == "title")

        try:
            cleaned = attempt()
        except Exception as error:
            if not fallback_model:
                return {"ok": False, "error": str(error)}
            cleaned = ""
        if fallback_model and not self._plausible_suggestion(cleaned, kind):
            try:
                cleaned = attempt(fallback_model)
            except Exception as error:
                return {"ok": False, "error": str(error)}
        if not self._plausible_suggestion(cleaned, kind):
            return {"ok": False, "error": "model returned unusable suggestion"}
        return {"ok": True, "text": cleaned}
