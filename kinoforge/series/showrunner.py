import json
from typing import Dict, Optional

from kinoforge.contract import MeterAction
from kinoforge.story.formats import prompt_label
from kinoforge.story.ports import StoryPorts


class Showrunner:
    def __init__(self, ports: StoryPorts) -> None:
        self.ports = ports

    def plan(self, series: Dict, count: int = 6, pick: Optional[Dict] = None) -> Dict:
        count = max(1, min(int(count or 6), 12))
        cast = (
            "; ".join(
                f"{character.get('name')}: {character.get('look')}"
                for character in series.get("cast") or []
                if character.get("name")
            )
            or "none"
        )
        episodes = [item for item in series.get("episodes") or [] if isinstance(item, dict)]
        prompt = self.ports.prompt(
            "episode_plan",
            count=count,
            format=prompt_label(series.get("aspect_ratio")),
            name=series.get("name", ""),
            premise=(series.get("premise") or "").strip() or "none",
            style=(series.get("style") or "").strip(),
            cast=cast,
            history=json.dumps(episodes, ensure_ascii=False),
        )
        try:
            data = self.ports.complete_json(
                "ideas",
                self.ports.prompt("story_system"),
                prompt,
                pick=dict(pick or {}),
                temperature=0.9,
                max_tokens=self.ports.story_budget(),
            )
        except Exception as error:
            return {"ok": False, "error": str(error)}
        self.ports.meter(MeterAction.AGENT_RUN, 1)
        episodes = [
            {
                "title": str(item.get("title") or "").strip(),
                "description": str(item.get("description") or "").strip(),
            }
            for item in data.get("episodes") or []
            if isinstance(item, dict) and (item.get("title") or item.get("description"))
        ]
        if not episodes:
            return {"ok": False, "error": "no episode ideas returned"}
        return {"ok": True, "episodes": episodes}
