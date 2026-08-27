"""Deterministic checks between story writing and expensive media generation.

Prompts are guidance; weak or routed models can ignore them while still returning
valid JSON. Two kinds of finding come out of this: blocking ones, where the plan
literally cannot become film (a scene with nothing to say), and advisory ones, where
the plan is usable but imperfect (a prompt that asks for on-screen text an image model
renders badly). Blocking stops the run; advisory rides along as a warning so the user
decides whether to keep the story or regenerate, rather than paying to rewrite over a nit.
"""

import re
from typing import Dict, List, Tuple

_VISIBLE_TEXT_RE = re.compile(
    r'["“”][^"“”]{2,}["“”]'
    r"|(?<!\w)'[^'\n]{2,}'(?!\w)"
    r"|\b(?:text|lettering|caption|logo|watermark)\b"
    r"|\b(?:sign|note|screen|label)\s+(?:saying|reading|showing|with)\b",
    re.IGNORECASE,
)


def validate_story(story: Dict, ctx: Dict) -> Tuple[List[str], List[str]]:
    """Split quality findings into (blocking, advisory).

    Blocking = the plan cannot become film, so the run stops. Advisory = usable but
    worth flagging, so the story proceeds with the messages attached as warnings and the
    user decides. Both lists empty means a clean story.
    """
    scenes = story.get("scenes") or []
    blocking: List[str] = []
    advisory: List[str] = []
    expected = int(ctx.get("scene_count") or 0)

    # Blocking: a scene with no narration has no voiceover to speak or time against.
    silent = [
        str(i + 1) for i, scene in enumerate(scenes) if not (scene.get("narration") or "").strip()
    ]
    if silent:
        blocking.append(f"scenes {', '.join(silent)} have no narration")

    if not (story.get("logline") or "").strip():
        advisory.append("logline is missing")
    if not (story.get("style") or "").strip():
        advisory.append("visual style is missing")
    if expected and len(scenes) != expected:
        advisory.append(f"expected {expected} scenes, got {len(scenes)}")

    text_prompts = [
        str(i + 1)
        for i, scene in enumerate(scenes)
        if _VISIBLE_TEXT_RE.search(str(scene.get("prompt") or ""))
    ]
    if text_prompts:
        advisory.append(f"scene prompts {', '.join(text_prompts)} may render on-screen text")

    if ctx.get("require_motion") and scenes and not any(s.get("motion") for s in scenes):
        advisory.append("Video Mode set but no scene is marked for motion")

    return blocking, advisory
