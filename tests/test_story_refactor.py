"""Story segment after the smell cleanup: StoryBrief, route-typed ports, media kernels."""

from __future__ import annotations

import json
from typing import List

from kinoforge.contract import ImageSpec, ModelRef, VideoSpec
from kinoforge.segments.story.agent import Screenwriter, StoryBrief
from kinoforge.segments.story.media.ceiling import PromptCeiling
from kinoforge.segments.story.media.images import ImagePainter
from kinoforge.segments.story.media.music import MusicComposer
from kinoforge.segments.story.media.videos import ClipMaker
from kinoforge.segments.story.operations import StoryOperations
from kinoforge.segments.story.ports import StoryPorts
from kinoforge.segments.story.run_agent import RunAgent


class _Text:
    """Fake text transport: records (purpose, route) and replays canned replies."""

    def __init__(self, replies: List[str] = ()) -> None:
        self.calls: list = []
        self._replies = list(replies)

    def complete(self, purpose, system, user, *, route, temperature, max_tokens):
        self.calls.append((purpose, route))
        text = self._replies.pop(0) if self._replies else "ok"
        return text, {"input_tokens": 1, "output_tokens": 2, "model": route.model}

    def complete_json(self, purpose, system, user, *, route, temperature, max_tokens):
        return json.loads(self.complete(purpose, system, user, route=route,
                                        temperature=temperature, max_tokens=max_tokens)[0])

    def ports(self) -> StoryPorts:
        return StoryPorts(
            complete=self.complete, complete_json=self.complete_json, parse_json=json.loads,
            prompt=lambda key, **_: f"<{key}>", contract=lambda _n: "",
            language_rule=lambda _subject, _language: "RULE",
            story_budget=lambda _override=None: 100,
        )


# --- StoryBrief / Screenwriter -------------------------------------------

def test_story_brief_ignores_unknown_keys():
    brief = StoryBrief.from_mapping({"title": "T", "require_motion": True, "extra": 1})
    assert brief.title == "T" and brief.scene_count == 8


def test_build_context_from_brief():
    ctx = Screenwriter(_Text().ports()).build_context(StoryBrief(
        title=" Heist ", scene_count=0, cast=[{"name": "Ava", "look": "tall"}, {"look": "x"}],
        mature=True,
    ))
    assert ctx["title"] == "Heist"
    assert ctx["scene_count"] == 8
    assert ctx["audience"] == "mature"
    assert ctx["language_rule"] == "RULE"
    assert json.loads(ctx["series_cast"]) == [{"name": "Ava", "look": "tall", "personality": ""}]


# --- routes through the ports --------------------------------------------

def test_run_agent_stage_overrides_the_project_route():
    text = _Text()
    agent = RunAgent(text.ports(), {"stages": []}, {"scene_count": 3},
                     route=ModelRef("op", "base"))
    assert agent._run_stage({"model": "stage-model"}, "writer", "idea") is None
    assert text.calls == [("story", ModelRef("op", "stage-model"))]
    assert agent.models_used == ["stage-model"]


def test_suggest_field_falls_back_to_the_fallback_model_on_the_same_provider():
    text = _Text(["no", "A Great Title"])
    result = StoryOperations(text.ports()).suggest_field(
        "title", idea="x", route=ModelRef("op", "m"), fallback_model="fb"
    )
    assert result == {"ok": True, "text": "A Great Title"}
    assert [route for _, route in text.calls] == [ModelRef("op", "m"), ModelRef("op", "fb")]


def test_suggest_field_rejects_unknown_kind():
    assert StoryOperations(_Text().ports()).suggest_field("poem")["ok"] is False


# --- media kernels --------------------------------------------------------

def test_image_painter_runs_under_the_ceiling_with_ref_and_spec():
    sent: list = []
    painter = ImagePainter(
        generate_image=lambda ref, prompt, spec: sent.append((ref, prompt, spec)) or b"img",
        image_scene=lambda *a, **k: "", image_character=lambda *a, **k: "",
        image_negative=lambda: "", ceiling=PromptCeiling(50),
    )
    spec = ImageSpec(aspect_ratio="1:1")
    assert painter.generate_for(lambda limit: f"p{limit}", ModelRef("fal", "flux"), spec) == b"img"
    assert sent == [(ModelRef("fal", "flux"), "p50", spec)]


def test_character_prompt_is_empty_without_described_cast():
    painter = ImagePainter(
        generate_image=lambda *a: b"", image_scene=lambda *a, **k: "",
        image_character=lambda *a, **k: "sheet", image_negative=lambda: "",
        ceiling=PromptCeiling(),
    )
    assert painter.character_prompt({"characters": [{"name": "A"}]}, {}) == ""


def _clip_spec(scene: dict) -> VideoSpec:
    sent: list = []
    maker = ClipMaker(generate_video=lambda ref, prompt, spec: sent.append(spec) or b"v",
                      ceiling=PromptCeiling())
    maker.generate_clip(scene, {}, "", ModelRef("fal"),
                        VideoSpec(seconds=5, resolution=720, aspect_ratio="9:16"))
    return sent[0]


def test_native_audio_scene_sends_dialogue_and_mutes_music():
    spec = _clip_spec({"prompt": "x", "audio_mode": "native", "narration": " hello "})
    assert (spec.dialogue, spec.music) == ("hello", False)


def test_voiceover_scene_keeps_music_and_no_dialogue():
    spec = _clip_spec({"prompt": "x", "narration": "hello"})
    assert (spec.dialogue, spec.music) == ("", True)


def test_music_composer_passes_route_prompt_and_capped_seconds():
    sent: list = []
    composer = MusicComposer(
        generate_music=lambda ref, prompt, *, seconds: sent.append((ref, prompt, seconds)) or b"m",
        music_preset=lambda mood: f"music {mood}",
    )
    assert composer.compose({"style": "noir"}, ModelRef("fal", "beats")) == b"m"
    assert sent == [(ModelRef("fal", "beats"), "music noir", 30)]


# --- translate ------------------------------------------------------------

_SCENES = {"logline": "l", "style": "s",
           "scenes": [{"prompt": "p1", "narration": "one"}, {"prompt": "p2", "narration": "two"}]}


def _translate(reply: dict, language: str = "es"):
    return StoryOperations(_Text([json.dumps(reply)]).ports()).translate(
        _SCENES, language, ModelRef("op", "m"))


def test_translate_success_keeps_prompts_and_cleans_lines():
    result = _translate({"logline": "L — es", "narration": ["uno", "dos"]})
    assert result["ok"] is True
    assert result["story"]["logline"] == "L, es"
    assert result["story"]["scenes"] == [
        {"prompt": "p1", "narration": "uno"}, {"prompt": "p2", "narration": "dos"}]


def test_translate_rejections():
    assert _translate({}, language="xx")["error"] == "unsupported language"
    assert _translate({"narration": ["uno"]})["error"] == (
        "translation returned 1 narration entries; expected 2")
    assert _translate({"narration": ["uno", 2]})["error"] == "translation returned invalid narration"
    assert _translate({"narration": ["uno", ""]})["error"] == "translation omitted narrated scenes"
    assert _translate({"narration": ["one", "two"]})["error"] == (
        "translation returned original narration unchanged")
