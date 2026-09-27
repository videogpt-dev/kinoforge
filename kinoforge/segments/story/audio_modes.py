"""Typed per-scene audio intent shared by generation, timing, and assembly."""

from enum import StrEnum


class SceneAudioMode(StrEnum):
    VOICEOVER = "voiceover"
    NATIVE = "native"


def mode_for(scene: dict) -> SceneAudioMode:
    try:
        return SceneAudioMode(scene.get("audio_mode") or SceneAudioMode.VOICEOVER)
    except ValueError:
        return SceneAudioMode.VOICEOVER


def includes_voiceover(scene: dict) -> bool:
    return mode_for(scene) == SceneAudioMode.VOICEOVER


def uses_native_audio(scene: dict) -> bool:
    return mode_for(scene) == SceneAudioMode.NATIVE


def clear_voiceover(scene: dict) -> None:
    for field in ("audio", "audio_v", "audio_seconds", "words"):
        scene[field] = None
