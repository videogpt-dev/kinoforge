from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from kinoforge.contract import Meter, ModelRef
from kinoforge.inference import InfrelayClient, InfrelayError
from kinoforge.segments.clips.transcription import Transcriber, TranscriptionError
from kinoforge.service.settings import ServiceSettings


class GatewayTranscription:
    """The Transcriber's transcribe_bytes port over Infrelay; gateway failures surface as
    TranscriptionError so the engine's fallbacks catch them."""

    def __init__(self, infrelay: InfrelayClient) -> None:
        self._infrelay = infrelay

    def __call__(self, audio: bytes, ref: ModelRef, **kwargs: Any) -> Tuple[List[Dict], Dict]:
        try:
            return self._infrelay.transcribe(audio, ref, **kwargs)
        except InfrelayError as exc:
            raise TranscriptionError(str(exc)) from exc


def gateway_transcriber(
    settings: ServiceSettings,
    owner: str,
    transcription: Mapping[str, Any],
    meter: Optional[Meter] = None,
) -> Transcriber:
    return Transcriber(
        transcription=dict(transcription),
        cache_dir=settings.cache_dir,
        transcribe_bytes=GatewayTranscription(settings.infrelay(owner)),
        meter=meter,
    )
