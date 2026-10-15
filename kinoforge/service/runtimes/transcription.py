from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from kinoforge.contract import AITask, ModelRef
from kinoforge.inference import InfrelayError
from kinoforge.segments.clips.transcription import Transcriber, TranscriptionError
from kinoforge.service.settings import ServiceSettings


class GatewayTranscriber(Transcriber):
    """A Transcriber whose audio goes to Infrelay."""

    def __init__(
        self,
        settings: ServiceSettings,
        owner: str,
        transcription: Mapping[str, Any],
    ) -> None:
        super().__init__(
            transcription=dict(transcription), cache_dir=settings.cache_dir,
            transcribe_bytes=self._gateway,
        )
        self._infrelay = settings.infrelay(owner)

    def _gateway(
        self, task: AITask, audio: bytes, ref: ModelRef, **kwargs: Any
    ) -> Tuple[List[Dict], Dict]:
        try:
            return self._infrelay.transcribe(task, audio, ref, **kwargs)
        except InfrelayError as exc:
            raise TranscriptionError(str(exc)) from exc
