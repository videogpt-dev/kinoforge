from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from kinoforge.contract import AITask, ModelRef
from kinoforge.observ import active
from kinoforge.segments.clips.render.ffmpeg import Ffmpeg, FfmpegError
from kinoforge.segments.clips.transcription.cache import TranscriptCache

TranscribeBytes = Callable[..., Tuple[List[Dict], Dict]]

_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a"}


class TranscriptionError(RuntimeError):
    pass


class Transcriber:
    """Transcribes media through the injected gateway port (`transcribe_bytes(task, audio, ref,
    ...)`), using the resolved Whisper settings, and caches transcripts. Infrelay prices the
    minutes it transcribed."""

    def __init__(
        self,
        *,
        transcription: Dict[str, Any],
        cache_dir: Path,
        transcribe_bytes: TranscribeBytes,
    ) -> None:
        self._settings = dict(transcription)
        self._cache_dir = Path(cache_dir)
        self._transcribe_bytes = transcribe_bytes

    @staticmethod
    def extract_audio(video_path: Path) -> Path:
        """16 kHz mp3 next to the source for upload; the source itself when it is already
        audio or extraction fails (the gateway can still decode it)."""
        if video_path.suffix.lower() in _AUDIO_SUFFIXES:
            return video_path
        audio_path = video_path.parent / f"{video_path.stem}_temp.mp3"
        try:
            Ffmpeg.run(["-i", str(video_path), "-vn", "-acodec", "libmp3lame", "-b:a", "192k",
                        "-ar", "16000", str(audio_path)], timeout=300)
            return audio_path
        except FfmpegError as exc:
            active().warning(f"  Audio extraction failed, uploading the source: {exc}")
            return video_path

    def _tuning(self) -> Dict[str, Any]:
        keys = ("device", "compute_type", "cpu_threads", "beam_size", "vad")
        return {key: self._settings.get(key) for key in keys}

    def transcribe_video(
        self,
        video_path: Path,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """Timed segments."""
        model = model_size or str(self._settings["model"])
        cache = TranscriptCache(Path(output_dir) / "transcripts" if output_dir else self._cache_dir)
        cache_path = cache.path_for(video_path, model, language)
        cached = cache.load(cache_path)
        if cached is not None:
            active().info(f"  Using cached transcript: {cache_path.name} ({len(cached)} segments)")
            return cached
        segments = self._transcribe(video_path, self._ref(model_size), language)
        if segments and cache.save(cache_path, segments):
            active().info(f"  Saved transcript: {cache_path}")
        return segments

    def transcribe_words(self, audio_path: Path, language: Optional[str] = None) -> List[Dict]:
        """Narration segments with word timings for captions: [{start, end, text, words}];
        empty when the gateway cannot do it."""
        try:
            segments, _meta = self._transcribe_bytes(
                AITask.STORY_CAPTIONS, audio_path.read_bytes(), self._ref(),
                language=language or self._settings.get("language") or "",
                word_timestamps=True, params=self._tuning(),
            )
            return segments
        except (TranscriptionError, OSError) as exc:
            active().warning(f"  Word-timing transcription failed: {exc}")
            return []

    def _ref(self, model_size: Optional[str] = None) -> ModelRef:
        """The task's route, whisper after it; a job naming its model size pins whisper."""
        if model_size:
            return ModelRef("whisper", model_size)
        return ModelRef(fallback=(ModelRef("whisper", str(self._settings["model"])),))

    def _transcribe(self, video_path: Path, ref: ModelRef, language: Optional[str]) -> List[Dict]:
        active().info(f"  Transcribing through the gateway ({ref})...")
        audio_path = self.extract_audio(video_path)
        try:
            segments, meta = self._transcribe_bytes(
                AITask.VIRAL_CLIP_TRANSCRIBE, audio_path.read_bytes(), ref,
                language=language or "", params=self._tuning(),
            )
        finally:
            if audio_path != video_path:
                audio_path.unlink(missing_ok=True)
        detected = meta.get("language")
        active().success(f"  Transcribed {len(segments)} segments"
                         + (f" | detected language: {detected}" if detected else ""))
        return segments
