import difflib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from kinoforge.contract import Meter, MeterAction

TranscribeBytes = Callable[..., Tuple[List[Dict], Dict]]


class TranscriptionError(RuntimeError):
    pass


def _whisper_tuning(transcription: Dict[str, Any]) -> Dict:
    """The local faster-whisper knobs the gateway's on-box engine reads."""
    return {
        "device": transcription.get("device"),
        "compute_type": transcription.get("compute_type"),
        "cpu_threads": transcription.get("cpu_threads"),
        "beam_size": transcription.get("beam_size"),
        "vad": transcription.get("vad"),
    }


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the MD5 of a file's contents (streamed in chunks)."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _transcript_cache_path(
    video_path: Path,
    model_size: str,
    language: Optional[str],
    cache_dir: Path,
    output_dir: Optional[Path] = None) -> Path:
    """Build a content-addressed cache path for a transcript.

    Keyed on the MD5 of the input file plus the model and language, so identical
    input + settings always reuse the same cached transcript (even if the file
    is renamed/moved). Stored under <output_dir>/transcripts when an output
    directory is given, otherwise under the injected cache directory.
    """
    import hashlib
    try:
        digest = _file_md5(video_path)
    except OSError:
        # Fall back to a path/size signature if the file can't be read
        digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()

    base_dir = Path(output_dir) / "transcripts" if output_dir else cache_dir
    lang = language or "auto"
    return base_dir / f"{digest}_{model_size}_{lang}.json"


def _load_cached_transcript(cache_path: Path) -> Optional[List[Dict]]:
    """Return a cached transcript if present and valid, else None."""
    try:
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _save_cached_transcript(cache_path: Path, segments: List[Dict]) -> None:
    """Persist a transcript to the cache directory (best-effort)."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)
    except OSError:
        pass


def transcribe_video(
        video_path: Path,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        transcriber_func: Optional[Callable] = None,
        output_dir: Optional[Path] = None,
        *,
        transcription: Dict[str, Any],
        cache_dir: Path,
        transcribe_bytes: TranscribeBytes,
        meter: Optional[Meter] = None) -> List[Dict]:
    """
    Transcribe video with local faster-whisper.

    Args:
        video_path: Path to video file
        model_size: Model size for local Whisper. Defaults to injected transcription config.
        language: Optional language code; auto-detect when empty.
        transcriber_func: Optional callable that handles transcription
        output_dir: When given, transcripts are cached under <output_dir>/transcripts
            keyed by the input file's MD5 (otherwise injected cache_dir is used).

    Returns:
        List of transcript segments with timestamps
    """

    # Resolve the model from platform config when a run did not pin one. Language is
    # per-run only (None = auto-detect); there is no stored language default.
    if model_size is None:
        model_size = str(transcription["model"])

    # Return a cached transcript when available (skips re-transcription entirely)
    cache_path = _transcript_cache_path(video_path, model_size, language, cache_dir, output_dir)
    cached = _load_cached_transcript(cache_path)
    if cached is not None:
        print(f"  Using cached transcript: {cache_path.name} ({len(cached)} segments)")
        return cached

    segments = _dispatch_transcription(
        video_path,
        model_size,
        language,
        transcription,
        transcribe_bytes,
        transcriber_func)
    if segments:
        _save_cached_transcript(cache_path, segments)
        print(f"  Saved transcript: {cache_path}")
        _meter_transcription(segments, model_size, meter)
    return segments


def _meter_transcription(
    segments: List[Dict], model_size: str, meter: Optional[Meter] = None
) -> None:
    """Charge for the audio actually transcribed, priced by model, a bigger Whisper
    costs more to run. Only reached on a real transcription: a cache hit returns above,
    so re-running a job does not bill for work nobody did."""
    minutes = (max(float(s.get("end") or 0.0) for s in segments) / 60.0) if segments else 0.0
    if meter:
        meter(MeterAction.TRANSCRIBE_MINUTE, minutes, variant=(model_size or "").strip())


def _dispatch_transcription(
        video_path: Path,
        model_size: str,
        language: Optional[str],
        transcription: Dict[str, Any],
        transcribe_bytes: TranscribeBytes,
        transcriber_func: Optional[Callable] = None) -> List[Dict]:
    """Run transcription on the gateway (no caching layer)."""

    # A caller-supplied transcriber still wins (tests, custom flows).
    if transcriber_func is not None:
        print("  Transcribing with custom transcriber...")
        return transcriber_func(video_path, model_size, language)
    return _transcribe_with_local_whisper(
        video_path, model_size, transcription, transcribe_bytes, language
    )


def transcribe_words(
        audio_path: Path,
        language: Optional[str] = None,
        *,
        transcription: Dict[str, Any],
        transcribe_bytes: TranscribeBytes) -> List[Dict]:
    """Transcribe an audio file with word-level timestamps (for captions), on the gateway's
    on-box faster-whisper. Returns segments [{start, end, text, words:[{word,start,end}]}];
    empty list if the gateway can't do it."""
    try:
        segments, _meta = transcribe_bytes(
            audio_path.read_bytes(), "whisper", transcription["model"],
            language=language or transcription.get("language") or "",
            word_timestamps=True, params=_whisper_tuning(transcription))
        return segments
    except (TranscriptionError, OSError) as e:
        print(f"  Word-timing transcription failed: {e}")
        return []


_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(token: str) -> str:
    """Lowercased, punctuation-stripped form for matching a spoken token to the
    script (so 'AI', 'to‑do', 'Moon's' compare on their letters/digits only)."""
    return _ALNUM.sub("", token.lower())


def _interpolate_word_times(starts: List[Optional[float]], ends: List[Optional[float]],
                            duration: float) -> Tuple[List[float], List[float]]:
    """Fill in timings for words the recognizer merged/dropped/misheard (they have
    no anchor) by spreading each unanchored run evenly across the gap between its
    neighbours, 0 on the left, the clip duration on the right.

    Returns the two fully-populated lists rather than filling the ones passed in: the
    caller's lists hold Optionals by construction, so every later read of them had to
    either re-check for None or lie about it."""
    n = len(starts)
    out_starts: List[float] = [0.0] * n
    out_ends: List[float] = [0.0] * n
    i = 0
    prev = 0.0
    while i < n:
        start, end = starts[i], ends[i]
        if start is not None and end is not None:
            out_starts[i], out_ends[i] = start, end
            prev = end
            i += 1
            continue
        j = i
        while j < n and (starts[j] is None or ends[j] is None):
            j += 1
        anchor = starts[j] if j < n else None
        right = anchor if anchor is not None else duration
        left = min(prev, right)
        width = (right - left) / (j - i)
        for k in range(j - i):
            out_starts[i + k] = left + k * width
            out_ends[i + k] = left + (k + 1) * width
        prev = out_ends[j - 1]
        i = j
    return out_starts, out_ends


def align_words(text: str, segments: List[Dict], duration: float) -> List[Dict]:
    """Fit the known narration text to spoken word timings, for captions.

    We already have the exact script, so the recognizer's *text* is never trusted, only its *timing*. Each true word is anchored to the matching recognized word;
    any word the recognizer merged, dropped, or misheard is interpolated between its
    neighbours. The result is correctly-spelled captions (no 'Naeai' / 'to -do') that
    still track the voice. With no timings at all, the words are spread evenly.

    Returns one caption segment: [{start, end, text, words: [{word, start, end}]}]."""
    truth = [t for t in text.split() if t]
    if not truth:
        return []
    duration = max(float(duration or 0), 0.1)

    heard: List[tuple] = []
    for seg in segments or []:
        for w in (seg.get("words") or []):
            s, e = w.get("start"), w.get("end")
            if s is None or e is None:
                continue
            heard.append((_norm(w.get("word") or ""), float(s), float(e)))

    starts: List[Optional[float]] = [None] * len(truth)
    ends: List[Optional[float]] = [None] * len(truth)
    if heard:
        matcher = difflib.SequenceMatcher(
            a=[h[0] for h in heard], b=[_norm(t) for t in truth], autojunk=False)
        for tag, i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            for k in range(j2 - j1):
                starts[j1 + k] = heard[i1 + k][1]
                ends[j1 + k] = heard[i1 + k][2]

    word_starts, word_ends = _interpolate_word_times(starts, ends, duration)
    words = [
        {"word": truth[i], "start": round(word_starts[i], 3), "end": round(word_ends[i], 3)}
        for i in range(len(truth))
    ]
    return [{"start": 0.0, "end": round(duration, 3), "text": " ".join(truth), "words": words}]


def _transcribe_with_local_whisper(
        video_path: Path,
        model_size: str,
        transcription: Dict[str, Any],
        transcribe_bytes: TranscribeBytes,
        language: Optional[str] = None
) -> List[Dict]:
    """Transcribe on the gateway's on-box faster-whisper, the default backend, keyless.
    Core extracts the audio and ships the bytes along with the store's Whisper tuning
    (device, compute type, beam size, VAD)."""
    print(f"  Transcribing with faster-whisper (gateway, {model_size})...")
    audio_path = extract_audio_for_transcription(video_path)
    try:
        segments, meta = transcribe_bytes(
            audio_path.read_bytes(), "whisper", model_size or "turbo",
            language=language or "", params=_whisper_tuning(transcription))
        detected = meta.get("language")
        print(f"  Transcribed {len(segments)} segments"
              + (f" | detected language: {detected}" if detected else ""))
        return segments
    finally:
        if audio_path != video_path and audio_path.exists():
            audio_path.unlink()


def extract_audio_for_transcription(video_path: Path) -> Path:
    """Extract audio from video for Whisper API"""
    # If already audio, return as-is
    if video_path.suffix.lower() in ['.mp3', '.wav', '.m4a']:
        return video_path

    # Extract audio to temp file
    audio_path = video_path.parent / f"{video_path.stem}_temp.mp3"

    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'libmp3lame',
        '-b:a', '192k',
        '-ar', '16000',
        str(audio_path)
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=300
        )
        return audio_path
    except Exception as e:
        print(f"  Warning: Audio extraction failed: {e}")
        return video_path


class Transcriber:
    def __init__(
        self,
        *,
        transcription: Dict[str, Any],
        cache_dir: Path,
        transcribe_bytes: TranscribeBytes,
        meter: Optional[Meter] = None) -> None:
        self._transcription = dict(transcription)
        self._cache_dir = Path(cache_dir)
        self._transcribe_bytes = transcribe_bytes
        self._meter = meter

    def transcribe_video(
        self,
        video_path: Path,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        transcriber_func: Optional[Callable] = None,
        output_dir: Optional[Path] = None) -> List[Dict]:
        return transcribe_video(
            video_path,
            model_size,
            language,
            transcriber_func,
            output_dir,
            transcription=self._transcription,
            cache_dir=self._cache_dir,
            transcribe_bytes=self._transcribe_bytes,
            meter=self._meter)

    def transcribe_words(
        self, audio_path: Path, language: Optional[str] = None
    ) -> List[Dict]:
        return transcribe_words(
            audio_path,
            language,
            transcription=self._transcription,
            transcribe_bytes=self._transcribe_bytes)
