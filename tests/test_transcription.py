"""Transcriber (cache, metering, word timings), transcript cache, and caption alignment."""

from __future__ import annotations

from kinoforge.contract import MeterAction, ModelRef
from kinoforge.segments.clips.transcription import Transcriber, TranscriptionError, WordAligner
from kinoforge.segments.clips.transcription.cache import TranscriptCache

_SEGMENTS = [{"start": 0.0, "end": 30.0, "text": "a"}, {"start": 30.0, "end": 90.0, "text": "b"}]


class _Gateway:
    def __init__(self, result=None, error=None):
        self.calls = []
        self._result = result if result is not None else list(_SEGMENTS)
        self._error = error

    def __call__(self, audio, ref, **kwargs):
        self.calls.append((ref, kwargs))
        if self._error:
            raise self._error
        return self._result, {"language": "en"}


def _transcriber(tmp_path, gateway, meter=None):
    return Transcriber(
        transcription={"model": "base", "device": "cpu", "beam_size": 5},
        cache_dir=tmp_path / "cache", transcribe_bytes=gateway, meter=meter,
    )


def test_transcribes_once_then_serves_cache_without_rebilling(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"sound")
    gateway, metered = _Gateway(), []
    transcriber = _transcriber(tmp_path, gateway, lambda *a, **k: metered.append((a, k)))
    assert transcriber.transcribe_video(audio) == _SEGMENTS
    assert transcriber.transcribe_video(audio) == _SEGMENTS
    assert len(gateway.calls) == 1
    ref, kwargs = gateway.calls[0]
    assert ref == ModelRef("whisper", "base")
    assert kwargs["params"]["beam_size"] == 5 and kwargs["params"]["device"] == "cpu"
    assert metered == [((MeterAction.TRANSCRIBE_MINUTE, 1.5), {"variant": "base"})]


def test_output_dir_cache_and_model_override(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"sound")
    gateway = _Gateway()
    _transcriber(tmp_path, gateway).transcribe_video(audio, "small", "hi", tmp_path / "out")
    assert gateway.calls[0][0] == ModelRef("whisper", "small")
    assert list((tmp_path / "out" / "transcripts").glob("*_small_hi.json"))


def test_empty_result_is_not_cached_or_billed(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"sound")
    metered = []
    transcriber = _transcriber(tmp_path, _Gateway(result=[]), lambda *a, **k: metered.append(a))
    assert transcriber.transcribe_video(audio) == []
    assert metered == [] and not (tmp_path / "cache").exists()


def test_word_timing_failure_returns_empty(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"sound")
    gateway = _Gateway(error=TranscriptionError("down"))
    assert _transcriber(tmp_path, gateway).transcribe_words(audio, "en") == []


def test_word_timing_requests_word_timestamps(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"sound")
    gateway = _Gateway()
    _transcriber(tmp_path, gateway).transcribe_words(audio)
    assert gateway.calls[0][1]["word_timestamps"] is True


def test_cache_key_follows_content_not_path(tmp_path):
    one, two = tmp_path / "one.mp3", tmp_path / "two.mp3"
    one.write_bytes(b"same")
    two.write_bytes(b"same")
    cache = TranscriptCache(tmp_path)
    assert cache.path_for(one, "base", None) == cache.path_for(two, "base", None)
    assert cache.path_for(one, "base", None).name.endswith("_base_auto.json")
    assert TranscriptCache.load(tmp_path / "missing.json") is None


# --- alignment ------------------------------------------------------------

def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_align_keeps_script_spelling_and_heard_timing():
    heard = [{"words": [_w("the", 0, .3), _w("ai", .3, .6), _w("todo", .6, 1.0)]}]
    words = WordAligner.align("The AI to-do", heard, 1.2)[0]["words"]
    assert [w["word"] for w in words] == ["The", "AI", "to-do"]
    assert [(w["start"], w["end"]) for w in words] == [(0, .3), (.3, .6), (.6, 1.0)]


def test_align_interpolates_missed_words_between_neighbours():
    heard = [{"words": [_w("one", 0, .5), _w("three", 1.0, 1.5)]}]
    words = WordAligner.align("one two three", heard, 2.0)[0]["words"]
    assert (words[1]["start"], words[1]["end"]) == (0.5, 1.0)


def test_align_spreads_evenly_without_timings():
    words = WordAligner.align("a b", [], 2.0)[0]["words"]
    assert [(w["start"], w["end"]) for w in words] == [(0.0, 1.0), (1.0, 2.0)]
    assert WordAligner.align("", [], 2.0) == []
