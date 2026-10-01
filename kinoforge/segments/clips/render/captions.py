from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

_MAX_CHUNK_CHARS = 42


@dataclass(frozen=True)
class CaptionWindow:
    """The transcript lines to burn into one clip, and the source span that clip covers."""

    transcript: List[Dict[str, Any]]
    start: float
    end: float


class AssCaptions:
    """Writes the window as an ASS file sized to the output frame."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def write(self, window: CaptionWindow, path: Path) -> bool:
        """False when the window has no caption text (nothing is written)."""
        events = self._events(window)
        if not events:
            return False
        path.write_text(self._header() + "\n".join(events) + "\n", encoding="utf-8")
        return True

    def _events(self, window: CaptionWindow) -> List[str]:
        events: List[str] = []
        for seg in window.transcript or []:
            start, end = float(seg.get("start") or 0), float(seg.get("end") or 0)
            text = (seg.get("text") or "").strip()
            if end <= window.start or start >= window.end or not text:
                continue
            rel_start = max(0.0, start - window.start)
            rel_end = max(rel_start + 0.1, min(end, window.end) - window.start)
            events.extend(self._timed_chunks(text, rel_start, rel_end))
        return events

    def _timed_chunks(self, text: str, start: float, end: float) -> List[str]:
        chunks = self.chunks(text)
        total = sum(len(c) for c in chunks) or 1
        lines, cursor = [], start
        for chunk in chunks:
            chunk_end = min(end, cursor + max(0.4, (end - start) * len(chunk) / total))
            lines.append(f"Dialogue: 0,{self.timestamp(cursor)},{self.timestamp(chunk_end)},"
                         f"Default,,0,0,0,,{self.escape(chunk)}")
            cursor = chunk_end
        return lines

    def _header(self) -> str:
        height, width = self._height, self._width
        font_size = max(24, round(height * 0.040))
        outline = max(2, round(height * 0.003))
        margin_v, margin_h = round(height * 0.10), round(width * 0.07)
        return (
            "[Script Info]\nScriptType: v4.00+\n"
            f"PlayResX: {width}\nPlayResY: {height}\n"
            "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
            f"-1,0,0,0,100,100,0,0,1,{outline},1,2,{margin_h},{margin_h},{margin_v},1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    @staticmethod
    def timestamp(seconds: float) -> str:
        """ASS timestamp H:MM:SS.cc."""
        seconds = max(0.0, seconds)
        return (f"{int(seconds // 3600):d}:{int(seconds % 3600 // 60):02d}:"
                f"{int(seconds % 60):02d}.{int(seconds % 1 * 100):02d}")

    @staticmethod
    def escape(text: str) -> str:
        return (text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
                .replace("\n", "\\N"))

    @staticmethod
    def chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> List[str]:
        """Word-boundary chunks of at most max_chars."""
        chunks: List[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [text]
