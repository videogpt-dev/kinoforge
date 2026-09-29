from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TranscriptCache:
    """Content-addressed transcripts: keyed on the media file's MD5 plus model and language,
    so the same input and settings reuse one transcript even after the file moves."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def path_for(self, media: Path, model: str, language: Optional[str]) -> Path:
        try:
            digest = _file_md5(media)
        except OSError:
            digest = hashlib.md5(str(media).encode("utf-8")).hexdigest()
        return self._root / f"{digest}_{model}_{language or 'auto'}.json"

    @staticmethod
    def load(path: Path) -> Optional[List[Dict]]:
        """A cached non-empty transcript, else None."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, list) and data else None

    @staticmethod
    def save(path: Path, segments: List[Dict]) -> bool:
        """Best-effort write; False when the cache is not writable."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
            return True
        except OSError:
            return False
