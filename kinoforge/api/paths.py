"""Shared-volume path guards and segment availability checks used across routers.

Kinoforge only touches media under KINOFORGE_SHARED_ROOT (the volume the caller also mounts);
every path a request names is resolved and confined here so a route cannot read or write
outside it."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

from kinoforge.contract import JobKind


def shared_root() -> Path:
    return Path(os.getenv("KINOFORGE_SHARED_ROOT") or "/app/output").resolve()


def shared_path(raw: str, *, must_exist: bool = False) -> Path:
    path = Path(raw).resolve()
    if not path.is_relative_to(shared_root()):
        raise HTTPException(status_code=400, detail="path is outside shared root")
    if must_exist and not path.is_file():
        raise HTTPException(status_code=404, detail=f"media not found: {path.name}")
    return path


def require_available(code_name: JobKind) -> None:
    if code_name not in (JobKind.CLIPS, JobKind.STORY):
        raise HTTPException(
            status_code=501,
            detail=f"segment is not implemented: {code_name.value}",
        )


def require_clips(code_name: JobKind) -> None:
    if code_name is not JobKind.CLIPS:
        raise HTTPException(status_code=404, detail="operation is available only for clips")
