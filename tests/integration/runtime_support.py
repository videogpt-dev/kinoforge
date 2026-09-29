"""Shared fixtures for runtime tests: settings, definition bundles, a faked Infrelay."""

from __future__ import annotations

from kinoforge.service.settings import ServiceSettings

SETTINGS = ServiceSettings(infrelay_url="http://infrelay")


def bundle(*definitions: dict) -> dict:
    return {
        "schema_version": 1, "id": "test", "version": "1", "engine": {"minimum": "0.1.0"},
        "definitions": list(definitions),
    }


def preset(key: str, body: str, **extras) -> dict:
    return {"type": "preset", "key": key, "body": body, **({"extras": extras} if extras else {})}


def fake(monkeypatch, method: str, fn) -> None:
    monkeypatch.setattr(f"kinoforge.inference.InfrelayClient.{method}", fn)
