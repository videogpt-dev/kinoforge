from typing import Any, Dict, List

from kinoforge.contract import MeterAction


class EventLogger:
    def __init__(self) -> None:
        self.entries: List[Dict[str, str]] = []

    def _add(self, level: str, text: str) -> None:
        self.entries.append({"level": level, "text": text})

    def info(self, text: str) -> None:
        self._add("info", text)

    def success(self, text: str) -> None:
        self._add("success", text)

    def warning(self, text: str) -> None:
        self._add("warning", text)

    def error(self, text: str) -> None:
        self._add("error", text)


class EventMeter:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def __call__(self, action: MeterAction, qty: float, variant: str = "") -> None:
        self.events.append(
            {"action": action.value, "qty": float(qty), "variant": variant}
        )
