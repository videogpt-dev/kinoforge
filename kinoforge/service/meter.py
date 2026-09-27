from typing import Any, Dict, List

from kinoforge.contract import MeterAction


class EventMeter:
    """Collects the meter events one run emits, returned as `meter_events` in the response.
    The caller (core) prices and bills them; kinoforge only records what happened."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def __call__(self, action: MeterAction, qty: float, variant: str = "") -> None:
        self.events.append(
            {"action": action.value, "qty": float(qty), "variant": variant}
        )
