"""The per-request logger + EventMeter service adapters."""

from __future__ import annotations

from kinoforge.contract import MeterAction
from kinoforge.observ import build_logger
from kinoforge.service.meter import EventMeter

# --- EventMeter -----------------------------------------------------------

def test_meter_records_action_qty_and_variant():
    meter = EventMeter()
    action = next(iter(MeterAction))
    meter(action, 3, "hd")
    meter(action, 2)
    assert meter.events == [
        {"action": action.value, "qty": 3.0, "variant": "hd"},
        {"action": action.value, "qty": 2.0, "variant": ""},
    ]


def test_meter_coerces_qty_to_float():
    meter = EventMeter()
    meter(next(iter(MeterAction)), 5)
    assert isinstance(meter.events[0]["qty"], float)


# --- KinoLogger ----------------------------------------------------------

def test_event_logger_captures_entries_and_keeps_idempotency():
    logger = build_logger(job_id="p1", segment="clips", idempotency_key="k9")
    logger.info("started")
    logger.success("done", kept=2)
    assert logger.idempotency_key == "k9"
    assert logger.entries == [
        {"level": "info", "text": "started"},
        {"level": "success", "text": "done"},
    ]
