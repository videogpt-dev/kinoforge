"""The logging layer, built on stdlib `logging`: KinoLogger facade, config, ambient, @logged."""

from __future__ import annotations

import json
import logging

from kinoforge.observ import (
    KinoLogger,
    active,
    bind,
    build_logger,
    configure,
    current,
    logged,
    logger_for,
    reset,
    with_context,
)
from kinoforge.observ.config import default_level
from kinoforge.observ.logger import SUCCESS, TRACE


# --- KinoLogger facade ----------------------------------------------------

def test_levels_capture_entries_including_success():
    logger = build_logger(job_id="p1", segment="clips", idempotency_key="k")
    logger.info("started")
    logger.success("chose", kept=3)
    logger.warning("careful")
    logger.error("boom")
    logger.debug("detail")
    assert logger.entries == [
        {"level": "info", "text": "started"},
        {"level": "success", "text": "chose"},
        {"level": "warning", "text": "careful"},
        {"level": "error", "text": "boom"},
        {"level": "debug", "text": "detail"},
    ]
    assert logger.idempotency_key == "k"


def test_capture_is_independent_of_console_level():
    # Entries accrue even though CRITICAL-level handlers emit nothing (conftest sets CRITICAL).
    logger = build_logger(segment="clips")
    logger.info("quiet but captured")
    assert logger.entries == [{"level": "info", "text": "quiet but captured"}]


def test_at_stage_shares_capture_and_tags_stage():
    parent = build_logger(segment="clips")
    child = parent.at_stage("moments")
    child.info("scoring")
    assert child.stage == "moments"
    assert parent.entries == [{"level": "info", "text": "scoring"}]  # shared capture


def test_success_level_registered_between_info_and_warning():
    assert logging.INFO < SUCCESS < logging.WARNING
    assert logging.getLevelName(SUCCESS) == "SUCCESS"


# --- config / stdlib wiring ----------------------------------------------

def test_logger_for_lives_under_kinoforge_tree():
    assert logger_for("clips").name == "kinoforge.clips"
    assert logger_for("").name == "kinoforge.core"


def test_configure_floors_tree_and_env_sets_default_level(monkeypatch):
    # The tree sits at the TRACE floor; the env sets the per-request default, not the tree level.
    monkeypatch.setenv("KINOFORGE_LOG_LEVEL", "warning")
    root = configure(force=True)
    try:
        assert root.name == "kinoforge"
        assert root.level == TRACE
        assert root.propagate is False
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
        assert default_level() == logging.WARNING
    finally:
        monkeypatch.setenv("KINOFORGE_LOG_LEVEL", "CRITICAL")
        configure(force=True)


def test_with_context_carries_bound_logger_into_threads():
    from concurrent.futures import ThreadPoolExecutor

    logger = build_logger(segment="clips", level="debug")
    token = bind(logger)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Wrapper built here on the main thread carries the bound logger into the worker.
            assert pool.submit(with_context(lambda: active() is logger)).result() is True
            # A bare submit does not: the worker thread never inherited the ContextVar.
            assert pool.submit(lambda: active() is logger).result() is False
    finally:
        reset(token)


def test_request_level_gates_console_but_not_capture(monkeypatch):
    # Default (error) suppresses an info line on the console; the capture keeps it regardless.
    monkeypatch.setenv("KINOFORGE_LOG_LEVEL", "error")
    quiet = build_logger(segment="clips")
    quiet.info("hidden on console")
    assert quiet.entries == [{"level": "info", "text": "hidden on console"}]
    # A request asking for debug lowers its own threshold; trace stays out of the capture.
    loud = build_logger(segment="clips", level="debug")
    loud.debug("shown")
    loud.trace("finest")
    assert {"level": "debug", "text": "shown"} in loud.entries
    assert all(entry["level"] != "trace" for entry in loud.entries)


def test_file_sink_writes_json_with_correlation(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "events.jsonl"
    monkeypatch.setenv("KINOFORGE_LOG_LEVEL", "INFO")
    monkeypatch.setenv("KINOFORGE_LOG_FILE", str(path))
    configure(force=True)
    try:
        build_logger(job_id="p1", segment="clips", idempotency_key="k1").success("done", kept=2)
        for handler in logging.getLogger("kinoforge").handlers:
            handler.flush()
        row = json.loads(path.read_text().splitlines()[0])
        assert row["level"] == "success"
        assert row["message"] == "done"
        assert row["job_id"] == "p1"
        assert row["idempotency_key"] == "k1"
        assert row["fields"] == {"kept": 2}
    finally:
        monkeypatch.delenv("KINOFORGE_LOG_FILE", raising=False)
        monkeypatch.setenv("KINOFORGE_LOG_LEVEL", "CRITICAL")
        configure(force=True)


# --- ambient context ------------------------------------------------------

def test_current_is_none_by_default():
    assert current() is None


def test_bind_current_reset_roundtrip():
    logger = build_logger()
    token = bind(logger)
    try:
        assert current() is logger
    finally:
        reset(token)
    assert current() is None


def test_active_returns_bound_then_falls_back():
    bound = build_logger()
    token = bind(bound)
    try:
        assert active() is bound
    finally:
        reset(token)
    assert isinstance(active(), KinoLogger)  # lazy env fallback


# --- @logged decorator ----------------------------------------------------

def test_logged_is_transparent_without_a_bound_logger():
    @logged
    def add(a, b):
        return a + b

    assert current() is None
    assert add(2, 3) == 5  # no logger bound: runs untouched


def test_logged_records_enter_and_exit_on_bound_logger():
    logger = build_logger(segment="clips")
    token = bind(logger)
    try:
        @logged
        def mul(a, b):
            return a * b

        assert mul(3, 4) == 12
    finally:
        reset(token)
    texts = [e["text"] for e in logger.entries]
    assert any(t.startswith("enter ") and t.endswith("mul") for t in texts)
    assert any(t.startswith("exit ") and t.endswith("mul") for t in texts)


def test_logged_records_and_reraises_on_exception():
    logger = build_logger(segment="clips")
    token = bind(logger)
    raised = False
    try:
        @logged
        def boom():
            raise ValueError("nope")

        try:
            boom()
        except ValueError:
            raised = True
    finally:
        reset(token)
    assert raised
    assert any(
        e["level"] == "error" and e["text"].startswith("raised ") and "boom: nope" in e["text"]
        for e in logger.entries
    )


def test_logged_info_level_option():
    logger = build_logger(segment="clips")
    token = bind(logger)
    try:
        @logged(level="info")
        def ping():
            return "pong"

        ping()
    finally:
        reset(token)
    assert any(
        e["level"] == "info" and e["text"].startswith("enter ") and e["text"].endswith("ping")
        for e in logger.entries
    )
