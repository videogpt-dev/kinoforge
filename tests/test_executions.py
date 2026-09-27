"""Execution registry: single-flight begin, cooperative cancel, cleanup."""

from __future__ import annotations

import pytest

from kinoforge.service.executions import ExecutionConflict, ExecutionRegistry


def test_begin_returns_uncancelled_control():
    control = ExecutionRegistry().begin("job1")
    assert control.execution_id == "job1"
    assert control.is_cancelled() is False


def test_duplicate_begin_conflicts():
    reg = ExecutionRegistry()
    reg.begin("job1")
    with pytest.raises(ExecutionConflict):
        reg.begin("job1")


def test_finish_frees_the_id_for_reuse():
    reg = ExecutionRegistry()
    reg.begin("job1")
    reg.finish("job1")
    reg.begin("job1")  # no conflict after finish


def test_cancel_sets_the_flag_and_reports_true():
    reg = ExecutionRegistry()
    control = reg.begin("job1")
    assert reg.cancel("job1") is True
    assert control.is_cancelled() is True


def test_cancel_unknown_id_returns_false():
    assert ExecutionRegistry().cancel("ghost") is False


def test_finish_unknown_id_is_noop():
    ExecutionRegistry().finish("ghost")  # must not raise
