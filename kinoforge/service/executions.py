from __future__ import annotations

import threading


class ExecutionConflict(RuntimeError):
    pass


class ExecutionControl:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


class ExecutionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, ExecutionControl] = {}

    def begin(self, execution_id: str) -> ExecutionControl:
        with self._lock:
            if execution_id in self._active:
                raise ExecutionConflict(f"execution is already active: {execution_id}")
            control = ExecutionControl(execution_id)
            self._active[execution_id] = control
            return control

    def finish(self, execution_id: str) -> None:
        with self._lock:
            self._active.pop(execution_id, None)

    def cancel(self, execution_id: str) -> bool:
        with self._lock:
            control = self._active.get(execution_id)
            if control is None:
                return False
            control.cancel()
            return True


executions = ExecutionRegistry()
