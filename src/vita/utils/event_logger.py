import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

from vita.utils.utils import get_now


_current_logger: ContextVar[Optional["EventLogger"]] = ContextVar(
    "current_event_logger", default=None
)
_event_context: ContextVar[dict[str, Any]] = ContextVar("event_context", default={})


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v) for v in value]
    if hasattr(value, "model_dump"):
        return _safe_json_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _safe_json_value(value.__dict__)
    return str(value)


class EventLogger:
    def __init__(self, file_path: Path, run_id: str):
        self.file_path = file_path
        self.run_id = run_id
        self._lock = threading.Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **payload: Any) -> None:
        event = {
            "timestamp": get_now("%Y-%m-%d %H:%M:%S"),
            "run_id": self.run_id,
            "event_type": event_type,
        }
        event.update(_event_context.get({}))
        event.update({k: _safe_json_value(v) for k, v in payload.items()})
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(event, ensure_ascii=False) + "\n")


@contextmanager
def use_event_logger(logger: Optional[EventLogger]):
    token = _current_logger.set(logger)
    try:
        yield
    finally:
        _current_logger.reset(token)


@contextmanager
def bind_event_context(**context: Any):
    merged = dict(_event_context.get({}))
    for key, value in context.items():
        if value is not None:
            merged[key] = _safe_json_value(value)
    token = _event_context.set(merged)
    try:
        yield
    finally:
        _event_context.reset(token)


def get_event_logger() -> Optional[EventLogger]:
    return _current_logger.get()


def log_event(event_type: str, **payload: Any) -> None:
    logger = get_event_logger()
    if logger is None:
        return
    logger.log(event_type, **payload)
