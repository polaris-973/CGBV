"""
Sample context — propagates sample_id and per-sample log directory
through asyncio tasks via ContextVar.

Each async task (one per sample) gets its own copy of the context when
asyncio.gather() wraps coroutines in Tasks, so there is no cross-sample
contamination even under high concurrency.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from pathlib import Path

_sample_id: ContextVar[str] = ContextVar("cgbv_sample_id", default="")
_sample_display_id: ContextVar[str] = ContextVar("cgbv_sample_display_id", default="")
_log_dir: ContextVar[Path | None] = ContextVar("cgbv_log_dir", default=None)


def set_sample(sample_id: str, log_dir: Path | None = None, display_id: str | None = None) -> None:
    """Set the current sample context (call once per task, before pipeline.run)."""
    _sample_id.set(sample_id)
    _sample_display_id.set(display_id or sample_id)
    _log_dir.set(log_dir)


def get_sample_id() -> str:
    return _sample_id.get()


def get_sample_display_id() -> str:
    return _sample_display_id.get()


def get_log_dir() -> Path | None:
    return _log_dir.get()


class SampleContextFilter(logging.Filter):
    """Inject sample_id into every LogRecord (no-op when outside a sample task)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.sample_id = _sample_id.get()  # type: ignore[attr-defined]
        record.sample_display_id = _sample_display_id.get()  # type: ignore[attr-defined]
        return True
