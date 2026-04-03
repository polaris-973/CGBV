"""
Custom logging handlers.

JsonLineHandler  — one JSON object per log record → global experiment.jsonl
SampleFileRouter — routes DEBUG+ records to per-sample run.log files
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path


class JsonLineHandler(logging.FileHandler):
    """
    Appends one JSON object per log record to a .jsonl file.

    Fields: ts, level, logger, sample_id, module, line, msg, [exc]
    """

    def __init__(self, path: Path, level: int = logging.DEBUG) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(path), mode="a", encoding="utf-8", delay=False)
        self.setLevel(level)
        self._exc_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry: dict = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "sample_id": getattr(record, "sample_id", ""),
                "sample_display_id": getattr(record, "sample_display_id", ""),
                "module": record.module,
                "line": record.lineno,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                entry["exc"] = self._exc_formatter.formatException(record.exc_info)
            self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


class SampleFileRouter(logging.Handler):
    """
    Routes log records to per-sample plain-text log files.

    The target directory is read from the cgbv_log_dir ContextVar on each
    emit() call, so each async task naturally routes to its own file without
    any explicit coordination.
    """

    _FMT = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
        datefmt="%H:%M:%S",
    )

    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self._handlers: dict[str, logging.FileHandler] = {}
        self._lock = threading.Lock()

    def _get_handler(self, log_dir: Path | None) -> logging.FileHandler | None:
        if log_dir is None:
            return None
        key = str(log_dir)
        with self._lock:
            if key not in self._handlers:
                log_dir.mkdir(parents=True, exist_ok=True)
                h = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
                h.setFormatter(self._FMT)
                self._handlers[key] = h
        return self._handlers[key]

    def emit(self, record: logging.LogRecord) -> None:
        from cgbv.logging.context import get_log_dir
        h = self._get_handler(get_log_dir())
        if h:
            h.emit(record)

    def close(self) -> None:
        with self._lock:
            for h in self._handlers.values():
                h.close()
            self._handlers.clear()
        super().close()
