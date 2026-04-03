"""
Logging setup for CGBV experiments.

Call setup_logging() once at startup (in main.py) before any other
logging occurs.  It returns the shared Rich Console that should be
passed to ExperimentProgress so both the log output and the progress
panel use the same terminal stream.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from cgbv.logging.context import SampleContextFilter
from cgbv.logging.handlers import JsonLineHandler, SampleFileRouter

# Third-party loggers whose output is only useful at WARNING+
_QUIET_LOGGERS = ("httpx", "httpcore", "openai", "z3", "asyncio")


class _CGBVRichHandler(RichHandler):
    """
    RichHandler subclass that prepends a human-readable sample label to
    terminal messages when the record was emitted from inside a sample task.
    """

    def render_message(  # type: ignore[override]
        self,
        record: logging.LogRecord,
        message: str,
    ) -> "ConsoleRenderable":  # noqa: F821
        sid = getattr(record, "sample_display_id", "") or getattr(record, "sample_id", "")
        if sid:
            from rich.text import Text
            prefix = Text(f"[{sid}] ", style="dim cyan")
            body = (
                Text.from_markup(message)
                if self.markup
                else Text(message)
            )
            return prefix + body
        return super().render_message(record, message)


def setup_logging(level: str, results_dir: Path) -> Console:
    """
    Configure the CGBV logging system.

    Handlers installed on the root logger:
    ┌──────────────────┬───────┬──────────────────────────────────────┐
    │ Handler          │ Level │ Destination                          │
    ├──────────────────┼───────┼──────────────────────────────────────┤
    │ _CGBVRichHandler │ INFO  │ Terminal (Rich, colours, [sample])   │
    │ JsonLineHandler  │ DEBUG │ results_dir/experiment.jsonl         │
    │ SampleFileRouter │ DEBUG │ results_dir/.../run.log (per-sample) │
    └──────────────────┴───────┴──────────────────────────────────────┘

    Args:
        level:       Terminal log level string ("DEBUG" / "INFO" / "WARNING").
        results_dir: Experiment output directory — experiment.jsonl is written here.

    Returns:
        The shared Rich Console instance (pass to ExperimentProgress).
    """
    console = Console(highlight=False, soft_wrap=True)
    sample_filter = SampleContextFilter()
    terminal_level = getattr(logging, level.upper(), logging.INFO)

    # --- Rich terminal handler ---
    rich_handler = _CGBVRichHandler(
        console=console,
        level=terminal_level,
        rich_tracebacks=True,
        show_path=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.addFilter(sample_filter)

    # --- Global structured JSONL ---
    results_dir.mkdir(parents=True, exist_ok=True)
    json_handler = JsonLineHandler(results_dir / "experiment.jsonl", level=logging.DEBUG)
    json_handler.addFilter(sample_filter)

    # --- Per-sample plain-text file ---
    sample_router = SampleFileRouter(level=logging.DEBUG)
    sample_router.addFilter(sample_filter)

    # --- Root logger ---
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(rich_handler)
    root.addHandler(json_handler)
    root.addHandler(sample_router)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return console
