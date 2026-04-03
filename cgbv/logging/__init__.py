"""
cgbv.logging — Rich-based structured logging for CGBV experiments.

Public API
----------
setup_logging(level, results_dir) -> Console
    Call once at startup.  Returns the shared Rich Console.

set_sample(sample_id, log_dir=None)
    Set the ContextVar for the current sample task.  Must be called
    before pipeline.run() so all child coroutines inherit the context.

update_phase(phase_key, round_info="")
    Update the progress display for the current sample (reads sample_id
    from ContextVar automatically).  phase_key ∈ PHASE_LABELS keys.

complete_sample(success=True)
    Remove sample from progress table and advance the overall bar.

register_progress(p)
    Register the ExperimentProgress instance so update_phase /
    complete_sample know where to send updates.

ExperimentProgress
    Rich Live context manager — use as ``with ExperimentProgress(...) as prog:``.
"""
from __future__ import annotations

from pathlib import Path

from cgbv.logging.context import get_sample_id, set_sample
from cgbv.logging.progress import ExperimentProgress
from cgbv.logging.setup import setup_logging

# Module-level progress singleton — set by ExperimentRunner
_progress: ExperimentProgress | None = None


def register_progress(p: ExperimentProgress | None) -> None:
    """Register (or clear) the active ExperimentProgress instance."""
    global _progress
    _progress = p


def update_phase(phase_key: str, round_info: str = "") -> None:
    """
    Update the current sample's phase in the progress display.

    No-op when called outside a sample task (no sample_id in context)
    or before register_progress() has been called.
    """
    if _progress is None:
        return
    sid = get_sample_id()
    if sid:
        _progress.update_phase(sid, phase_key, round_info)


def complete_sample(*, success: bool = True) -> None:
    """Advance the overall progress bar and remove sample from the active table."""
    if _progress is None:
        return
    sid = get_sample_id()
    if sid:
        _progress.complete_sample(sid, success=success)


__all__ = [
    "setup_logging",
    "set_sample",
    "update_phase",
    "complete_sample",
    "register_progress",
    "ExperimentProgress",
]
