"""
Rich Live progress display for CGBV experiments.

Layout (pinned at bottom of terminal):
┌──────────────────────────────────────────────────────────────────────┐
│  sample_042   Phase 3  Grounded Re-Formalization   round 2/3         │
│  sample_043   Phase 1  Formalizing                                   │
│  sample_099   Phase 5  Repair                      round 1/3         │
│                                                                       │
│  Experiment Progress  ━━━━━━━━━━━━━━━━━━━━━━░░░  42/100  02:34<03:10 │
└──────────────────────────────────────────────────────────────────────┘

Logs printed above via RichHandler naturally scroll upward while the
panel stays pinned (Rich Live handles the interleaving).
"""
from __future__ import annotations

import threading

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Phase key → (short tag, description)
PHASE_LABELS: dict[str, tuple[str, str]] = {
    "phase1": ("1/5", "Formalizing"),
    "phase2": ("2/5", "Boundary Witness"),
    "phase3": ("3/5", "Grounded Re-Formalization"),
    "phase4": ("4/5", "Cross-Granularity Check"),
    "phase5": ("5/5", "Repair"),
    "done":   ("✓",   "Done"),
    "error":  ("✗",   "Error"),
    "start":  ("…",   "Starting"),
}


class ExperimentProgress:
    """
    Rich Live progress panel.

    Usage::

        with ExperimentProgress(total=100, console=console) as prog:
            cgbv_log.register_progress(prog)
            ...
    """

    def __init__(self, total: int, console: Console) -> None:
        self._total = total
        self._console = console
        self._lock = threading.Lock()
        self._active: dict[str, tuple[str, str, str]] = {}   # sample_key → (display_id, phase_key, round_info)
        self._started = False

        self._progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold]Experiment Progress"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]•"),
            TimeElapsedColumn(),
            TextColumn("[dim]<"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self._task_id = self._progress_bar.add_task("", total=total)
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=6,
            transient=False,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> Panel:
        rows: list = []

        if self._active:
            table = Table.grid(padding=(0, 1))
            table.add_column(no_wrap=True, min_width=26, style="cyan")   # sample_id
            table.add_column(no_wrap=True, min_width=4,  style="bold")   # phase tag
            table.add_column(no_wrap=True)                               # description + round
            for display_id, phase_key, round_info in self._active.values():
                tag, desc = PHASE_LABELS.get(phase_key, ("?", phase_key))
                if phase_key in ("done",):
                    tag_markup = f"[bold green]{tag}[/]"
                elif phase_key in ("error",):
                    tag_markup = f"[bold red]{tag}[/]"
                else:
                    tag_markup = f"[bold cyan]{tag}[/]"
                table.add_row(
                    display_id,
                    tag_markup,
                    f"{desc}  [dim]{round_info}[/]" if round_info else desc,
                )
            rows.append(table)
            rows.append(Rule(style="bright_black"))

        rows.append(self._progress_bar)
        return Panel(Group(*rows), border_style="bright_black", padding=(0, 1))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentProgress":
        self._live.start()
        self._started = True
        return self

    def __exit__(self, *_: object) -> None:
        self._live.stop()
        self._started = False

    # ------------------------------------------------------------------
    # Public update API (thread-safe, safe to call before __enter__)
    # ------------------------------------------------------------------

    def start_sample(self, sample_key: str, display_id: str | None = None) -> None:
        with self._lock:
            self._active[sample_key] = (display_id or sample_key, "start", "")
            if self._started:
                self._live.update(self._render())

    def update_phase(
        self,
        sample_key: str,
        phase_key: str,
        round_info: str = "",
    ) -> None:
        with self._lock:
            display_id = self._active.get(sample_key, (sample_key, "start", ""))[0]
            self._active[sample_key] = (display_id, phase_key, round_info)
            if self._started:
                self._live.update(self._render())

    def complete_sample(self, sample_key: str, *, success: bool = True) -> None:
        with self._lock:
            self._active.pop(sample_key, None)
            self._progress_bar.advance(self._task_id)
            if self._started:
                self._live.update(self._render())
