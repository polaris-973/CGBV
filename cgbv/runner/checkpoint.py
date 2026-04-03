from __future__ import annotations

import json
import logging
from pathlib import Path

from cgbv.data.base import DataSample

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint-based experiment resumption.

    A sample is considered "done" if its result.json exists and is valid JSON.
    On restart, already-done samples are skipped.
    """

    def __init__(self, results_dir: str, run_id: str):
        self.base = Path(results_dir) / run_id

    def is_done(self, sample: DataSample) -> bool:
        result_path = self.base / sample.dataset / sample.id / "result.json"
        if not result_path.exists():
            return False
        try:
            with open(result_path) as f:
                data = json.load(f)
            # A result is valid if it has a verdict field
            return "verdict" in data
        except Exception:
            return False

    def filter_pending(self, samples: list[DataSample]) -> list[DataSample]:
        """Return only samples that have not been completed yet."""
        pending = [s for s in samples if not self.is_done(s)]
        done_count = len(samples) - len(pending)
        if done_count:
            logger.info(
                "Checkpoint: skipping %d already-completed samples, %d remaining",
                done_count, len(pending),
            )
        return pending

    def load_all_results(self, dataset: str | None = None) -> list[dict]:
        """Load all completed result.json files for aggregation."""
        results = []
        search_root = self.base / dataset if dataset else self.base
        for path in search_root.rglob("result.json"):
            try:
                with open(path) as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.warning("Could not load result %s: %s", path, e)
        return results
