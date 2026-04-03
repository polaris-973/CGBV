from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Datasets whose labels are A/B/C/D choices — the CGBV pipeline treats all tasks
# as binary entailment (Entailed / Not Entailed), so these datasets cannot be
# evaluated for accuracy.  They are loaded but will trigger a warning at runtime.
MULTI_CHOICE_DATASETS: frozenset[str] = frozenset({
    "ar_lsat",
    "logical_deduction",
    "proverqa",
})

THREE_CLASS_DATASETS: frozenset[str] = frozenset({"folio", "proofwriter"})


@dataclass
class DataSample:
    id: str
    dataset: str
    premises: list[str]         # NL premises (one sentence per element)
    conclusion: str             # NL conclusion
    label: str                  # ground truth: True | False | Uncertain | A | B | C ...
    task_type: str = "entailment"   # "entailment" | "multi_choice"
    options: Optional[list[str]] = None
    display_id: Optional[str] = None
    source_id: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.display_id:
            self.display_id = self.id
        if not self.source_id:
            self.source_id = self.id
