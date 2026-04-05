from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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

    def __post_init__(self) -> None:
        if not self.display_id:
            self.display_id = self.id
