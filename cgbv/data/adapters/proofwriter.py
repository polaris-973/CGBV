from __future__ import annotations

import json
import re
from pathlib import Path

from cgbv.data.base import DataSample


def _split_context(context: str) -> list[str]:
    """Split context into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_conclusion(question: str) -> str:
    """Extract the statement being evaluated from the question."""
    # "Based on the above information, is the following statement true, false, or unknown? X"
    match = re.search(r'(?:true|false|unknown)\?\s*(.+)', question, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return question.strip()


def _answer_to_label(answer: str, options: list[str]) -> str:
    """Convert answer letter (A/B/C) to label string (lowercase).

    Normalizes "Unknown" to "uncertain" for consistency with FOLIO and ProverQA.
    """
    idx = ord(answer.upper()) - ord('A')
    if 0 <= idx < len(options):
        option_text = options[idx]
        # Extract "True", "False", or "Unknown" from "A) True"
        match = re.search(r'\)\s*(\w+)', option_text)
        if match:
            label = match.group(1).lower()
            # Normalize "unknown" to "uncertain" for consistency
            if label == "unknown":
                label = "uncertain"
            return label
    return answer.lower()


def load(dataset_path: str, split: str, limit: int | None = None) -> list[DataSample]:
    split_map = {"train": "train.json", "dev": "dev.json", "test": "test.json", "validation": "dev.json"}
    filename = split_map.get(split, f"{split}.json")
    path = Path(dataset_path) / "ProofWriter" / filename

    with open(path) as f:
        raw_list = json.load(f)

    samples: list[DataSample] = []
    for raw in raw_list:
        premises = _split_context(raw["context"])
        conclusion = _extract_conclusion(raw["question"])
        options = raw.get("options", [])
        label = _answer_to_label(raw["answer"], options)
        samples.append(DataSample(
            id=raw["id"],
            dataset="proofwriter",
            premises=premises,
            conclusion=conclusion,
            label=label,
            task_type="three_class",
            options=options,
        ))
        if limit and len(samples) >= limit:
            break

    return samples
