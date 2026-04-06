from __future__ import annotations

import json
import re
from pathlib import Path

from cgbv.data.base import DataSample


def _split_context(context: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    return [s.strip() for s in sentences if s.strip()]


def _answer_to_label(answer: str, options: list[str]) -> str:
    """Convert answer letter (A/B/C) to label string.

    ProverQA options are always ["A) True", "B) False", "C) Uncertain"],
    so A→true, B→false, C→uncertain.
    """
    idx = ord(answer.upper()) - ord('A')
    if 0 <= idx < len(options):
        option_text = options[idx]
        # Extract "True", "False", or "Uncertain" from "A) True"
        match = re.search(r'\)\s*(\w+)', option_text)
        if match:
            return match.group(1).lower()
    return answer.lower()


def load(dataset_path: str, split: str, limit: int | None = None) -> list[DataSample]:
    # ProverQA has easy/medium/hard splits
    split_map = {
        "easy": "easy.json",
        "medium": "medium.json",
        "hard": "hard.json",
        "train": "easy.json",
        "validation": "easy.json",
    }
    filename = split_map.get(split, f"{split}.json")
    path = Path(dataset_path) / "ProverQA" / filename

    with open(path) as f:
        raw_list = json.load(f)

    samples: list[DataSample] = []
    for raw in raw_list:
        premises = _split_context(raw.get("context", ""))
        conclusion = raw.get("question", "").strip()
        options = raw.get("options", [])
        answer = raw.get("answer", "A")
        # Convert A/B/C to true/false/uncertain
        label = _answer_to_label(answer, options)
        samples.append(DataSample(
            id=str(raw.get("id", len(samples))),
            dataset="proverqa",
            premises=premises,
            conclusion=conclusion,
            label=label,
            task_type="three_class",  # ProverQA is three-class, not multi-choice
            options=options,
        ))
        if limit and len(samples) >= limit:
            break

    return samples
