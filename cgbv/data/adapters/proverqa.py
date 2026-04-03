from __future__ import annotations

import json
import re
from pathlib import Path

from cgbv.data.base import DataSample


def _split_context(context: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    return [s.strip() for s in sentences if s.strip()]


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
        label = raw.get("answer", "A")
        samples.append(DataSample(
            id=raw.get("id", str(len(samples))),
            dataset="proverqa",
            premises=premises,
            conclusion=conclusion,
            label=label,
            task_type="multi_choice",
            options=options,
            raw=raw,
        ))
        if limit and len(samples) >= limit:
            break

    return samples
