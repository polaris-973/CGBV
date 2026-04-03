from __future__ import annotations

import json
import re
from pathlib import Path

from cgbv.data.base import DataSample


def _split_context(context: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_conclusion(question: str) -> str:
    match = re.search(r'(?:true|false)\?\s*(.+)', question, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return question.strip()


def _answer_to_label(answer: str, options: list[str]) -> str:
    idx = ord(answer.upper()) - ord('A')
    if 0 <= idx < len(options):
        match = re.search(r'\)\s*(\w+)', options[idx])
        if match:
            return match.group(1)
    return answer


def load(dataset_path: str, split: str, limit: int | None = None) -> list[DataSample]:
    split_map = {"dev": "dev.json", "validation": "dev.json", "train": "dev.json"}
    filename = split_map.get(split, "dev.json")
    path = Path(dataset_path) / "ProntoQA" / filename

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
            dataset="prontoqa",
            premises=premises,
            conclusion=conclusion,
            label=label,
            options=options,
            raw=raw,
        ))
        if limit and len(samples) >= limit:
            break

    return samples
