from __future__ import annotations

import json
import re
from pathlib import Path

from cgbv.data.base import DataSample


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return text.strip("-._") or "unknown"


def load(dataset_path: str, split: str, limit: int | None = None) -> list[DataSample]:
    """Load FOLIO dataset.

    FOLIO JSONL fields:
        story_id, premises (newline-separated), premises-FOL (unused),
        conclusion, conclusion-FOL (unused), label, example_id
    """
    split_map = {
        "train": "folio_v2_train.jsonl",
        "validation": "folio_v2_validation.jsonl",
        "dev": "folio_v2_validation.jsonl",
    }
    filename = split_map.get(split, f"folio_v2_{split}.jsonl")
    path = Path(dataset_path) / "FOLIO" / filename

    samples: list[DataSample] = []
    seen_ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            premises = [p.strip() for p in raw["premises"].split("\n") if p.strip()]
            ordinal = len(samples)
            story_id = str(raw.get("story_id", f"idx{len(samples)}"))
            example_id = raw.get("example_id")
            if example_id is not None:
                uid = _slug(example_id)
                display_id = f"story:{story_id} example:{example_id}"
            else:
                uid = f"story_{_slug(story_id)}__idx_{ordinal:05d}"
                display_id = f"story:{story_id} idx:{ordinal}"
            if uid in seen_ids:
                uid = f"{uid}__dup_{ordinal:05d}"
            seen_ids.add(uid)
            samples.append(DataSample(
                id=uid,
                dataset="folio",
                premises=premises,
                conclusion=raw["conclusion"].strip(),
                label=raw["label"].lower(),  # true | false | uncertain (normalized to lowercase)
                task_type="three_class",
                options=None,
                display_id=display_id,
            ))
            if limit and len(samples) >= limit:
                break

    return samples
