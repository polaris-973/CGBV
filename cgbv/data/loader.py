from __future__ import annotations

from cgbv.data.base import DataSample
from cgbv.data.adapters import folio, proofwriter, prontoqa, ar_lsat, logical_deduction, proverqa

_ADAPTERS = {
    "folio": folio.load,
    "proofwriter": proofwriter.load,
    "prontoqa": prontoqa.load,
    "ar_lsat": ar_lsat.load,
    "logical_deduction": logical_deduction.load,
    "proverqa": proverqa.load,
}


def load_dataset(name: str, split: str, path: str, limit: int | None = None) -> list[DataSample]:
    """Unified entry point for loading any supported dataset."""
    key = name.lower().replace("-", "_")
    loader_fn = _ADAPTERS.get(key)
    if loader_fn is None:
        raise ValueError(f"Unknown dataset: {name!r}. Supported: {list(_ADAPTERS)}")
    return loader_fn(path, split, limit=limit)
