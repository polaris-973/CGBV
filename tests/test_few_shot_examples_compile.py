from __future__ import annotations

import json
from pathlib import Path

import yaml

from cgbv.core.logic_compiler import compile_theory_dsl


def _load_examples(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        examples = raw.get("examples", [])
        if isinstance(examples, list):
            return examples
    return []


def test_all_phase1_few_shot_examples_compile() -> None:
    few_shot_dir = Path("cgbv/prompts/few_shot")
    for yaml_path in sorted(few_shot_dir.glob("*.yaml")):
        for ex in _load_examples(yaml_path):
            premises = ex.get("premises", [])
            conclusion = ex.get("conclusion", "")
            payload = json.loads(ex.get("dsl", "{}"))
            compiled = compile_theory_dsl(
                payload=payload,
                premises_nl=premises,
                conclusion_nl=conclusion,
            )
            assert len(compiled.premises) == len(premises)


def test_phase1_few_shot_ids_and_order_are_frozen() -> None:
    expected = {
        "folio.yaml": ["21", "1128", "1338"],
        "prontoqa.yaml": ["ProntoQA_6", "ProntoQA_12", "ProntoQA_4"],
        "proofwriter.yaml": [
            "ProofWriter_AttNoneg-OWA-D5-355_Q3",
            "ProofWriter_AttNeg-OWA-D5-951_Q4",
            "ProofWriter_RelNoneg-OWA-D5-774_Q11",
        ],
        "proverqa.yaml": ["418", "267", "4"],
    }

    few_shot_dir = Path("cgbv/prompts/few_shot")
    for filename, expected_ids in expected.items():
        examples = _load_examples(few_shot_dir / filename)
        actual_ids = [str(ex.get("source_id", "")) for ex in examples]
        assert actual_ids == expected_ids


def test_phase1_few_shot_uses_minimal_logic_schema() -> None:
    few_shot_dir = Path("cgbv/prompts/few_shot")
    for yaml_path in sorted(few_shot_dir.glob("*.yaml")):
        for ex in _load_examples(yaml_path):
            payload = json.loads(ex.get("dsl", "{}"))
            assert set(payload.keys()) == {"symbols", "sentences", "query", "background"}
            symbols = payload.get("symbols", {})
            assert set(symbols.keys()) == {"sorts", "predicates", "functions", "constants"}
            for sentence in payload.get("sentences", []):
                assert set(sentence.keys()) == {"nl", "logic"}
                assert isinstance(sentence.get("logic"), str)
            query = payload.get("query", {})
            assert set(query.keys()) == {"nl", "logic"}
            assert isinstance(query.get("logic"), str)
            assert isinstance(payload.get("background", []), list)
            assert all(isinstance(item, str) for item in payload.get("background", []))
