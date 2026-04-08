from __future__ import annotations

from naive_llm.prompting import normalize_label, parse_prediction


def test_parse_prediction_prefers_final_answer_line() -> None:
    parsed = parse_prediction(
        "Reasoning: the premises support the claim.\nFinal answer: true"
    )
    assert parsed.verdict == "true"
    assert parsed.parse_status == "parsed"


def test_parse_prediction_maps_unknown_to_uncertain() -> None:
    parsed = parse_prediction(
        "I cannot decide this from the premises alone.\nFinal answer: unknown"
    )
    assert parsed.verdict == "uncertain"


def test_normalize_label_handles_dataset_synonyms() -> None:
    assert normalize_label("Unknown") == "uncertain"
    assert normalize_label("Refuted") == "false"
    assert normalize_label("Entailed") == "true"
