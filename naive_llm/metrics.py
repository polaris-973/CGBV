from __future__ import annotations

from cgbv.data.base import DataSample

from naive_llm.prompting import normalize_label

_LABELS: tuple[str, ...] = ("true", "false", "uncertain")
_MISSING = "__missing__"


def compute_metrics(results: list[dict], samples: list[DataSample]) -> dict:
    total = len(samples)
    sample_map = {sample.id: sample for sample in samples}
    result_map = {result["sample_id"]: result for result in results if "sample_id" in result}

    correct = 0
    successful = 0
    parse_errors = 0
    runtime_errors = 0
    binary_total = 0
    binary_correct = 0
    uncertain_total = 0
    uncertain_correct = 0

    label_totals = {label: 0 for label in _LABELS}
    label_correct = {label: 0 for label in _LABELS}
    confusion_matrix = {
        gold: {pred: 0 for pred in (*_LABELS, _MISSING)}
        for gold in _LABELS
    }

    error_sample_ids: list[str] = []
    parse_error_sample_ids: list[str] = []
    wrong_sample_ids: list[str] = []
    reasoning_error_sample_ids: list[str] = []
    correct_sample_ids: list[str] = []
    error_details: list[dict] = []

    for sample_id, sample in sample_map.items():
        gold = normalize_label(sample.label) or str(sample.label).strip().lower()
        label_totals.setdefault(gold, 0)
        label_correct.setdefault(gold, 0)
        label_totals[gold] += 1

        result = result_map.get(sample_id)
        if result is None:
            error_sample_ids.append(sample_id)
            runtime_errors += 1
            error_details.append(
                {
                    "sample_id": sample_id,
                    "execution_status": "missing",
                    "parse_status": "missing",
                    "error": "Missing result.json",
                }
            )
            confusion_matrix.setdefault(gold, {}).setdefault(_MISSING, 0)
            confusion_matrix[gold][_MISSING] += 1
            if gold in ("true", "false"):
                binary_total += 1
            if gold == "uncertain":
                uncertain_total += 1
            continue

        status = result.get("execution_status", "missing")
        parse_status = result.get("parse_status", "")
        prediction = normalize_label(result.get("prediction"))
        success = status == "success" and prediction is not None

        if success:
            successful += 1
        else:
            error_sample_ids.append(sample_id)
            error_details.append(
                {
                    "sample_id": sample_id,
                    "execution_status": status,
                    "parse_status": parse_status,
                    "error": result.get("error"),
                }
            )
            if status == "parse_error":
                parse_errors += 1
                parse_error_sample_ids.append(sample_id)
            else:
                runtime_errors += 1

        predicted_label = prediction if success else _MISSING
        confusion_matrix.setdefault(gold, {}).setdefault(predicted_label, 0)
        confusion_matrix[gold][predicted_label] += 1

        if gold in ("true", "false"):
            binary_total += 1
        if gold == "uncertain":
            uncertain_total += 1

        if success and prediction == gold:
            correct += 1
            label_correct[gold] = label_correct.get(gold, 0) + 1
            correct_sample_ids.append(sample_id)
            if gold in ("true", "false"):
                binary_correct += 1
            if gold == "uncertain":
                uncertain_correct += 1
        elif success:
            wrong_sample_ids.append(sample_id)
            reasoning_error_sample_ids.append(sample_id)

    accuracy_by_label = {
        label: round(label_correct.get(label, 0) / count, 4) if count else 0.0
        for label, count in label_totals.items()
    }

    return {
        "total_samples": total,
        "successful_samples": successful,
        "correct_samples": correct,
        "error_samples": len(error_sample_ids),
        "accuracy": round(correct / total, 4) if total else 0.0,
        "conditional_accuracy": round(correct / successful, 4) if successful else 0.0,
        "completion_rate": round(successful / total, 4) if total else 0.0,
        "binary_accuracy": round(binary_correct / binary_total, 4) if binary_total else 0.0,
        "uncertain_recall": round(uncertain_correct / uncertain_total, 4) if uncertain_total else 0.0,
        "parse_error_rate": round(parse_errors / total, 4) if total else 0.0,
        "runtime_error_rate": round(runtime_errors / total, 4) if total else 0.0,
        "accuracy_by_label": accuracy_by_label,
        "label_totals": label_totals,
        "confusion_matrix": confusion_matrix,
        "sample_id_audit": {
            "error_sample_ids": error_sample_ids,
            "parse_error_sample_ids": parse_error_sample_ids,
            "wrong_sample_ids": wrong_sample_ids,
            "reasoning_error_sample_ids": reasoning_error_sample_ids,
            "correct_sample_ids": correct_sample_ids,
            "error_details": error_details,
        },
    }
