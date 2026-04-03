from __future__ import annotations

from cgbv.data.base import DataSample


def compute_metrics(
    results: list[dict],
    samples: list[DataSample],
) -> dict:
    """
    Compute all CGBV evaluation metrics.

    Metrics:
      1. General Accuracy          — final verdict matches label
      2. Binary Accuracy           — True/False samples only
      3. Uncertain Recall          — Uncertain samples only
      4. Verification Precision    — among verified samples, fraction correct
      5. Verification Coverage     — fraction of samples that got verified
      6. Mismatch Detection Precision — among samples with mismatches, fraction initially wrong
      7. Mismatch Detection Recall    — fraction of initially-wrong samples that had mismatches
      8. Repair Parse Success Rate    — fraction of repair rounds where LLM produced valid z3 expr
      9. Repair Local Fix Rate        — fraction of mismatches that passed local acceptance check
     10. Repair Verdict Recovery Rate — among initially-wrong repaired samples, fraction fixed
     11. Repair Regression Rate       — among initially-correct repaired samples, fraction broken
    """
    label_map: dict[str, str] = {s.id: _normalise_label(s.label) for s in samples}
    total = len(samples)

    if total == 0:
        return _empty_metrics()

    result_map: dict[str, dict] = {r["sample_id"]: r for r in results if "sample_id" in r}

    # Counters
    correct = 0
    verified = 0
    verified_correct = 0
    has_mismatch = 0
    mismatch_wrong = 0          # mismatch AND initially wrong (numerator for both prec/recall)
    wrong_initial = 0           # initially wrong (denominator for recall)
    repair_formula_attempts = 0
    repair_formula_successes = 0
    repair_mismatch_attempts = 0
    repair_local_validations = 0
    repair_recovery_attempted = 0
    repair_recovery_correct = 0
    repair_regression_attempted = 0
    repair_regressions = 0
    binary_correct = 0
    binary_total = 0
    uncertain_correct = 0
    uncertain_total = 0

    for sample in samples:
        r = result_map.get(sample.id)
        if r is None:
            continue

        label = label_map[sample.id]
        verdict_final = _normalise_label(r.get("verdict", ""))
        verdict_pre = _normalise_label(r.get("verdict_initial", r.get("verdict", "")))
        is_verified = r.get("verified", False)
        rounds = r.get("rounds", [])

        final_correct = (verdict_final == label)
        initial_correct = (verdict_pre == label)

        if final_correct:
            correct += 1
        if not initial_correct:
            wrong_initial += 1

        if is_verified:
            verified += 1
            if final_correct:
                verified_correct += 1

        # Mismatch detection uses verdict_initial throughout (both numerator and denominator)
        sample_has_mismatch = any(
            len(rnd.get("mismatches", [])) > 0
            for rnd in rounds
        )
        if sample_has_mismatch:
            has_mismatch += 1
            if not initial_correct:
                mismatch_wrong += 1

        # Repair metrics (round-level and mismatch-level)
        for rnd in rounds:
            if rnd.get("repair_attempted", False):
                repair_formula_attempts += 1
                if rnd.get("repair_success", False) and not rnd.get("repair_reverted", False):
                    repair_formula_successes += 1
                repair_mismatch_attempts += rnd.get("num_mismatches", 0)
                repair_local_validations += rnd.get("repair_local_validated", 0)

        sample_repaired = any(
            rnd.get("repair_attempted", False) for rnd in rounds
        )
        sample_repaired_kept = any(
            rnd.get("repair_attempted", False) and not rnd.get("repair_reverted", False)
            for rnd in rounds
        )

        if sample_repaired and not initial_correct:
            repair_recovery_attempted += 1
            if final_correct:
                repair_recovery_correct += 1

        if sample_repaired_kept and initial_correct:
            repair_regression_attempted += 1
            if not final_correct:
                repair_regressions += 1

        if label in ("true", "false"):
            binary_total += 1
            if final_correct:
                binary_correct += 1
        elif label == "uncertain":
            uncertain_total += 1
            if final_correct:
                uncertain_correct += 1

    def safe_div(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom > 0 else 0.0

    completed = len(result_map)

    return {
        "total_samples": total,
        "completed_samples": completed,
        # Accuracy
        "general_accuracy": safe_div(correct, total),
        "binary_accuracy": safe_div(binary_correct, binary_total),
        "uncertain_recall": safe_div(uncertain_correct, uncertain_total),
        # Verification
        "verification_precision": safe_div(verified_correct, verified),
        "verification_coverage": safe_div(verified, total),
        # Mismatch detection (both use initial verdict)
        "mismatch_detection_precision": safe_div(mismatch_wrong, has_mismatch),
        "mismatch_detection_recall": safe_div(mismatch_wrong, wrong_initial),
        # Repair metrics
        "repair_parse_success_rate": safe_div(repair_formula_successes, repair_formula_attempts),
        "repair_local_fix_rate": safe_div(repair_local_validations, repair_mismatch_attempts),
        "repair_verdict_recovery_rate": safe_div(repair_recovery_correct, repair_recovery_attempted),
        "repair_regression_rate": safe_div(repair_regressions, repair_regression_attempted),
        # Raw counts (for inspection)
        "_counts": {
            "correct": correct,
            "wrong_initial": wrong_initial,
            "verified": verified,
            "verified_correct": verified_correct,
            "has_mismatch": has_mismatch,
            "mismatch_wrong": mismatch_wrong,
            "repair_formula_attempts": repair_formula_attempts,
            "repair_formula_successes": repair_formula_successes,
            "repair_mismatch_attempts": repair_mismatch_attempts,
            "repair_local_validations": repair_local_validations,
            "repair_recovery_attempted": repair_recovery_attempted,
            "repair_recovery_correct": repair_recovery_correct,
            "repair_regression_attempted": repair_regression_attempted,
            "repair_regressions": repair_regressions,
            "binary_correct": binary_correct,
            "binary_total": binary_total,
            "uncertain_correct": uncertain_correct,
            "uncertain_total": uncertain_total,
        },
    }


def _normalise_label(label: str) -> str:
    """Normalise label/verdict strings for comparison."""
    if label is None:
        return ""
    s = str(label).strip().lower()
    if s in ("true", "entailed", "yes", "correct"):
        return "true"
    if s in ("false", "not entailed", "no", "incorrect", "refuted"):
        return "false"
    if s in ("uncertain", "unknown", "neither"):
        return "uncertain"
    return s


def _empty_metrics() -> dict:
    return {
        "total_samples": 0,
        "completed_samples": 0,
        "general_accuracy": 0.0,
        "binary_accuracy": 0.0,
        "uncertain_recall": 0.0,
        "verification_precision": 0.0,
        "verification_coverage": 0.0,
        "mismatch_detection_precision": 0.0,
        "mismatch_detection_recall": 0.0,
        "repair_parse_success_rate": 0.0,
        "repair_local_fix_rate": 0.0,
        "repair_verdict_recovery_rate": 0.0,
        "repair_regression_rate": 0.0,
        "_counts": {},
    }
