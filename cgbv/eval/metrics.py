from __future__ import annotations

from cgbv.data.base import DataSample


def _is_successful_result(result: dict | None) -> bool:
    """Return True iff *result* contains a meaningful semantic verdict."""
    if result is None:
        return False
    exec_status = result.get("execution_status", "")
    if exec_status:
        return exec_status == "success"
    # Legacy results without execution_status field.
    verdict_raw = str(result.get("verdict", "")).strip().lower()
    return (
        result.get("error") is None
        and verdict_raw not in ("unknown", "")
    )


def _read_verdicts(result: dict) -> tuple[str, str, str]:
    """
    Extract the three-level verdict tuple from a result dict.

    Returns (verdict_pre_bridge, verdict_post_bridge, verdict_final).

    Backward-compat fallback chain:
      - verdict_pre_bridge: new field name → legacy verdict_initial → verdict
      - verdict_post_bridge: new/old field → verdict_pre_bridge fallback → verdict
      - verdict_final: verdict field
    """
    raw_pre = result.get("verdict_pre_bridge") or result.get("verdict_initial") or result.get("verdict", "")
    raw_post = result.get("verdict_post_bridge") or raw_pre
    raw_final = result.get("verdict", "")
    return (
        _normalise_label(raw_pre),
        _normalise_label(raw_post),
        _normalise_label(raw_final),
    )


def compute_cgbv_repair_audit(
    results: list[dict],
    samples: list[DataSample],
) -> dict:
    """
    Summarise sample-level verdict recoveries caused by explicit CGBV repairs.

    Verdict level semantics:
      verdict_pre_bridge  = raw Phase 1 output before Phase 1.5
      verdict_post_bridge = after Phase 1.5 bridge repair, before Phase 5 loop
      verdict_final       = after all repairs

    A sample counts as "repaired by CGBV" iff all of the following hold:
      1. The pre-bridge verdict is wrong  (eligibility — bridge had something to fix).
      2. The final verdict is correct.
      3. There is explicit CGBV repair evidence:
           - Phase 1.5 bridge repair changed the verdict to the correct label, or
           - a committed Phase 5 repair changed the verdict from incorrect to correct.

    Phase 3 re-grounding is intentionally excluded from this audit because it
    repairs grounding/verification structure, not the semantic verdict itself.
    """
    label_map: dict[str, str] = {s.id: _normalise_label(s.label) for s in samples}
    result_map: dict[str, dict] = {r["sample_id"]: r for r in results if "sample_id" in r}

    eligible_initially_wrong = 0
    repaired_sample_ids: list[str] = []
    bridge_repaired_sample_ids: list[str] = []
    phase5_repaired_sample_ids: list[str] = []

    for sample in samples:
        result = result_map.get(sample.id)
        if not _is_successful_result(result):
            continue

        label = label_map[sample.id]
        v_pre, v_post, v_final = _read_verdicts(result)

        # Eligibility: pre-bridge verdict was wrong (something to fix)
        if v_pre == label:
            continue

        eligible_initially_wrong += 1

        # Bridge fixed: pre-bridge wrong → post-bridge correct
        bridge_fixed = (v_post == label and v_post != v_pre)
        # Phase 5 fixed: any committed repair round changed verdict from wrong→correct
        phase5_fixed = any(
            rnd.get("repair_attempted", False)
            and not rnd.get("repair_reverted", False)
            and _normalise_label(rnd.get("verdict_before")) != label
            and _normalise_label(rnd.get("verdict_after")) == label
            for rnd in result.get("rounds", [])
        )
        cgbv_repaired = v_final == label and (bridge_fixed or phase5_fixed)

        if not cgbv_repaired:
            continue

        repaired_sample_ids.append(sample.id)
        if bridge_fixed:
            bridge_repaired_sample_ids.append(sample.id)
        if phase5_fixed:
            phase5_repaired_sample_ids.append(sample.id)

    repaired_count = len(repaired_sample_ids)
    rate = round(repaired_count / eligible_initially_wrong, 4) if eligible_initially_wrong > 0 else 0.0

    return {
        "eligible_initially_wrong": eligible_initially_wrong,
        "repaired_count": repaired_count,
        "bridge_repaired_count": len(bridge_repaired_sample_ids),
        "phase5_repaired_count": len(phase5_repaired_sample_ids),
        "cgbv_repair_recovery_rate": rate,
        "repaired_sample_ids": repaired_sample_ids,
        "bridge_repaired_sample_ids": bridge_repaired_sample_ids,
        "phase5_repaired_sample_ids": phase5_repaired_sample_ids,
    }


def compute_sample_id_audit(
    results: list[dict],
    samples: list[DataSample],
) -> dict:
    """
    Collect high-signal sample ID slices for error analysis.

    Definitions:
      - error_sample_ids:
          samples that did not successfully produce a semantic verdict. This
          includes missing result files and non-success execution_status values
          such as phase1_error / solver_unknown / pipeline_error.
      - reasoning_error_sample_ids:
          successful runs whose FINAL verdict does not match the ground-truth label.
          Execution failures are excluded; those are pipeline/runtime errors, not
          semantic reasoning errors.
      - phase1_wrong_but_final_correct_sample_ids:
          successful runs where the earliest Phase 1 verdict snapshot
          (verdict_pre_bridge) was wrong, but the FINAL verdict is correct.
          This answers: "Which samples were initially formalized to the wrong
          verdict, but were corrected by the full CGBV stack?"
    """
    label_map: dict[str, str] = {s.id: _normalise_label(s.label) for s in samples}
    result_map: dict[str, dict] = {r["sample_id"]: r for r in results if "sample_id" in r}

    error_sample_ids: list[str] = []
    reasoning_error_sample_ids: list[str] = []
    phase1_wrong_but_final_correct_sample_ids: list[str] = []

    for sample in samples:
        result = result_map.get(sample.id)
        if not _is_successful_result(result):
            error_sample_ids.append(sample.id)
            continue

        label = label_map[sample.id]
        v_pre, _, v_final = _read_verdicts(result)

        if v_final != label:
            reasoning_error_sample_ids.append(sample.id)

        if v_pre != label and v_final == label:
            phase1_wrong_but_final_correct_sample_ids.append(sample.id)

    return {
        "error_count": len(error_sample_ids),
        "error_sample_ids": error_sample_ids,
        "reasoning_error_count": len(reasoning_error_sample_ids),
        "reasoning_error_sample_ids": reasoning_error_sample_ids,
        "phase1_wrong_but_final_correct_count": len(phase1_wrong_but_final_correct_sample_ids),
        "phase1_wrong_but_final_correct_sample_ids": phase1_wrong_but_final_correct_sample_ids,
    }


def compute_metrics(
    results: list[dict],
    samples: list[DataSample],
) -> dict:
    """
    Compute all CGBV evaluation metrics.

    Accuracy is reported at two levels (see Layer 7 design note):
      - end_to_end_accuracy   = correct / total_samples
          "If I deploy this system on N problems, what fraction do I get right?"
          Missing/errored samples count as wrong (they produce no useful answer).
      - conditional_accuracy  = correct / successful_samples
          "Among problems the system actually answered, how accurate is it?"
          Completed = execution_status == 'success' (or, for old results without
          that field, error is None).
      - completion_rate        = successful_samples / total_samples

    Metrics:
      1.  end_to_end_accuracy       — final verdict matches label (denom: all samples)
      1b. conditional_accuracy      — final verdict matches label (denom: successful runs)
      1c. completion_rate           — fraction of samples that ran successfully
      2.  binary_accuracy           — True/False samples only (denom: all T/F samples)
      3.  uncertain_recall          — Uncertain samples only (denom: all Uncertain samples)
      4.  verification_precision    — among verified samples, fraction correct
      5.  verification_coverage     — fraction of all samples that got verified
    Verdict level semantics (3-level separation):
      verdict_pre_bridge  = raw Phase 1 output before Phase 1.5 bridge repair
      verdict_post_bridge = after Phase 1.5 bridge repair, before Phase 5 repair loop
      verdict_final       = after all Phase 5 repairs

    Baseline policy:
      - Metrics 6-11 use verdict_post_bridge as "initially wrong/correct" baseline, because
        bridge repair has already happened before the verification loop.  A sample fixed by
        bridge is not "initially wrong" from the loop's perspective.
      - The bridge audit (compute_cgbv_repair_audit) uses verdict_pre_bridge for eligibility
        and verdict_pre_bridge → verdict_post_bridge for bridge_fixed detection.

      6.  mismatch_detection_precision — among samples with mismatches, fraction post-bridge wrong
      7.  mismatch_detection_recall    — fraction of post-bridge-wrong samples that had mismatches
      8.  repair_round_commit_rate  — fraction of repair rounds where at least one mismatch
                                      was fixed and the verdict was not reverted (partial commit)
      9.  repair_local_fix_rate     — fraction of mismatches that passed local acceptance check
     10.  repair_verdict_recovery_rate — among post-bridge-wrong repaired samples, fraction fixed
     11.  repair_regression_rate       — among post-bridge-correct repaired samples, fraction broken
     12.  cgbv_repair_recovery_rate    — among pre-bridge-wrong samples, fraction whose
                                         final correctness is attributable to an explicit
                                         CGBV repair mechanism (Phase 1.5 bridge repair
                                         or committed Phase 5 repair)

    Denominator policy (Findings 1 & 6):
      All per-label accuracy denominators (binary_total, uncertain_total) use the
      full label distribution, not just completed samples.  This ensures that a
      system that crashes on all Uncertain samples cannot claim a perfect
      uncertain_recall by having an empty denominator.  Unanswered/errored samples
      count as incorrect for those label-specific metrics.
    """
    label_map: dict[str, str] = {s.id: _normalise_label(s.label) for s in samples}
    total = len(samples)

    if total == 0:
        return _empty_metrics()

    result_map: dict[str, dict] = {r["sample_id"]: r for r in results if "sample_id" in r}

    # Counters
    correct = 0
    successful = 0            # execution_status == "success" (or legacy: error is None)
    verified = 0
    verified_correct = 0
    has_mismatch = 0
    mismatch_wrong = 0        # mismatch AND initially wrong (numerator for both prec/recall)
    wrong_initial = 0         # initially wrong (denominator for recall)
    repair_round_attempts = 0
    repair_round_commits = 0
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
    phase3_errors_detected = 0    # structural Phase 3 grounding errors detected
    phase3_reground_success = 0   # Phase 3 errors resolved via targeted re-grounding
    # P6: Verification confidence distribution
    confidence_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "none": 0}

    # Pre-count label totals from the full sample list (not just completed results)
    # so that missing/errored samples count as incorrect for label-specific metrics.
    for sample in samples:
        lbl = label_map[sample.id]
        if lbl in ("true", "false"):
            binary_total += 1
        elif lbl == "uncertain":
            uncertain_total += 1

    for sample in samples:
        r = result_map.get(sample.id)
        label = label_map[sample.id]

        # Determine if this sample ran successfully (has a meaningful semantic verdict)
        is_successful = _is_successful_result(r)

        if r is None or not is_successful:
            # Missing or errored samples count as wrong for end-to-end accuracy
            # (they produced no useful answer), but they must NOT increment
            # wrong_initial.  wrong_initial is the denominator of
            # mismatch_detection_recall; execution failures never enter the
            # mismatch detection pipeline and can never produce a mismatch, so
            # including them would systematically deflate recall.
            continue

        successful += 1

        v_pre, v_post, v_final = _read_verdicts(r)
        is_verified = r.get("verified", False)
        rounds = r.get("rounds", [])

        # P6: Accumulate confidence distribution
        conf = r.get("verification_confidence", "none")
        if conf in confidence_counts:
            confidence_counts[conf] += 1

        final_correct = (v_final == label)
        # post_bridge_correct: baseline after bridge, before repair loop.
        # Used for wrong_initial (mismatch detection denominator) and repair metrics.
        # Bridge has already happened before the verification loop, so samples that
        # were fixed by bridge are not "initially wrong" from the loop's perspective.
        post_bridge_correct = (v_post == label)

        if final_correct:
            correct += 1
        if not post_bridge_correct:
            wrong_initial += 1

        if is_verified:
            verified += 1
            if final_correct:
                verified_correct += 1

        if label in ("true", "false") and final_correct:
            binary_correct += 1
        elif label == "uncertain" and final_correct:
            uncertain_correct += 1

        # Mismatch detection baseline = verdict_post_bridge (bridge has already happened
        # before the verification loop; samples fixed by bridge are not "wrong" here).
        # Structural Phase 3 grounding errors (is_phase3_error=True) are excluded: they
        # are not actionable FOL mismatches and must not pollute mismatch precision/recall.
        def _has_actionable_mismatch(rnd: dict) -> bool:
            return any(
                not m.get("is_phase3_error", False)
                for m in rnd.get("mismatches", [])
            ) or any(
                not m.get("is_phase3_error", False)
                for m in rnd.get("carried_issues", [])
            )
        sample_has_mismatch = any(_has_actionable_mismatch(rnd) for rnd in rounds)
        if sample_has_mismatch:
            has_mismatch += 1
            if not post_bridge_correct:
                mismatch_wrong += 1

        # Repair metrics (round-level and mismatch-level)
        for rnd in rounds:
            if rnd.get("repair_attempted", False):
                repair_round_attempts += 1
                if rnd.get("repair_success", False) and not rnd.get("repair_reverted", False):
                    repair_round_commits += 1
                repair_mismatch_attempts += rnd.get("num_mismatches", 0)
                repair_local_validations += rnd.get("repair_local_validated", 0)
            # num_phase3_detected = original count before re-grounding (correct denominator)
            # num_phase3_errors   = remaining after re-grounding (not the denominator)
            phase3_errors_detected += rnd.get("num_phase3_detected", rnd.get("num_phase3_errors", 0))
            phase3_reground_success += rnd.get("num_phase3_reground_success", 0)

        sample_repaired = any(
            rnd.get("repair_attempted", False) for rnd in rounds
        )
        sample_repaired_kept = any(
            rnd.get("repair_attempted", False) and not rnd.get("repair_reverted", False)
            for rnd in rounds
        )

        # Repair recovery/regression baseline = verdict_post_bridge
        if sample_repaired and not post_bridge_correct:
            repair_recovery_attempted += 1
            if final_correct:
                repair_recovery_correct += 1

        if sample_repaired_kept and post_bridge_correct:
            repair_regression_attempted += 1
            if not final_correct:
                repair_regressions += 1

    def safe_div(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom > 0 else 0.0

    cgbv_repair_audit = compute_cgbv_repair_audit(results, samples)
    sample_id_audit = compute_sample_id_audit(results, samples)

    return {
        "total_samples": total,
        "successful_samples": successful,
        # Accuracy
        "end_to_end_accuracy": safe_div(correct, total),
        "conditional_accuracy": safe_div(correct, successful),
        "completion_rate": safe_div(successful, total),
        "binary_accuracy": safe_div(binary_correct, binary_total),
        "uncertain_recall": safe_div(uncertain_correct, uncertain_total),
        # Verification
        "verification_precision": safe_div(verified_correct, verified),
        "verification_coverage": safe_div(verified, total),
        # Mismatch detection (both use initial verdict)
        "mismatch_detection_precision": safe_div(mismatch_wrong, has_mismatch),
        "mismatch_detection_recall": safe_div(mismatch_wrong, wrong_initial),
        # Repair metrics
        "repair_round_commit_rate": safe_div(repair_round_commits, repair_round_attempts),
        "repair_local_fix_rate": safe_div(repair_local_validations, repair_mismatch_attempts),
        "repair_verdict_recovery_rate": safe_div(repair_recovery_correct, repair_recovery_attempted),
        "repair_regression_rate": safe_div(repair_regressions, repair_regression_attempted),
        "cgbv_repair_recovery_rate": cgbv_repair_audit["cgbv_repair_recovery_rate"],
        # Phase 3 re-grounding metrics
        "phase3_reground_rate": safe_div(phase3_reground_success, phase3_errors_detected),
        # P6: Verification confidence distribution
        "verification_confidence": confidence_counts,
        "sample_id_audit": sample_id_audit,
        # Raw counts (for inspection)
        "_counts": {
            "correct": correct,
            "wrong_initial": wrong_initial,
            "verified": verified,
            "verified_correct": verified_correct,
            "has_mismatch": has_mismatch,
            "mismatch_wrong": mismatch_wrong,
            "repair_round_attempts": repair_round_attempts,
            "repair_round_commits": repair_round_commits,
            "repair_mismatch_attempts": repair_mismatch_attempts,
            "repair_local_validations": repair_local_validations,
            "repair_recovery_attempted": repair_recovery_attempted,
            "repair_recovery_correct": repair_recovery_correct,
            "repair_regression_attempted": repair_regression_attempted,
            "repair_regressions": repair_regressions,
            "cgbv_repair_eligible": cgbv_repair_audit["eligible_initially_wrong"],
            "cgbv_repaired_count": cgbv_repair_audit["repaired_count"],
            "cgbv_bridge_repaired_count": cgbv_repair_audit["bridge_repaired_count"],
            "cgbv_phase5_repaired_count": cgbv_repair_audit["phase5_repaired_count"],
            "binary_correct": binary_correct,
            "binary_total": binary_total,
            "phase3_errors_detected": phase3_errors_detected,
            "phase3_reground_success": phase3_reground_success,
            "uncertain_correct": uncertain_correct,
            "uncertain_total": uncertain_total,
        },
    }


def _normalise_label(label: str) -> str:
    """Normalise label/verdict strings for comparison.

    "unknown" is intentionally NOT mapped to "uncertain":
      - "uncertain" = semantic (premises neither entail nor refute conclusion)
      - "unknown"   = solver/execution failure (Z3 timeout, Phase 1 error)
    Keeping them distinct prevents errored samples from spuriously matching
    Uncertain-labelled ground truth.
    """
    if label is None:
        return ""
    s = str(label).strip().lower()
    if s in ("true", "entailed", "yes", "correct"):
        return "true"
    if s in ("false", "not entailed", "no", "incorrect", "refuted"):
        return "false"
    if s in ("uncertain", "neither"):
        return "uncertain"
    return s  # "unknown" → "unknown" (does not match any label)


def _empty_metrics() -> dict:
    return {
        "total_samples": 0,
        "successful_samples": 0,
        "end_to_end_accuracy": 0.0,
        "conditional_accuracy": 0.0,
        "completion_rate": 0.0,
        "binary_accuracy": 0.0,
        "uncertain_recall": 0.0,
        "verification_precision": 0.0,
        "verification_coverage": 0.0,
        "mismatch_detection_precision": 0.0,
        "mismatch_detection_recall": 0.0,
        "repair_round_commit_rate": 0.0,
        "repair_local_fix_rate": 0.0,
        "repair_verdict_recovery_rate": 0.0,
        "repair_regression_rate": 0.0,
        "cgbv_repair_recovery_rate": 0.0,
        "phase3_reground_rate": 0.0,
        "verification_confidence": {"high": 0, "medium": 0, "low": 0, "none": 0},
        "sample_id_audit": {
            "error_count": 0,
            "error_sample_ids": [],
            "reasoning_error_count": 0,
            "reasoning_error_sample_ids": [],
            "phase1_wrong_but_final_correct_count": 0,
            "phase1_wrong_but_final_correct_sample_ids": [],
        },
        "_counts": {},
    }
