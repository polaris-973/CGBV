from __future__ import annotations

import json
import logging
from pathlib import Path

from cgbv.config.settings import ExperimentConfig
from cgbv.data.base import DataSample
from cgbv.eval.metrics import compute_cgbv_repair_audit, compute_sample_id_audit

logger = logging.getLogger(__name__)

_METRIC_LABELS = {
    "end_to_end_accuracy": "End-to-End Accuracy (correct / all samples)",
    "conditional_accuracy": "Conditional Accuracy  (correct / successful runs)",
    "completion_rate": "Completion Rate        (successful / all samples)",
    "binary_accuracy": "Binary Accuracy (True/False only)",
    "uncertain_recall": "Uncertain Recall",
    "verification_precision": "Verification Precision",
    "verification_coverage": "Verification Coverage",
    "verified_uncertain_precision": "Verified Uncertain Precision",
    "mismatch_detection_precision": "Mismatch Detection Precision",
    "mismatch_detection_recall": "Mismatch Detection Recall",
    "repair_round_commit_rate": "Repair Round Commit Rate",
    "repair_local_fix_rate": "Repair Local Fix Rate",
    "repair_verdict_recovery_rate": "Repair Verdict Recovery Rate",
    "repair_regression_rate": "Repair Regression Rate",
    "cgbv_repair_recovery_rate": "CGBV Repair Recovery Rate",
    "cgbv_regression_rate": "CGBV Regression Rate (phase1-correct→final-wrong)",
    "phase3_reground_rate": "Phase 3 Re-grounding Resolution Rate",
    "underformalized_rate": "Underformalized Rate",
    "phase1_repeat_failure_rate": "Phase 1 Repeat Failure Rate",
    "obligation_resolution_rate": "Obligation Resolution Rate",
}


def write_report(
    metrics: dict,
    results: list[dict],
    samples: list[DataSample],
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """
    Write experiment report as JSON and Markdown.

    Outputs:
      {output_dir}/report.json    — full metrics + config snapshot
      {output_dir}/report.md      — human-readable Markdown table
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cgbv_repair_audit = compute_cgbv_repair_audit(results, samples)
    sample_id_audit = metrics.get("sample_id_audit") or compute_sample_id_audit(results, samples)

    report = {
        "experiment_id": config.experiment_id,
        "run_id": config.run_id,
        "run_timestamp": config.run_timestamp,
        "description": config.description,
        "dataset": config.dataset.name,
        "split": config.dataset.split,
        "dataset_filters": {
            "limit": config.dataset.limit,
            "sample_index_range": (
                list(config.dataset.sample_index_range)
                if config.dataset.sample_index_range is not None
                else None
            ),
            "only_ids": config.dataset.only_ids,
        },
        "llm_model": config.llm.model,
        "pipeline": {
            "num_witnesses": config.pipeline.num_witnesses,
            "r_max": config.pipeline.r_max,
            "code_exec_timeout": config.pipeline.code_exec_timeout,
            "solver_timeout": config.pipeline.solver_timeout,
            "formalize_retries": config.pipeline.formalize_retries,
            "grounding_retries": config.pipeline.grounding_retries,
            "repair_retries": config.pipeline.repair_retries,
            "world_assumption": config.pipeline.world_assumption,
        },
        "metrics": metrics,
        "cgbv_repair_audit": cgbv_repair_audit,
        "sample_id_audit": sample_id_audit,
    }

    # JSON report
    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report written to %s", json_path)

    # Markdown report
    md_path = output_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# CGBV Experiment Report\n\n")
        f.write(f"**Experiment ID:** {config.experiment_id}  \n")
        f.write(f"**Run ID:** {config.run_id}  \n")
        f.write(f"**Dataset:** {config.dataset.name} / {config.dataset.split}  \n")
        f.write(f"**LLM:** {config.llm.model}  \n")
        f.write(f"**Witnesses (K):** {config.pipeline.num_witnesses}  \n")
        f.write(f"**Max repair rounds (R_max):** {config.pipeline.r_max}  \n\n")
        f.write(f"Completed: {metrics.get('successful_samples', 0)} / {metrics.get('total_samples', 0)} samples\n\n")
        f.write("## Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, label in _METRIC_LABELS.items():
            val = metrics.get(key, 0.0)
            f.write(f"| {label} | {val:.4f} |\n")
        counts = metrics.get("_counts", {})
        if counts:
            f.write("\n## Raw Counts\n\n")
            f.write("| Counter | Value |\n")
            f.write("|---------|-------|\n")
            for k, v in counts.items():
                f.write(f"| {k} | {v} |\n")
        if sample_id_audit:
            f.write("\n## Sample ID Audit\n\n")
            f.write(
                f"- Error samples ({sample_id_audit.get('error_count', 0)}): "
                f"{', '.join(sample_id_audit.get('error_sample_ids', [])) or '-'}\n"
            )
            error_details = sample_id_audit.get("error_details", [])
            if error_details:
                f.write("\n### Error Details\n\n")
                for item in error_details:
                    tags = ", ".join(item.get("diagnostic_tags", [])) or "-"
                    err = item.get("error") or "-"
                    f.write(
                        f"- {item.get('sample_id', '-')}: "
                        f"status={item.get('execution_status', '-')}, "
                        f"acceptance={item.get('acceptance_state', '-')}, "
                        f"tags={tags}, error={err}\n"
                    )
            f.write(
                f"- Reasoning-error samples ({sample_id_audit.get('reasoning_error_count', 0)}): "
                f"{', '.join(sample_id_audit.get('reasoning_error_sample_ids', [])) or '-'}\n"
            )
            f.write(
                f"- Phase1-wrong but final-correct samples "
                f"({sample_id_audit.get('phase1_wrong_but_final_correct_count', 0)}): "
                f"{', '.join(sample_id_audit.get('phase1_wrong_but_final_correct_sample_ids', [])) or '-'}\n"
            )
            f.write(
                f"- Phase1-correct but final-wrong samples (CGBV regression) "
                f"({sample_id_audit.get('phase1_correct_but_final_wrong_count', 0)}): "
                f"{', '.join(sample_id_audit.get('phase1_correct_but_final_wrong_sample_ids', [])) or '-'}\n"
            )
    logger.info("Markdown report written to %s", md_path)
