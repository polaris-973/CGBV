from __future__ import annotations

import json
import logging
from pathlib import Path

from cgbv.config.settings import ExperimentConfig

logger = logging.getLogger(__name__)

_METRIC_LABELS = {
    "general_accuracy": "General Accuracy",
    "binary_accuracy": "Binary Accuracy (True/False only)",
    "uncertain_recall": "Uncertain Recall",
    "verification_precision": "Verification Precision",
    "verification_coverage": "Verification Coverage",
    "mismatch_detection_precision": "Mismatch Detection Precision",
    "mismatch_detection_recall": "Mismatch Detection Recall",
    "repair_parse_success_rate": "Repair Parse Success Rate",
    "repair_local_fix_rate": "Repair Local Fix Rate",
    "repair_verdict_recovery_rate": "Repair Verdict Recovery Rate",
    "repair_regression_rate": "Repair Regression Rate",
}


def write_report(
    metrics: dict,
    results: list[dict],
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

    report = {
        "experiment_id": config.experiment_id,
        "run_id": config.run_id,
        "run_timestamp": config.run_timestamp,
        "description": config.description,
        "dataset": config.dataset.name,
        "split": config.dataset.split,
        "llm_model": config.llm.model,
        "pipeline": {
            "num_witnesses": config.pipeline.num_witnesses,
            "r_max": config.pipeline.r_max,
            "code_exec_timeout": config.pipeline.code_exec_timeout,
            "solver_timeout": config.pipeline.solver_timeout,
            "formalize_retries": config.pipeline.formalize_retries,
            "grounding_retries": config.pipeline.grounding_retries,
            "repair_retries": config.pipeline.repair_retries,
        },
        "metrics": metrics,
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
        f.write(f"Completed: {metrics.get('completed_samples', 0)} / {metrics.get('total_samples', 0)} samples\n\n")
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
    logger.info("Markdown report written to %s", md_path)
