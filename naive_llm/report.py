from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from naive_llm.config import ExperimentConfig

logger = logging.getLogger(__name__)

_METRIC_LABELS = (
    ("accuracy", "Accuracy"),
    ("conditional_accuracy", "Conditional Accuracy"),
    ("completion_rate", "Completion Rate"),
    ("binary_accuracy", "Binary Accuracy"),
    ("uncertain_recall", "Uncertain Recall"),
    ("parse_error_rate", "Parse Error Rate"),
    ("runtime_error_rate", "Runtime Error Rate"),
)


def write_report(
    metrics: dict,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "algorithm": "naive_llm",
        "method": "zero_shot_cot",
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
        "llm": asdict(config.llm),
        "metrics": metrics,
    }

    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report written to %s", json_path)

    md_path = output_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Naive LLM Report\n\n")
        f.write(f"**Method:** Zero-Shot CoT  \n")
        f.write(f"**Experiment ID:** {config.experiment_id}  \n")
        f.write(f"**Run ID:** {config.run_id}  \n")
        f.write(f"**Dataset:** {config.dataset.name} / {config.dataset.split}  \n")
        f.write(f"**LLM:** {config.llm.provider} / {config.llm.model}  \n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, label in _METRIC_LABELS:
            f.write(f"| {label} | {metrics.get(key, 0.0):.4f} |\n")

        f.write("\n## Counts\n\n")
        f.write(f"- Correct / Total: {metrics.get('correct_samples', 0)} / {metrics.get('total_samples', 0)}\n")
        f.write(f"- Successful / Total: {metrics.get('successful_samples', 0)} / {metrics.get('total_samples', 0)}\n")
        f.write(f"- Error samples: {metrics.get('error_samples', 0)}\n")

        accuracy_by_label = metrics.get("accuracy_by_label", {})
        if accuracy_by_label:
            f.write("\n## Accuracy By Label\n\n")
            for label, value in accuracy_by_label.items():
                f.write(f"- {label}: {value:.4f}\n")

        confusion = metrics.get("confusion_matrix", {})
        if confusion:
            f.write("\n## Confusion Matrix\n\n")
            headers = ["gold \\ pred", "true", "false", "uncertain", "__missing__"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for gold, row in confusion.items():
                f.write(
                    "| "
                    + " | ".join(
                        [
                            gold,
                            str(row.get("true", 0)),
                            str(row.get("false", 0)),
                            str(row.get("uncertain", 0)),
                            str(row.get("__missing__", 0)),
                        ]
                    )
                    + " |\n"
                )

        sample_id_audit = metrics.get("sample_id_audit", {})
        if sample_id_audit:
            f.write("\n## Sample Audit\n\n")
            f.write(
                f"- Reasoning-error sample IDs ({len(sample_id_audit.get('reasoning_error_sample_ids', []))}): "
                f"{', '.join(sample_id_audit.get('reasoning_error_sample_ids', [])) or '-'}\n"
            )
            f.write(
                f"- Wrong sample IDs ({len(sample_id_audit.get('wrong_sample_ids', []))}): "
                f"{', '.join(sample_id_audit.get('wrong_sample_ids', [])) or '-'}\n"
            )
            f.write(
                f"- Error sample IDs ({len(sample_id_audit.get('error_sample_ids', []))}): "
                f"{', '.join(sample_id_audit.get('error_sample_ids', [])) or '-'}\n"
            )
            f.write(
                f"- Parse-error sample IDs ({len(sample_id_audit.get('parse_error_sample_ids', []))}): "
                f"{', '.join(sample_id_audit.get('parse_error_sample_ids', [])) or '-'}\n"
            )
            error_details = sample_id_audit.get("error_details", [])
            if error_details:
                f.write("\n### Error Details\n\n")
                for item in error_details:
                    f.write(
                        f"- {item.get('sample_id', '-')}: "
                        f"status={item.get('execution_status', '-')}, "
                        f"parse_status={item.get('parse_status', '-')}, "
                        f"error={item.get('error') or '-'}\n"
                    )
    logger.info("Markdown report written to %s", md_path)
