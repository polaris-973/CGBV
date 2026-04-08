"""
CGBV experiment runner CLI.

Usage:
    python -m cgbv.main --config path/to/experiment_config.yaml
    python -m cgbv.main --config path/to/experiment_config.yaml --log-level DEBUG
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import cgbv.logging as cgbv_log


_METRIC_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("end_to_end_accuracy", "End-to-End Accuracy"),
    ("conditional_accuracy", "Conditional Accuracy"),
    ("completion_rate", "Completion Rate"),
    ("binary_accuracy", "Binary Accuracy"),
    ("uncertain_recall", "Uncertain Recall"),
    ("verification_precision", "Verification Precision"),
    ("verification_coverage", "Verification Coverage"),
    ("mismatch_detection_precision", "Mismatch Detection Precision"),
    ("mismatch_detection_recall", "Mismatch Detection Recall"),
    ("repair_round_commit_rate", "Repair Round Commit Rate"),
    ("repair_local_fix_rate", "Repair Local Fix Rate"),
    ("repair_verdict_recovery_rate", "Repair Verdict Recovery Rate"),
    ("repair_regression_rate", "Repair Regression Rate"),
    ("cgbv_repair_recovery_rate", "CGBV Repair Recovery Rate"),
    ("phase3_reground_rate", "Phase 3 Re-grounding Rate"),
)


def main() -> None:
    # Load environment variables before config parsing.
    from cgbv.config.env import load_project_env

    load_project_env()
    parser = argparse.ArgumentParser(
        description="CGBV: Cross-Granularity Boundary Verification experiment runner",
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Terminal log level (default: INFO; DEBUG+ always written to files)",
    )
    args = parser.parse_args()

    # Logging is set up after config is loaded so we know results_dir.
    # Use a temporary basicConfig for the config-load phase.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    logger = logging.getLogger("cgbv.main")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    from cgbv.config.settings import load_config
    from cgbv.runner.experiment import ExperimentRunner

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        sys.exit(1)

    # Replace the bootstrap handler with the full Rich logging system.
    results_dir = config.output_dir
    cgbv_log.setup_logging(level=args.log_level, results_dir=results_dir)
    logger = logging.getLogger("cgbv.main")
    logger.info(
        "Config loaded: experiment=[bold]%s[/] run=[bold]%s[/] dataset=%s/%s llm=%s",
        config.experiment_id, config.run_id, config.dataset.name, config.dataset.split, config.llm.model,
    )

    runner = ExperimentRunner(config)
    metrics = asyncio.run(runner.run())

    # Print key metrics to stdout
    print("\n" + "=" * 50)
    print(f"Experiment: {config.experiment_id}")
    print(f"Run:        {config.run_id}")
    print(f"Dataset:    {config.dataset.name}/{config.dataset.split}")
    print(f"LLM:        {config.llm.model}")
    print("=" * 50)
    for key, label in _METRIC_DISPLAY_ORDER:
        val = metrics.get(key, 0.0)
        print(f"  {label:<36} {val:.4f}")

    counts = metrics.get("_counts", {})
    if counts:
        total = metrics.get("total_samples", 0)
        successful = metrics.get("successful_samples", 0)
        verified = counts.get("verified", 0)
        verified_correct = counts.get("verified_correct", 0)
        correct = counts.get("correct", 0)
        print("=" * 50)
        print(f"  {'Correct / Total':<36} {correct}/{total}")
        print(f"  {'Successful / Total':<36} {successful}/{total}")
        print(f"  {'Verified / Total':<36} {verified}/{total}")
        print(f"  {'Verified Correct / Verified':<36} {verified_correct}/{verified}")

    confidence = metrics.get("verification_confidence", {})
    if confidence:
        high = confidence.get("high", 0)
        medium = confidence.get("medium", 0)
        low = confidence.get("low", 0)
        none = confidence.get("none", 0)
        print(f"  {'Verification Confidence H/M/L/N':<36} {high}/{medium}/{low}/{none}")

    sample_id_audit = metrics.get("sample_id_audit", {})
    if sample_id_audit:
        error_ids = sample_id_audit.get("error_sample_ids", [])
        reasoning_error_ids = sample_id_audit.get("reasoning_error_sample_ids", [])
        phase1_wrong_final_correct_ids = sample_id_audit.get(
            "phase1_wrong_but_final_correct_sample_ids", []
        )
        print("=" * 50)
        print(
            f"  {'Error Samples':<36} "
            f"{sample_id_audit.get('error_count', len(error_ids))}"
        )
        print(
            f"  {'Error IDs':<36} "
            f"{', '.join(error_ids) if error_ids else '-'}"
        )
        print(
            f"  {'Reasoning Error Samples':<36} "
            f"{sample_id_audit.get('reasoning_error_count', len(reasoning_error_ids))}"
        )
        print(
            f"  {'Reasoning Error IDs':<36} "
            f"{', '.join(reasoning_error_ids) if reasoning_error_ids else '-'}"
        )
        print(
            f"  {'Phase1 Wrong -> Final Correct':<36} "
            f"{sample_id_audit.get('phase1_wrong_but_final_correct_count', len(phase1_wrong_final_correct_ids))}"
        )
        print(
            f"  {'Phase1 Wrong -> Final Correct IDs':<36} "
            f"{', '.join(phase1_wrong_final_correct_ids) if phase1_wrong_final_correct_ids else '-'}"
        )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
