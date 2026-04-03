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
from dotenv import load_dotenv


def main() -> None:
    # Load .env from the current working directory (project root) before
    # anything else so API keys are available when config is loaded.
    load_dotenv()
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
    for key in [
        "general_accuracy",
        "binary_accuracy",
        "uncertain_recall",
        "verification_precision",
        "verification_coverage",
        "mismatch_detection_precision",
        "mismatch_detection_recall",
        "repair_parse_success_rate",
        "repair_local_fix_rate",
        "repair_verdict_recovery_rate",
        "repair_regression_rate",
    ]:
        val = metrics.get(key, 0.0)
        label = key.replace("_", " ").title()
        print(f"  {label:<36} {val:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
