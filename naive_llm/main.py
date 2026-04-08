from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import cgbv.logging as cgbv_log

from cgbv.config.env import load_project_env
from naive_llm.config import load_config
from naive_llm.runner import NaiveLLMRunner

_METRIC_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("accuracy", "Accuracy"),
    ("conditional_accuracy", "Conditional Accuracy"),
    ("completion_rate", "Completion Rate"),
    ("binary_accuracy", "Binary Accuracy"),
    ("uncertain_recall", "Uncertain Recall"),
    ("parse_error_rate", "Parse Error Rate"),
    ("runtime_error_rate", "Runtime Error Rate"),
)


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Naive LLM zero-shot CoT experiment runner")
    parser.add_argument("--config", "-c", required=True, help="Path to experiment YAML config file")
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Terminal log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    logger = logging.getLogger("naive_llm.main")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    cgbv_log.setup_logging(level=args.log_level, results_dir=config.output_dir)
    logger = logging.getLogger("naive_llm.main")
    logger.info(
        "Config loaded: experiment=%s run=%s dataset=%s/%s llm=%s",
        config.experiment_id, config.run_id, config.dataset.name, config.dataset.split, config.llm.model,
    )

    runner = NaiveLLMRunner(config)
    metrics = asyncio.run(runner.run())

    print("\n" + "=" * 50)
    print(f"Experiment: {config.experiment_id}")
    print(f"Run:        {config.run_id}")
    print(f"Dataset:    {config.dataset.name}/{config.dataset.split}")
    print(f"LLM:        {config.llm.model}")
    print("=" * 50)
    for key, label in _METRIC_DISPLAY_ORDER:
        print(f"  {label:<24} {metrics.get(key, 0.0):.4f}")
    print("=" * 50)
    print(f"  {'Correct / Total':<24} {metrics.get('correct_samples', 0)}/{metrics.get('total_samples', 0)}")
    print(f"  {'Successful / Total':<24} {metrics.get('successful_samples', 0)}/{metrics.get('total_samples', 0)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
