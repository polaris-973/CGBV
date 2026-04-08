from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from cgbv.config.settings import (
    DatasetConfig,
    LLMConfig,
    _default_api_key_env,
    _parse_sample_index_range,
)


@dataclass
class RunnerConfig:
    max_concurrency: int = 10
    checkpoint: bool = True
    results_dir: str = "./results"
    results_subdir: str = "naive_llm"


@dataclass
class ExperimentConfig:
    experiment_id: str
    description: str
    dataset: DatasetConfig
    llm: LLMConfig
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    run_id: str = ""
    run_timestamp: str = ""

    @property
    def results_root(self) -> Path:
        return Path(self.runner.results_dir) / self.runner.results_subdir

    @property
    def output_dir(self) -> Path:
        return self.results_root / self.run_id


def _make_run_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _sanitize_path_component(value: object) -> str:
    text = str(value).strip()
    allowed: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("-")
    sanitized = "".join(allowed).strip("-._")
    return sanitized or "default"


def _build_run_id(base_experiment_id: str, llm_model: str, timestamp: str) -> str:
    return "_".join(
        [
            _sanitize_path_component(base_experiment_id),
            _sanitize_path_component(llm_model),
            _sanitize_path_component(timestamp),
        ]
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    exp = raw.get("experiment", {})
    dataset_raw = raw.get("dataset", {})
    llm_raw = raw.get("llm", {})
    runner_raw = raw.get("runner", {})

    experiment_id = exp.get("id", "naive_llm_default")
    run_timestamp = str(exp.get("run_timestamp") or _make_run_timestamp())
    provider = llm_raw["provider"]
    llm_model = llm_raw["model"]
    run_id = str(exp.get("run_id") or _build_run_id(experiment_id, llm_model, run_timestamp))

    return ExperimentConfig(
        experiment_id=experiment_id,
        description=exp.get("description", ""),
        dataset=DatasetConfig(
            name=dataset_raw["name"],
            split=dataset_raw.get("split", "validation"),
            path=dataset_raw.get("path", "./datasets"),
            limit=dataset_raw.get("limit"),
            sample_index_range=_parse_sample_index_range(dataset_raw.get("sample_index_range")),
            only_ids=dataset_raw.get("only_ids"),
        ),
        llm=LLMConfig(
            provider=provider,
            model=llm_model,
            api_key_env=llm_raw.get("api_key_env") or _default_api_key_env(provider),
            base_url=llm_raw.get("base_url"),
            extra_body=llm_raw.get("extra_body"),
            max_parallel_requests=llm_raw.get("max_parallel_requests"),
            min_request_interval=llm_raw.get("min_request_interval", 0.0),
            temperature=llm_raw.get("temperature", 0.0),
            max_tokens=llm_raw.get("max_tokens", 4096),
            api_retry_count=llm_raw.get("api_retry_count", 3),
            api_retry_delay=llm_raw.get("api_retry_delay", 2.0),
        ),
        runner=RunnerConfig(**{k: v for k, v in runner_raw.items() if v is not None}),
        run_id=run_id,
        run_timestamp=run_timestamp,
    )
