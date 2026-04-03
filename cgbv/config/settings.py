from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class LLMConfig:
    provider: str           # openai | deepseek | qwen | glm
    model: str
    api_key_env: str        # environment variable name holding the API key
    base_url: Optional[str] = None
    extra_body: Optional[dict[str, Any]] = None
    max_parallel_requests: Optional[int] = None
    min_request_interval: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 4096
    api_retry_count: int = 3      # network-level retries on transient API errors
    api_retry_delay: float = 2.0  # exponential backoff base (seconds)

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EnvironmentError(
                f"LLM API key not found. Set environment variable: {self.api_key_env}"
            )
        return key


@dataclass
class DatasetConfig:
    name: str               # folio | proofwriter | prontoqa | ar_lsat | logical_deduction | proverqa
    split: str              # train | validation | test | dev | easy | medium | hard
    path: str = "./datasets"
    limit: Optional[int] = None  # limit number of samples (for debugging)


@dataclass
class PipelineConfig:
    num_witnesses: int = 1      # K: Multi-Witness parameter (1 = single witness)
    r_max: int = 3              # maximum repair rounds
    code_exec_timeout: int = 30  # seconds
    solver_timeout: int = 60    # seconds
    formalize_retries: int = 3
    grounding_retries: int = 2
    repair_retries: int = 2       # Phase 5 LLM repair retries per mismatch


@dataclass
class RunnerConfig:
    max_concurrency: int = 10
    checkpoint: bool = True
    results_dir: str = "./results"


@dataclass
class PromptsConfig:
    templates_dir: str = "./cgbv/prompts"
    few_shot_dir: str = "./cgbv/prompts/few_shot"


@dataclass
class ExperimentConfig:
    experiment_id: str
    description: str
    dataset: DatasetConfig
    llm: LLMConfig
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    run_id: str = ""
    run_timestamp: str = ""

    @property
    def output_dir(self) -> Path:
        return Path(self.runner.results_dir) / self.run_id


_PIPELINE_RUN_ID_FIELDS: tuple[tuple[str, str], ...] = (
    ("num_witnesses", "k"),
    ("r_max", "r"),
    ("formalize_retries", "fr"),
    ("grounding_retries", "gr"),
    ("repair_retries", "rr"),
    ("code_exec_timeout", "ce"),
    ("solver_timeout", "st"),
)


def _make_run_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _sanitize_path_component(value: object) -> str:
    text = str(value).strip()
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("-")
    sanitized = "".join(allowed).strip("-._")
    return sanitized or "default"


def _build_run_id(base_experiment_id: str, pipeline: PipelineConfig, timestamp: str) -> str:
    parts = [_sanitize_path_component(base_experiment_id)]
    for field_name, short_name in _PIPELINE_RUN_ID_FIELDS:
        parts.append(f"{short_name}{getattr(pipeline, field_name)}")
    parts.append(_sanitize_path_component(timestamp))
    return "_".join(parts)


def load_config(config_path: str | Path) -> ExperimentConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    exp = raw.get("experiment", {})
    dataset_raw = raw.get("dataset", {})
    llm_raw = raw.get("llm", {})
    pipeline_raw = raw.get("pipeline", {})
    runner_raw = raw.get("runner", {})
    prompts_raw = raw.get("prompts", {})
    pipeline_cfg = PipelineConfig(**{k: v for k, v in pipeline_raw.items() if v is not None})
    runner_cfg = RunnerConfig(**{k: v for k, v in runner_raw.items() if v is not None})
    prompts_cfg = PromptsConfig(**{k: v for k, v in prompts_raw.items() if v is not None})
    experiment_id = exp.get("id", "exp_default")
    run_timestamp = str(exp.get("run_timestamp") or _make_run_timestamp())
    run_id = str(exp.get("run_id") or _build_run_id(experiment_id, pipeline_cfg, run_timestamp))

    return ExperimentConfig(
        experiment_id=experiment_id,
        description=exp.get("description", ""),
        dataset=DatasetConfig(
            name=dataset_raw["name"],
            split=dataset_raw.get("split", "validation"),
            path=dataset_raw.get("path", "./datasets"),
            limit=dataset_raw.get("limit"),
        ),
        llm=LLMConfig(
            provider=llm_raw["provider"],
            model=llm_raw["model"],
            api_key_env=llm_raw.get("api_key_env", "OPENAI_API_KEY"),
            base_url=llm_raw.get("base_url"),
            extra_body=llm_raw.get("extra_body"),
            max_parallel_requests=llm_raw.get("max_parallel_requests"),
            min_request_interval=llm_raw.get("min_request_interval", 0.0),
            temperature=llm_raw.get("temperature", 0.0),
            max_tokens=llm_raw.get("max_tokens", 4096),
            api_retry_count=llm_raw.get("api_retry_count", 3),
            api_retry_delay=llm_raw.get("api_retry_delay", 2.0),
        ),
        pipeline=pipeline_cfg,
        runner=runner_cfg,
        prompts=prompts_cfg,
        run_id=run_id,
        run_timestamp=run_timestamp,
    )
