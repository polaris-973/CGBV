from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cgbv.logging as cgbv_log
from cgbv.data.base import DataSample
from cgbv.data.loader import load_dataset
from cgbv.llm.factory import create_llm_client
from cgbv.runner.checkpoint import CheckpointManager
from cgbv.runner.concurrency import run_concurrent
from naive_llm.config import ExperimentConfig
from naive_llm.metrics import compute_metrics
from naive_llm.prompting import build_messages, build_prompt_text, normalize_label, parse_prediction
from naive_llm.report import write_report

logger = logging.getLogger(__name__)


@dataclass
class NaiveLLMResult:
    sample_id: str
    dataset: str
    label: str
    prediction: str | None
    is_correct: bool | None
    execution_status: str
    parse_status: str
    error: str | None = None
    raw_response: str = ""
    prompt_text: str = ""
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    display_id: str = ""
    conclusion: str = ""
    num_premises: int = 0


class NaiveLLMRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm = create_llm_client(config.llm)
        self.checkpoint = CheckpointManager(
            results_dir=config.results_root,
            run_id=config.run_id,
        )

    async def run(self) -> dict:
        cfg = self.config
        logger.info(
            "NaiveLLM [%s] run [%s] starting: dataset=%s split=%s llm=%s",
            cfg.experiment_id, cfg.run_id, cfg.dataset.name, cfg.dataset.split, cfg.llm.model,
        )

        load_limit = cfg.dataset.limit
        if cfg.dataset.sample_index_range is not None:
            _, end = cfg.dataset.sample_index_range
            load_limit = end

        all_samples = load_dataset(
            name=cfg.dataset.name,
            split=cfg.dataset.split,
            path=cfg.dataset.path,
            limit=load_limit,
        )
        logger.info("Loaded %d samples from %s/%s", len(all_samples), cfg.dataset.name, cfg.dataset.split)

        if cfg.dataset.sample_index_range is not None:
            start, end = cfg.dataset.sample_index_range
            all_samples = all_samples[start - 1:end]
            logger.info(
                "Filtered to %d samples by sample_index_range=%d-%d",
                len(all_samples), start, end,
            )

        if cfg.dataset.only_ids:
            id_set = {str(item) for item in cfg.dataset.only_ids}
            all_samples = [sample for sample in all_samples if str(sample.id) in id_set]
            logger.info("Filtered to %d samples by only_ids", len(all_samples))

        pending = (
            self.checkpoint.filter_pending(all_samples)
            if cfg.runner.checkpoint
            else all_samples
        )

        if pending:
            logger.info(
                "Running Naive LLM on %d samples (concurrency=%d)",
                len(pending), cfg.runner.max_concurrency,
            )
            with cgbv_log.ExperimentProgress(
                total=len(pending),
                console=_get_console(),
            ) as prog:
                cgbv_log.register_progress(prog)
                tasks = [
                    (
                        sample.id,
                        _run_tracked(
                            sample=sample,
                            runner=self,
                            results_base=cfg.output_dir,
                            prog=prog,
                        ),
                    )
                    for sample in pending
                ]
                await run_concurrent(tasks, max_concurrency=cfg.runner.max_concurrency)
                cgbv_log.register_progress(None)
        else:
            logger.info("All samples already completed. Aggregating existing Naive LLM results.")

        all_results = self.checkpoint.load_all_results(dataset=cfg.dataset.name)
        metrics = compute_metrics(all_results, all_samples)
        write_report(metrics=metrics, config=cfg, output_dir=cfg.output_dir)

        sample_id_audit = metrics.get("sample_id_audit", {})
        logger.info(
            "Wrong sample IDs (%d): %s",
            len(sample_id_audit.get("wrong_sample_ids", [])),
            ", ".join(sample_id_audit.get("wrong_sample_ids", [])) or "-",
        )
        logger.info(
            "Error sample IDs (%d): %s",
            len(sample_id_audit.get("error_sample_ids", [])),
            ", ".join(sample_id_audit.get("error_sample_ids", [])) or "-",
        )
        logger.info(
            "Parse-error sample IDs (%d): %s",
            len(sample_id_audit.get("parse_error_sample_ids", [])),
            ", ".join(sample_id_audit.get("parse_error_sample_ids", [])) or "-",
        )
        logger.info(
            "NaiveLLM [%s] run [%s] complete. Metrics: %s",
            cfg.experiment_id, cfg.run_id, metrics,
        )
        return metrics

    async def run_sample(self, sample: DataSample, results_base: Path) -> NaiveLLMResult:
        out_dir = results_base / sample.dataset / sample.id
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = build_prompt_text(sample)
        prompt_messages = build_messages(sample)
        logger.info("Calling LLM for sample %s", sample.id)
        logger.debug("Prompt for %s:\n%s", sample.id, prompt_text)

        raw_response = await self.llm.complete_with_retry(prompt_messages)
        logger.debug("Raw response for %s:\n%s", sample.id, raw_response)

        parsed = parse_prediction(raw_response)
        execution_status = "success" if parsed.verdict is not None else "parse_error"
        gold_label = normalize_label(sample.label) or sample.label
        is_correct = parsed.verdict == gold_label if parsed.verdict is not None else None
        result = NaiveLLMResult(
            sample_id=sample.id,
            dataset=sample.dataset,
            label=sample.label,
            prediction=parsed.verdict,
            is_correct=is_correct,
            execution_status=execution_status,
            parse_status=parsed.parse_status,
            error=parsed.parse_error,
            raw_response=raw_response,
            prompt_text=prompt_text,
            prompt_messages=prompt_messages,
            model=self.config.llm.model,
            provider=self.config.llm.provider,
            display_id=sample.display_id or sample.id,
            conclusion=sample.conclusion,
            num_premises=len(sample.premises),
        )
        _write_json(out_dir / "result.json", asdict(result))
        return result


def _get_console():
    import logging as _logging

    for handler in _logging.getLogger().handlers:
        if hasattr(handler, "console"):
            return handler.console
    return None


async def _run_tracked(
    sample: DataSample,
    runner: NaiveLLMRunner,
    results_base: Path,
    prog: "cgbv_log.ExperimentProgress",
) -> NaiveLLMResult:
    log_dir = results_base / sample.dataset / sample.id
    cgbv_log.set_sample(sample.id, log_dir, display_id=sample.display_id)
    prog.start_sample(sample.id, sample.display_id)
    try:
        result = await runner.run_sample(sample, results_base)
        cgbv_log.complete_sample(success=(result.execution_status == "success"))
        return result
    except Exception as exc:
        cgbv_log.complete_sample(success=False)
        error_msg = f"Unhandled baseline exception: {type(exc).__name__}: {exc}"
        logger.error("Sample %s: %s", sample.id, error_msg, exc_info=True)
        result = NaiveLLMResult(
            sample_id=sample.id,
            dataset=sample.dataset,
            label=sample.label,
            prediction=None,
            is_correct=None,
            execution_status="pipeline_error",
            parse_status="not_run",
            error=error_msg,
            model=runner.config.llm.model,
            provider=runner.config.llm.provider,
            display_id=sample.display_id or sample.id,
            conclusion=sample.conclusion,
            num_premises=len(sample.premises),
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_json(log_dir / "result.json", asdict(result))
        return result


def _write_json(path: Path, payload: dict) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
