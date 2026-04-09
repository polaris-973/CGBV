from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

import cgbv.logging as cgbv_log
from cgbv.config.settings import ExperimentConfig
from cgbv.core.pipeline import CGBVPipeline, PipelineResult
from cgbv.data.base import DataSample
from cgbv.data.loader import load_dataset
from cgbv.eval.metrics import compute_metrics
from cgbv.eval.report import write_report
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.runner.checkpoint import CheckpointManager
from cgbv.runner.concurrency import run_concurrent

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Top-level experiment orchestrator.

    Loads dataset → filters already-done samples → runs pipeline concurrently
    → computes metrics → writes report.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.checkpoint = CheckpointManager(
            results_dir=config.results_root,
            run_id=config.run_id,
        )

    async def run(self) -> dict:
        """Run the full experiment. Returns the final metrics dict."""
        cfg = self.config

        logger.info(
            "Experiment [%s] run [%s] starting: dataset=%s split=%s llm=%s",
            cfg.experiment_id, cfg.run_id, cfg.dataset.name, cfg.dataset.split, cfg.llm.model,
        )

        load_limit = cfg.dataset.limit
        if cfg.dataset.sample_index_range is not None:
            start, end = cfg.dataset.sample_index_range
            load_limit = end
            if cfg.dataset.limit is not None:
                logger.warning(
                    "dataset.limit=%s is ignored because dataset.sample_index_range=%d-%d is set",
                    cfg.dataset.limit, start, end,
                )

        # Load dataset
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
                "Filtered to %d samples by sample_index_range=%d-%d (1-based, inclusive)",
                len(all_samples), start, end,
            )

        # Filter to specific sample IDs if requested
        if cfg.dataset.only_ids:
            id_set = set(str(i) for i in cfg.dataset.only_ids)
            all_samples = [s for s in all_samples if str(s.id) in id_set]
            logger.info("Filtered to %d samples by only_ids", len(all_samples))

        # Exclude any evaluation samples that are also used as few-shot examples
        # for the current dataset/split.
        prompt_engine = PromptEngine(
            templates_dir=cfg.prompts.templates_dir,
            few_shot_dir=cfg.prompts.few_shot_dir,
        )
        excluded_ids = set(
            prompt_engine.get_excluded_ids(cfg.dataset.name, cfg.dataset.split)
        )
        if excluded_ids:
            before = len(all_samples)
            all_samples = [s for s in all_samples if str(s.id) not in excluded_ids]
            removed = before - len(all_samples)
            logger.info(
                "Excluded %d few-shot sample(s) from %s/%s evaluation",
                removed, cfg.dataset.name, cfg.dataset.split,
            )

        # Filter already-done samples (checkpoint)
        pending: list[DataSample]
        if cfg.runner.checkpoint:
            pending = self.checkpoint.filter_pending(all_samples)
        else:
            pending = all_samples

        if not pending:
            logger.info("All samples already completed. Computing metrics from existing results.")
        else:
            logger.info(
                "Running pipeline on [bold]%d[/] samples (concurrency=%d)",
                len(pending), cfg.runner.max_concurrency,
            )
            pipeline = CGBVPipeline(cfg)
            results_base = cfg.output_dir

            # Progress display — wraps the entire batch run
            with cgbv_log.ExperimentProgress(
                total=len(pending),
                console=_get_console(),
            ) as prog:
                cgbv_log.register_progress(prog)
                tasks = [
                    (s.id, _run_tracked(s, pipeline, results_base, prog))
                    for s in pending
                ]
                results = await run_concurrent(
                    tasks, max_concurrency=cfg.runner.max_concurrency
                )
                cgbv_log.register_progress(None)

            # Log summary of this batch (only count results without execution errors)
            successes = sum(
                1 for r in results
                if isinstance(r, PipelineResult) and r.error is None
            )
            errors = len(results) - successes
            logger.info(
                "Batch complete: [green]%d succeeded[/], [red]%d errored[/]",
                successes, errors,
            )

        # Load all results (including previously completed ones)
        all_results = self.checkpoint.load_all_results(dataset=cfg.dataset.name)
        # all_results includes errored runs; compute_metrics handles them correctly
        # by checking execution_status (or legacy error field) per result.
        logger.info(
            "Computing metrics over %d result files (%d total samples)",
            len(all_results), len(all_samples),
        )

        metrics = compute_metrics(all_results, all_samples)
        write_report(
            metrics=metrics,
            results=all_results,
            samples=all_samples,
            config=cfg,
            output_dir=cfg.output_dir,
        )

        sample_id_audit = metrics.get("sample_id_audit", {})
        error_ids = sample_id_audit.get("error_sample_ids", [])
        reasoning_error_ids = sample_id_audit.get("reasoning_error_sample_ids", [])
        phase1_wrong_final_correct_ids = sample_id_audit.get(
            "phase1_wrong_but_final_correct_sample_ids", []
        )
        phase1_correct_final_wrong_ids = sample_id_audit.get(
            "phase1_correct_but_final_wrong_sample_ids", []
        )
        logger.info(
            "Error sample IDs (%d): %s",
            len(error_ids),
            ", ".join(error_ids) if error_ids else "-",
        )
        logger.info(
            "Reasoning-error sample IDs (%d): %s",
            len(reasoning_error_ids),
            ", ".join(reasoning_error_ids) if reasoning_error_ids else "-",
        )
        logger.info(
            "Phase1-wrong but final-correct sample IDs (%d): %s",
            len(phase1_wrong_final_correct_ids),
            ", ".join(phase1_wrong_final_correct_ids) if phase1_wrong_final_correct_ids else "-",
        )
        logger.info(
            "Phase1-correct but final-wrong (CGBV regression) sample IDs (%d): %s",
            len(phase1_correct_final_wrong_ids),
            ", ".join(phase1_correct_final_wrong_ids) if phase1_correct_final_wrong_ids else "-",
        )

        logger.info(
            "Experiment [bold]%s[/] run [bold]%s[/] complete. Metrics: %s",
            cfg.experiment_id, cfg.run_id, metrics,
        )
        return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_console():
    """Return the shared Rich Console if logging has been set up, else None."""
    import logging as _logging
    for h in _logging.getLogger().handlers:
        if hasattr(h, "console"):
            return h.console
    return None


async def _run_tracked(
    sample: DataSample,
    pipeline: CGBVPipeline,
    results_base: Path,
    prog: "cgbv_log.ExperimentProgress",
) -> PipelineResult:
    """
    Wrapper that sets the sample context ContextVar, manages progress updates,
    and ensures complete_sample() is always called.

    Unhandled exceptions from the pipeline are caught here and materialised as
    a PipelineResult with execution_status="pipeline_error".  Writing result.json
    ensures the checkpoint and metrics layers can account for the failure on
    subsequent runs rather than silently ignoring it.
    """
    log_dir = results_base / sample.dataset / sample.id
    cgbv_log.set_sample(sample.id, log_dir, display_id=sample.display_id)
    prog.start_sample(sample.id, sample.display_id)
    try:
        result = await pipeline.run(sample)
        cgbv_log.complete_sample(success=(result.error is None))
        return result
    except Exception as exc:
        cgbv_log.complete_sample(success=False)
        error_msg = f"Unhandled pipeline exception: {type(exc).__name__}: {exc}"
        logger.error("Sample %s: %s", sample.id, error_msg, exc_info=True)
        result = PipelineResult(
            sample_id=sample.id,
            dataset=sample.dataset,
            label=sample.label,
            verdict=None,
            verdict_pre_bridge=None,
            verdict_post_bridge=None,
            verified=False,
            num_rounds=0,
            execution_status="pipeline_error",
            verification_status="not_run",
            error=error_msg,
        )
        out_dir = results_base / sample.dataset / sample.id
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        except Exception as write_exc:
            logger.warning(
                "Could not write pipeline_error result.json for %s: %s",
                sample.id, write_exc,
            )
        return result
