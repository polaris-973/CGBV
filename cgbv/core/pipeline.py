from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cgbv.logging as cgbv_log
from cgbv.config.settings import ExperimentConfig
from cgbv.core.multi_witness import MultiWitnessResult, run_multi_witness
from cgbv.core.phase1_formalize import Phase1Result, run_phase1
from cgbv.core.phase3_grounded import Phase3Result
from cgbv.core.phase4_check import Mismatch, Phase4Result
from cgbv.core.phase5_repair import Phase5Result, run_phase5
from cgbv.data.base import DataSample
from cgbv.llm.base import LLMClient
from cgbv.llm.factory import create_llm_client
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNKNOWN, VERDICT_REFUTED, VERDICT_UNCERTAIN

logger = logging.getLogger(__name__)


def _normalise_verdict(v: str) -> str:
    """Lightweight verdict normaliser for pipeline-internal use."""
    s = str(v).strip().lower()
    if s in ("true", "entailed", "yes"):
        return "true"
    if s in ("false", "not entailed", "no", "refuted"):
        return "false"
    if s in ("uncertain", "unknown", "neither"):
        return "uncertain"
    return s


@dataclass
class RoundRecord:
    """Record of one repair round (Phase 2+3+4+5)."""
    round_num: int                          # 1-indexed
    num_witnesses: int
    mismatches: list[dict]                  # serialisable mismatch records
    all_passed: bool
    repair_attempted: bool
    repair_success: bool
    verdict_before: str | None = None
    verdict_candidate: str | None = None
    verdict_after: str | None = None
    repair_reverted: bool = False           # P0.4: repair was reverted due to verdict regression
    repair_local_validated: int = 0        # P0.4: mismatches that passed local acceptance
    num_mismatches: int = 0               # total mismatches sent to Phase 5


@dataclass
class PipelineResult:
    sample_id: str
    dataset: str
    label: str                              # ground truth
    verdict: str                            # final solver verdict (after all repairs)
    verdict_initial: str                    # Phase 1 verdict, before any repair
    verified: bool                          # True if no mismatch found in any witness
    num_rounds: int
    rounds: list[RoundRecord] = field(default_factory=list)
    phase1_raw_code: str = ""
    error: str | None = None


class CGBVPipeline:
    """
    Main CGBV pipeline controller.

    Orchestrates Phase 1 → R_max rounds of (Multi-Witness Phase 2+3+4 → Phase 5) loops.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm: LLMClient = create_llm_client(config.llm)
        self.solver = Z3Solver(timeout_ms=config.pipeline.solver_timeout * 1000)
        self.prompt_engine = PromptEngine(
            templates_dir=config.prompts.templates_dir,
            few_shot_dir=config.prompts.few_shot_dir,
        )
        self._results_base = config.output_dir

    async def run(self, sample: DataSample) -> PipelineResult:
        """
        Run the full CGBV pipeline on one sample.

        Algorithm:
        1. Phase 1: NL → FOL → solver verdict (with Distinct constraints)
        2. for round in 1..R_max:
             Phase 2+3+4: Multi-Witness (¬q side + q side for Uncertain)
             if no mismatch: return verified
             Phase 5: repair with local acceptance check
             re-solve; if verdict regressed and was initially correct → revert (P0.4)
        3. return unverified (exhausted R_max rounds)
        """
        out_dir = self._results_base / sample.dataset / sample.id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Phase 1: Formalize & Solve
        # ----------------------------------------------------------------
        cgbv_log.update_phase("phase1")
        p1 = await run_phase1(
            premises_nl=sample.premises,
            conclusion_nl=sample.conclusion,
            llm=self.llm,
            solver=self.solver,
            prompt_engine=self.prompt_engine,
            dataset=sample.dataset,
            max_retries=self.config.pipeline.formalize_retries,
            task_type=sample.task_type,
            code_exec_timeout=self.config.pipeline.code_exec_timeout,
        )
        self._write_json(out_dir / "phase1.json", _phase1_to_dict(p1, sample))

        if p1.error or p1.q is None:
            result = PipelineResult(
                sample_id=sample.id,
                dataset=sample.dataset,
                label=sample.label,
                verdict=VERDICT_UNKNOWN,
                verdict_initial=VERDICT_UNKNOWN,
                verified=False,
                num_rounds=0,
                phase1_raw_code=p1.raw_code,
                error=p1.error or "Phase 1 produced no formula",
            )
            self._write_json(out_dir / "result.json", asdict(result))
            return result

        # Working copies
        premises = list(p1.premises)               # NL-only formulas
        background_constraints = list(p1.background_constraints)
        bound_var_names = p1.bound_var_names       # P0.1: ForAll/Exists variable names
        q = p1.q
        verdict = p1.verdict
        verdict_initial = p1.verdict               # never updated
        namespace = p1.namespace
        model_info = p1.model_info
        model_info_q = p1.model_info_q             # P1.1: q-side model for Uncertain

        sentences = sample.premises + [sample.conclusion]
        rounds: list[RoundRecord] = []

        # P0.4: is the Phase 1 verdict initially correct? (for regression guard)
        label_norm = _normalise_verdict(sample.label)
        initial_correct = (_normalise_verdict(verdict_initial) == label_norm)

        # ----------------------------------------------------------------
        # Repair loop (up to R_max rounds)
        # ----------------------------------------------------------------
        for round_num in range(1, self.config.pipeline.r_max + 1):
            logger.info(
                "Pipeline sample=%s round=%d/%d verdict=%s",
                sample.id, round_num, self.config.pipeline.r_max, verdict,
            )

            # Phase 2+3+4: Multi-Witness
            cgbv_log.update_phase("phase2", f"round {round_num}/{self.config.pipeline.r_max}")
            mw = await run_multi_witness(
                verdict=verdict,
                model_info=model_info,
                model_info_q=model_info_q,
                premises=premises,
                q=q,
                namespace=namespace,
                sentences=sentences,
                solver=self.solver,
                llm=self.llm,
                prompt_engine=self.prompt_engine,
                background_constraints=background_constraints,
                bound_var_names=bound_var_names,
                num_witnesses=self.config.pipeline.num_witnesses,
                grounding_retries=self.config.pipeline.grounding_retries,
            )
            self._write_json(
                out_dir / f"round{round_num}_witness.json",
                _mw_to_dict(mw),
            )
            for wr in mw.witness_results:
                k = wr.witness_index
                self._write_json(
                    out_dir / f"round{round_num}_w{k}_phase3.json",
                    _phase3_to_dict(wr.phase3),
                )
                self._write_json(
                    out_dir / f"round{round_num}_w{k}_phase4.json",
                    _phase4_to_dict(wr.phase4),
                )

            round_record = RoundRecord(
                round_num=round_num,
                num_witnesses=mw.num_witnesses_constructed,
                mismatches=[_mismatch_to_dict(m) for m in mw.mismatches],
                all_passed=mw.all_passed,
                repair_attempted=False,
                repair_success=False,
                verdict_before=verdict,
                verdict_after=verdict,
                num_mismatches=len(mw.mismatches),
            )

            if mw.all_passed:
                rounds.append(round_record)
                result = PipelineResult(
                    sample_id=sample.id,
                    dataset=sample.dataset,
                    label=sample.label,
                    verdict=verdict,
                    verdict_initial=verdict_initial,
                    verified=True,
                    num_rounds=round_num,
                    rounds=rounds,
                    phase1_raw_code=p1.raw_code,
                )
                self._write_json(out_dir / "result.json", asdict(result))
                logger.info("Pipeline sample=%s VERIFIED at round %d", sample.id, round_num)
                return result

            if mw.num_witnesses_constructed == 0:
                logger.warning(
                    "Pipeline sample=%s round=%d: all witness constructions failed",
                    sample.id, round_num,
                )
                rounds.append(round_record)
                continue

            if round_num < self.config.pipeline.r_max and mw.mismatches:
                # Use the first witness for repair prompt context (domain description)
                first_wr = mw.witness_results[0] if mw.witness_results else None
                first_witness_domain = first_wr.phase2.domain if first_wr else {}

                # P0.4: per-witness model and domain maps so each mismatch
                # is repaired and validated against the world it came from
                witness_models = {
                    wr.witness_index: wr.phase2.model
                    for wr in mw.witness_results
                    if wr.phase2.model is not None
                }
                witness_domains = {
                    wr.witness_index: wr.phase2.domain
                    for wr in mw.witness_results
                    if wr.phase2.domain is not None
                }

                cgbv_log.update_phase("phase5", f"round {round_num}/{self.config.pipeline.r_max}")
                p5 = await run_phase5(
                    mismatches=mw.mismatches,
                    premises=premises,
                    q=q,
                    namespace=namespace,
                    raw_code=p1.raw_code,
                    domain=first_witness_domain,   # fallback domain for prompt
                    llm=self.llm,
                    prompt_engine=self.prompt_engine,
                    max_retries=self.config.pipeline.repair_retries,
                    models=witness_models,         # P0.4 local acceptance (per-witness)
                    domains=witness_domains,       # per-witness domain for accurate prompts
                    solver=self.solver,
                )
                self._write_json(
                    out_dir / f"round{round_num}_repair.json",
                    _phase5_to_dict(p5),
                )

                round_record.repair_attempted = True
                round_record.repair_success = p5.all_repaired
                round_record.repair_local_validated = p5.num_local_validated

                if p5.all_repaired:
                    # Snapshot for possible revert
                    old_premises = list(premises)
                    old_q = q
                    old_verdict = verdict

                    premises = p5.repaired_premises
                    q = p5.repaired_q

                    solver_premises = list(premises) + background_constraints
                    if sample.task_type == "three_class":
                        new_verdict, new_model_info, new_model_info_q = (
                            self.solver.check_entailment_three_class(solver_premises, q)
                        )
                    else:
                        new_verdict, new_model_info = self.solver.check_entailment(
                            solver_premises, q
                        )
                        new_model_info_q = None

                    # P0.4 Global revert: undo repair if it regressed an initially correct verdict
                    new_correct = (_normalise_verdict(new_verdict) == label_norm)
                    round_record.verdict_candidate = new_verdict
                    if initial_correct and not new_correct:
                        logger.warning(
                            "Pipeline sample=%s round=%d: repair regressed verdict "
                            "(%s → %s vs label=%s); reverting",
                            sample.id, round_num, old_verdict, new_verdict, sample.label,
                        )
                        premises = old_premises
                        q = old_q
                        verdict = old_verdict
                        round_record.verdict_after = old_verdict
                        round_record.repair_reverted = True
                        round_record.repair_success = False
                    else:
                        verdict = new_verdict
                        round_record.verdict_after = new_verdict
                        model_info = new_model_info
                        model_info_q = new_model_info_q
                        p1 = _patched_p1(
                            p1, premises, q, new_verdict, new_model_info, new_model_info_q
                        )
                        logger.info(
                            "Pipeline sample=%s round=%d: repair committed, new verdict=%s",
                            sample.id, round_num, verdict,
                        )
                else:
                    logger.warning(
                        "Pipeline sample=%s round=%d: repair partially failed",
                        sample.id, round_num,
                    )

            rounds.append(round_record)

        # Exhausted R_max rounds without full verification
        result = PipelineResult(
            sample_id=sample.id,
            dataset=sample.dataset,
            label=sample.label,
            verdict=verdict,
            verdict_initial=verdict_initial,
            verified=False,
            num_rounds=self.config.pipeline.r_max,
            rounds=rounds,
            phase1_raw_code=p1.raw_code,
        )
        self._write_json(out_dir / "result.json", asdict(result))
        logger.info(
            "Pipeline sample=%s UNVERIFIED after %d rounds",
            sample.id, self.config.pipeline.r_max,
        )
        return result

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_make_serialisable(data), f, indent=2, ensure_ascii=False, default=_json_default)
        except Exception as e:
            logger.warning("Failed to write %s: %s", path, e)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _phase1_to_dict(p1: Phase1Result, sample: DataSample | None = None) -> dict:
    return {
        "verdict": p1.verdict,
        "raw_code": p1.raw_code,
        "premises_nl": list(sample.premises) if sample is not None else None,
        "conclusion_nl": sample.conclusion if sample is not None else None,
        "num_premises_nl": len(sample.premises) if sample is not None else None,
        "num_premises_formulas": len(p1.premises),
        "premises": [str(f) for f in p1.premises],
        "background_constraints": [str(f) for f in p1.background_constraints],
        "bound_var_names": sorted(p1.bound_var_names),
        "q": str(p1.q) if p1.q is not None else None,
        "attempts": [
            {
                "attempt_num": a.attempt_num,
                "messages": a.messages,
                "raw_output": a.raw_output,
                "code_exec_error": a.code_exec_error,
                "solver_error": a.solver_error,
                "verdict": a.verdict,
            }
            for a in p1.attempts
        ],
        "error": p1.error,
    }


def _mismatch_to_dict(m: Mismatch) -> dict:
    return {
        "sentence_index": m.sentence_index,
        "nl_sentence": m.nl_sentence,
        "mismatch_type": m.mismatch_type,
        "fol_truth": m.fol_truth,
        "grounded_truth": m.grounded_truth,
        "fol_formula_str": m.fol_formula_str,
        "grounded_formula": m.grounded_formula,
        "witness_index": m.witness_index,
        "witness_side": m.witness_side,
    }


def _mw_to_dict(mw: MultiWitnessResult) -> dict:
    return {
        "num_witnesses_constructed": mw.num_witnesses_constructed,
        "all_passed": mw.all_passed,
        "mismatches": [_mismatch_to_dict(m) for m in mw.mismatches],
        "witnesses": [
            {
                "witness_index": wr.witness_index,
                "witness_side": wr.phase2.witness_side,
                "domain_summary": _domain_summary(wr.phase2.domain),
                "domain": wr.phase2.domain,
                "phase4_all_passed": wr.phase4.all_passed,
                "phase4_mismatches": len(wr.phase4.mismatches),
                "phase4_unverifiable": wr.phase4.num_unverifiable,
            }
            for wr in mw.witness_results
        ],
    }


def _phase3_to_dict(p3: Phase3Result) -> dict:
    return {
        "grounded": [
            {
                "sentence_index": g.sentence_index,
                "nl_sentence": g.nl_sentence,
                "formula_code": g.formula_code,
                "failed": g.failed,
                "attempts": [
                    {
                        "attempt_num": a.attempt_num,
                        "messages": a.messages,
                        "raw_output": a.raw_output,
                        "extracted_formula": a.extracted_formula,
                        "validation_error": a.validation_error,
                        "accepted": a.accepted,
                    }
                    for a in g.attempts
                ],
                "error": g.error,
            }
            for g in p3.grounded
        ],
    }


def _phase4_to_dict(p4: Phase4Result) -> dict:
    return {
        "all_passed": p4.all_passed,
        "witness_index": p4.witness_index,
        "witness_side": p4.witness_side,
        "num_unverifiable": p4.num_unverifiable,
        "mismatches": [_mismatch_to_dict(m) for m in p4.mismatches],
        "evaluations": [
            {
                "sentence_index": e.sentence_index,
                "nl_sentence": e.nl_sentence,
                "fol_truth": e.fol_truth,
                "grounded_truth": e.grounded_truth,
                "mismatch": e.mismatch,
                "mismatch_type": e.mismatch_type,
                "grounding_failed": e.grounding_failed,
                "fol_formula_str": e.fol_formula_str,
                "grounded_formula": e.grounded_formula,
                "fol_eval_repr": e.fol_eval_repr,
                "witness_index": e.witness_index,
                "witness_side": e.witness_side,
                "error": e.error,
            }
            for e in p4.evaluations
        ],
        "error": p4.error,
    }


def _phase5_to_dict(p5: Phase5Result) -> dict:
    return {
        "all_repaired": p5.all_repaired,
        "num_local_validated": p5.num_local_validated,
        "error": p5.error,
        "repairs": [
            {
                "sentence_index": r.sentence_index,
                "mismatch_type": r.mismatch_type,
                "witness_index": r.witness_index,
                "witness_side": r.witness_side,
                "original": r.original_formula_str,
                "grounded_formula": r.grounded_formula,
                "fol_truth_before": r.fol_truth_before,
                "grounded_truth_expected": r.grounded_truth_expected,
                "repaired": r.repaired_expr_str,
                "success": r.success,
                "local_validated": r.local_validated,
                "attempts": [
                    {
                        "attempt_num": a.attempt_num,
                        "messages": a.messages,
                        "raw_output": a.raw_output,
                        "extracted_expression": a.extracted_expression,
                        "eval_error": a.eval_error,
                        "local_validation_error": a.local_validation_error,
                        "local_validation_truth": a.local_validation_truth,
                        "accepted": a.accepted,
                    }
                    for a in r.attempts
                ],
                "error": r.error,
            }
            for r in p5.repairs
        ],
    }


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _make_serialisable(obj):
    if isinstance(obj, dict):
        return {
            (", ".join(str(k) for k in key) if isinstance(key, tuple) else str(key) if not isinstance(key, str) else key): _make_serialisable(val)
            for key, val in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    return obj


def _domain_summary(domain: dict | None) -> dict:
    """Compact witness summary for faster result inspection."""
    if not domain:
        return {}

    sorts = domain.get("sorts", {})
    predicates = domain.get("predicates", {})
    true_atoms = 0
    false_atoms = 0
    for interp in predicates.values():
        for val in interp.values():
            if val:
                true_atoms += 1
            else:
                false_atoms += 1

    return {
        "num_entities": len(domain.get("entities", [])),
        "entities_by_sort": {sort_name: len(entities) for sort_name, entities in sorts.items()},
        "num_predicates": len(predicates),
        "true_atoms": true_atoms,
        "false_atoms": false_atoms,
    }


def _patched_p1(
    p1: Phase1Result,
    premises: list,
    q: object,
    verdict: str,
    model_info: object,
    model_info_q: object | None = None,
) -> Phase1Result:
    """Return a shallow copy of p1 with updated premises/q/verdict/model_info."""
    from cgbv.core.phase1_formalize import Phase1Result as P1
    return P1(
        verdict=verdict,
        premises=premises,
        background_constraints=p1.background_constraints,
        bound_var_names=p1.bound_var_names,
        q=q,
        model_info=model_info,
        model_info_q=model_info_q,
        namespace=p1.namespace,
        raw_code=p1.raw_code,
        attempts=p1.attempts,
        error=None,
    )
