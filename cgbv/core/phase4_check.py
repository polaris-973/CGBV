from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import z3

from cgbv.core.phase3_grounded import GroundedFormula
from cgbv.solver.z3_solver import Z3Solver

logger = logging.getLogger(__name__)

# Mismatch types (from Proposal-v2 Definition 3)
MISMATCH_WEAKENING = "weakening"        # M ⊨ f_i = ⊤  AND  M ⊨ φ_i^M = ⊥  (FOL too weak)
MISMATCH_STRENGTHENING = "strengthening"  # M ⊨ f_i = ⊥  AND  M ⊨ φ_i^M = ⊤  (FOL too strong)


@dataclass
class Mismatch:
    sentence_index: int      # index into sentences list (0..n-1 premises, n conclusion)
    nl_sentence: str
    mismatch_type: str       # "weakening" | "strengthening"
    fol_truth: bool          # M ⊨ f_i  (Phase 1 FOL formula truth value)
    grounded_truth: bool     # M ⊨ φ_i^M  (Phase 3 grounded formula truth value)
    fol_formula_str: str     # str() of the FOL formula (for repair prompt)
    grounded_formula: str    # the truth-table expression (for repair prompt)
    witness_index: int = 0   # which witness produced this mismatch
    witness_side: str = "not_q"


@dataclass
class SentenceEval:
    """Per-sentence evaluation record (for debugging / result logging)."""
    sentence_index: int
    nl_sentence: str
    fol_truth: bool | None
    grounded_truth: bool | None
    mismatch: bool
    mismatch_type: str | None
    grounding_failed: bool
    fol_formula_str: str | None = None
    grounded_formula: str | None = None
    fol_eval_repr: str | None = None
    witness_index: int = 0
    witness_side: str = "not_q"
    error: str | None = None


@dataclass
class Phase4Result:
    mismatches: list[Mismatch]
    all_passed: bool                      # True if no mismatches found
    evaluations: list[SentenceEval]       # per-sentence debug info
    witness_index: int = 0
    witness_side: str = "not_q"
    num_unverifiable: int = 0
    error: str | None = None


def run_phase4(
    sentences: list[str],
    fol_formulas: list,              # z3.ExprRef list, same length as sentences
    model: Any,                      # z3.ModelRef (from Phase 2)
    domain: dict,                    # structured domain (from Phase 2)
    grounded_formulas: list[GroundedFormula],  # Phase 3 output
    solver: Z3Solver,
    witness_index: int = 0,
    witness_side: str = "not_q",
) -> Phase4Result:
    """
    Phase 4: Cross-Granularity Check.

    For each sentence S_j (premises + conclusion):
      - Compute M ⊨ f_j  via z3 model.evaluate()          (FOL side, Phase 1)
      - Compute M ⊨ φ_j^M via truth-table eval()           (propositional side, Phase 3)
      - If they differ → record mismatch

    Mismatch types (Proposal-v2 Definition 3):
      - weakening:     fol_truth=True,  grounded_truth=False  (FOL accepts world NL rejects)
      - strengthening: fol_truth=False, grounded_truth=True   (FOL rejects world NL accepts)

    All comparison logic runs through the solver (zero LLM involvement).
    """
    if len(fol_formulas) != len(sentences) or len(grounded_formulas) != len(sentences):
        return Phase4Result(
            mismatches=[],
            all_passed=False,
            evaluations=[],
            witness_index=witness_index,
            witness_side=witness_side,
            error=(
                f"Length mismatch: sentences={len(sentences)}, "
                f"fol_formulas={len(fol_formulas)}, "
                f"grounded_formulas={len(grounded_formulas)}"
            ),
        )

    mismatches: list[Mismatch] = []
    evaluations: list[SentenceEval] = []

    for idx, (sentence, fol_formula, grounded) in enumerate(
        zip(sentences, fol_formulas, grounded_formulas)
    ):
        # Evaluate FOL formula on z3 model (Phase 1 side)
        fol_truth: bool | None = None
        grounded_truth: bool | None = None
        eval_error: str | None = None

        if grounded.failed:
            # Phase 3 failed for this sentence — skip comparison
            evaluations.append(SentenceEval(
                sentence_index=idx,
                nl_sentence=sentence,
                fol_truth=None,
                grounded_truth=None,
                mismatch=False,
                mismatch_type=None,
                grounding_failed=True,
                fol_formula_str=str(fol_formula),
                grounded_formula=grounded.formula_code,
                witness_index=witness_index,
                witness_side=witness_side,
                error=grounded.error,
            ))
            logger.debug("Phase 4 idx=%d: grounding failed, skipping", idx)
            continue

        try:
            fol_eval_val = model.evaluate(fol_formula, model_completion=True)
            fol_truth = z3.is_true(fol_eval_val)
            fol_eval_repr = str(fol_eval_val)
        except Exception as e:
            eval_error = f"FOL evaluation error: {e}"
            fol_eval_repr = None
            logger.warning("Phase 4 idx=%d: %s", idx, eval_error)

        if eval_error is None:
            grounded_truth = solver.evaluate_grounded_formula(domain, grounded.formula_code)
            if grounded_truth is None:
                eval_error = f"Grounded formula evaluation error for: {grounded.formula_code!r}"
                logger.warning("Phase 4 idx=%d: %s", idx, eval_error)

        if eval_error:
            evaluations.append(SentenceEval(
                sentence_index=idx,
                nl_sentence=sentence,
                fol_truth=fol_truth,
                grounded_truth=grounded_truth,
                mismatch=False,
                mismatch_type=None,
                grounding_failed=False,
                fol_formula_str=str(fol_formula),
                grounded_formula=grounded.formula_code,
                fol_eval_repr=fol_eval_repr,
                witness_index=witness_index,
                witness_side=witness_side,
                error=eval_error,
            ))
            continue

        # Compare truth values
        is_mismatch = (fol_truth != grounded_truth)
        mismatch_type: str | None = None

        if is_mismatch:
            if fol_truth and not grounded_truth:
                mismatch_type = MISMATCH_WEAKENING
            else:
                mismatch_type = MISMATCH_STRENGTHENING

            mismatches.append(Mismatch(
                sentence_index=idx,
                nl_sentence=sentence,
                mismatch_type=mismatch_type,
                fol_truth=fol_truth,
                grounded_truth=grounded_truth,
                fol_formula_str=str(fol_formula),
                grounded_formula=grounded.formula_code,
                witness_index=witness_index,
                witness_side=witness_side,
            ))
            logger.info(
                "Phase 4 idx=%d MISMATCH (%s): FOL=%s, grounded=%s | %r",
                idx, mismatch_type, fol_truth, grounded_truth, sentence[:60],
            )
        else:
            logger.debug("Phase 4 idx=%d OK: FOL=%s, grounded=%s", idx, fol_truth, grounded_truth)

        evaluations.append(SentenceEval(
            sentence_index=idx,
            nl_sentence=sentence,
            fol_truth=fol_truth,
            grounded_truth=grounded_truth,
            mismatch=is_mismatch,
            mismatch_type=mismatch_type,
            grounding_failed=False,
            fol_formula_str=str(fol_formula),
            grounded_formula=grounded.formula_code,
            fol_eval_repr=fol_eval_repr,
            witness_index=witness_index,
            witness_side=witness_side,
            error=eval_error,
        ))

    # A sentence that failed grounding or evaluation cannot be verified.
    # Treat it the same as a mismatch for the purposes of all_passed so that
    # the pipeline doesn't claim "verified" when some sentences were skipped.
    num_unverifiable = sum(
        1 for e in evaluations if e.grounding_failed or e.error is not None
    )
    return Phase4Result(
        mismatches=mismatches,
        all_passed=(len(mismatches) == 0 and num_unverifiable == 0),
        evaluations=evaluations,
        witness_index=witness_index,
        witness_side=witness_side,
        num_unverifiable=num_unverifiable,
    )
