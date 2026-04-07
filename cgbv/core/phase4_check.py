from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import z3

from cgbv.core.phase3_grounded import GroundedFormula
from cgbv.solver.finite_evaluator import FiniteModelEvaluator
from cgbv.solver.z3_solver import Z3Solver

logger = logging.getLogger(__name__)

# Mismatch types (from Proposal-v2 Definition 3)
MISMATCH_WEAKENING = "weakening"        # M ⊨ f_i = ⊤  AND  M ⊨ φ_i^M = ⊥  (FOL too weak)
MISMATCH_STRENGTHENING = "strengthening"  # M ⊨ f_i = ⊥  AND  M ⊨ φ_i^M = ⊤  (FOL too strong)

_evaluator = FiniteModelEvaluator()


def _quantifier_layout(formula: z3.ExprRef) -> str:
    """
    Return the quantifier layout of a Z3 formula:
      "none"   — no ForAll/Exists anywhere in the AST
      "top"    — the outermost node is a quantifier
      "buried" — quantifier exists but is not the outermost node
    """
    if z3.is_quantifier(formula):
        return "top"
    # Walk the AST to detect any nested quantifier.
    stack = list(formula.children()) if z3.is_app(formula) else []
    while stack:
        node = stack.pop()
        if z3.is_quantifier(node):
            return "buried"
        if z3.is_app(node):
            stack.extend(node.children())
    return "none"


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
    # Conclusion mismatch hint (NOT a hard routing decision).
    # True when this mismatch is on the conclusion AND the mismatch direction
    # aligns with the witness construction expectation.  This is a HINT that
    # the template may be wrong, but it does NOT prove the FOL side is correct
    # — the witness is constructed to make FOL(q) match its expected value by
    # definition, so FOL correctness on the witness is tautological.
    # Pipeline uses this to try retemplate first, then escalates to Phase 5 /
    # run_phase1_targeted if retemplate fails.
    is_phase3_error: bool = False
    # Quantifier layout of the original FOL formula, computed once from the Z3
    # object at Mismatch creation time so Phase 5 can avoid string heuristics.
    #   "none"   — no quantifiers anywhere
    #   "top"    — outermost constructor is ForAll/Exists
    #   "buried" — quantifier exists but is not the outermost constructor
    fol_quantifier_layout: str = "none"
    # 9.5: Number of repair rounds this mismatch has persisted without resolution.
    # Set by the pipeline when the same sentence_index re-appears across rounds.
    persist_rounds: int = 0


@dataclass
class SentenceEval:
    """Per-sentence evaluation record (for debugging / result logging)."""
    sentence_index: int
    nl_sentence: str
    fol_truth: bool | None           # None = unverifiable (quantifier universe unavailable)
    grounded_truth: bool | None
    mismatch: bool
    mismatch_type: str | None
    grounding_failed: bool
    fol_formula_str: str | None = None
    grounded_formula: str | None = None
    fol_eval_repr: str | None = None  # raw z3 repr before tri-state reduction
    witness_index: int = 0
    witness_side: str = "not_q"
    error: str | None = None


@dataclass
class Phase4Result:
    mismatches: list[Mismatch]
    all_passed: bool                      # True if no mismatches and no unverifiable sentences
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
    namespace: dict | None = None,   # Phase 1 exec namespace (for evaluator fallback)
    witness_index: int = 0,
    witness_side: str = "not_q",
) -> Phase4Result:
    """
    Phase 4: Cross-Granularity Check.

    For each sentence S_j (premises + conclusion):
      - Compute M ⊨ f_j via FiniteModelEvaluator (tri-state: True/False/None)
      - Compute M ⊨ φ_j^M via truth-table eval()  (propositional side, Phase 3)
      - Both must be concrete booleans (not None) for a comparison to occur
      - If both are concrete and differ → record mismatch
      - If fol_truth is None → unverifiable (counted but not a mismatch)

    Mismatch types (Proposal-v2 Definition 3):
      - weakening:     fol_truth=True,  grounded_truth=False
      - strengthening: fol_truth=False, grounded_truth=True
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
        fol_truth: bool | None = None
        grounded_truth: bool | None = None
        eval_error: str | None = None
        fol_eval_repr: str | None = None

        if grounded.failed:
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

        # --- FOL side: tri-state evaluation via FiniteModelEvaluator ---
        try:
            # Capture raw repr before reduction (for debugging)
            raw_val = model.evaluate(fol_formula, model_completion=True)
            fol_eval_repr = str(raw_val)
            fol_truth = _evaluator.evaluate(model, fol_formula, namespace=namespace)
        except Exception as e:
            eval_error = f"FOL evaluation error: {e}"
            logger.warning("Phase 4 idx=%d: %s", idx, eval_error)

        # --- Grounded side: propositional truth-table eval ---
        if eval_error is None:
            grounded_truth = solver.evaluate_grounded_formula(domain, grounded.formula_code)
            if grounded_truth is None:
                eval_error = (
                    f"Grounded formula evaluation error for: {grounded.formula_code!r}"
                )
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

        # --- If fol_truth is None, record as unverifiable (not a mismatch) ---
        if fol_truth is None:
            logger.debug(
                "Phase 4 idx=%d: fol_truth=None (unverifiable) — skipping mismatch check",
                idx,
            )
            evaluations.append(SentenceEval(
                sentence_index=idx,
                nl_sentence=sentence,
                fol_truth=None,
                grounded_truth=grounded_truth,
                mismatch=False,
                mismatch_type=None,
                grounding_failed=False,
                fol_formula_str=str(fol_formula),
                grounded_formula=grounded.formula_code,
                fol_eval_repr=fol_eval_repr,
                witness_index=witness_index,
                witness_side=witness_side,
                error="fol_truth=None: quantifier universe unavailable for finite instantiation",
            ))
            continue

        # --- Compare concrete truth values ---
        is_mismatch = (fol_truth != grounded_truth)
        mismatch_type: str | None = None

        if is_mismatch:
            mismatch_type = (
                MISMATCH_WEAKENING if fol_truth else MISMATCH_STRENGTHENING
            )
            # Conclusion mismatch detection (hint flag, NOT a definitive error attribution).
            # The witness is constructed so that the conclusion has a specific
            # expected truth value: False on a ¬q witness, True on a q witness.
            # When the FOL side matches that expected value but the grounded side
            # disagrees, the template MAY be wrong — but the FOL MAY ALSO be wrong
            # (FOL matching the witness is tautological by construction, not evidence
            # of FOL correctness).  The pipeline uses this as a hint to try
            # retemplate first, then escalates to Phase 5 / theory rewrite.
            #
            #   ¬q witness: expected conclusion = False
            #     → FOL=False (by construction) + grounded=True = strengthening
            #   q  witness: expected conclusion = True
            #     → FOL=True  (by construction) + grounded=False = weakening
            n_sentences = len(sentences)
            is_p3_error = (
                idx == n_sentences - 1  # conclusion is the last sentence
                and (
                    (witness_side == "not_q" and mismatch_type == MISMATCH_STRENGTHENING)
                    or (witness_side == "q"     and mismatch_type == MISMATCH_WEAKENING)
                )
            )
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
                is_phase3_error=is_p3_error,
                fol_quantifier_layout=_quantifier_layout(fol_formula),
            ))
            logger.info(
                "Phase 4 idx=%d MISMATCH (%s): FOL=%s, grounded=%s | %r",
                idx, mismatch_type, fol_truth, grounded_truth, sentence[:60],
            )
        else:
            logger.debug(
                "Phase 4 idx=%d OK: FOL=%s, grounded=%s", idx, fol_truth, grounded_truth
            )

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
        ))

    # Sentences that could not be fully verified (grounding failed, eval error,
    # or fol_truth=None) are tracked separately.  They do NOT count as mismatches
    # but DO prevent the result from being "all_passed" (we can't claim verified
    # when some sentences were unevaluable).
    num_unverifiable = sum(
        1 for e in evaluations
        if e.grounding_failed or e.error is not None or e.fol_truth is None
    )
    return Phase4Result(
        mismatches=mismatches,
        all_passed=(len(mismatches) == 0 and num_unverifiable == 0),
        evaluations=evaluations,
        witness_index=witness_index,
        witness_side=witness_side,
        num_unverifiable=num_unverifiable,
    )
