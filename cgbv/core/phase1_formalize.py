from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.code_executor import execute_z3_code, CodeExecutionError, build_name_error_hint, build_runtime_error_hint
from cgbv.solver.cwa_axioms import build_cwa_constraints
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNKNOWN, VERDICT_UNCERTAIN, VERDICT_ENTAILED

from cgbv.core.phase4_check import Mismatch

logger = logging.getLogger(__name__)


@dataclass
class Phase1Attempt:
    """One Phase 1 LLM attempt, including retry context and failure mode."""
    attempt_num: int
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    code_exec_error: str | None = None
    validation_error: str | None = None
    solver_error: str | None = None
    verdict: str | None = None


@dataclass
class Phase1Result:
    verdict: str                       # Entailed | Refuted | Uncertain | Not Entailed | Unknown
    premises: list                     # z3.ExprRef list — NL-corresponding formulas only
                                       # (same length as NL premises, aligned for Phase 3+4)
    q: object                          # z3.ExprRef
    background_constraints: list       # system-injected constraints (Distinct() etc.)
                                       # used for solver/witness but NOT in Phase 3+4 comparison
    bound_var_names: set               # ForAll/Exists quantifier variable names — excluded from domain
    model_info: object | None          # z3.ModelRef for P∧¬q side (Not Entailed / Uncertain)
    model_info_q: object | None        # z3.ModelRef for P∧q side (Uncertain only)
    namespace: dict[str, Any]          # full exec namespace
    raw_code: str                      # the generated Z3 code
    verdict_pre_bridge: str = ""       # verdict before Phase 1.5 bridge repair (empty = no bridge ran)
    attempts: list[Phase1Attempt] = field(default_factory=list)
    error: str | None = None           # set if all retries failed


async def run_phase1(
    premises_nl: list[str],
    conclusion_nl: str,
    llm: LLMClient,
    solver: Z3Solver,
    prompt_engine: PromptEngine,
    dataset: str = "",
    max_retries: int = 3,
    task_type: str = "entailment",
    code_exec_timeout: int = 30,
    world_assumption: str = "owa",
    bridge_retries: int = 2,
    enable_bridge: bool = True,
) -> Phase1Result:
    """
    Phase 1: Formalize & Solve.

    LLM translates NL → Z3-Python code.
    CodeExecutor exec()s the code.
    Z3Solver checks entailment with Unique Name Assumption (Distinct constraints).

    On code errors: feed error back to LLM and retry (up to max_retries).

    Phase 1.5 (bridge check) runs automatically after a successful formalization
    when the verdict is not Entailed, to detect and repair structurally
    disconnected premises.
    """
    messages = _build_messages(premises_nl, conclusion_nl, prompt_engine, dataset, world_assumption)
    last_error: str | None = None
    last_validation_feedback: list[str] = []
    last_name_error_hint: str | None = None   # specific spelling-error diagnosis for retry
    raw_code = ""
    attempts: list[Phase1Attempt] = []
    _vacuous_check_done: bool = False       # fire vacuous-model check at most once

    for attempt in range(max_retries):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_code},
                _build_retry_message(
                    premises_nl=premises_nl,
                    conclusion_nl=conclusion_nl,
                    raw_code=raw_code,
                    last_error=last_error or "Unknown error",
                    validation_feedback=last_validation_feedback,
                    name_error_hint=last_name_error_hint,
                    attempt_num=attempt + 1,
                    max_retries=max_retries,
                    prompt_engine=prompt_engine,
                ),
            ]
        last_name_error_hint = None   # reset; will be set only if this attempt has a NameError

        attempt_record = Phase1Attempt(
            attempt_num=attempt + 1,
            messages=_snapshot_messages(messages),
        )
        raw_code = await llm.complete_with_retry(messages)
        attempt_record.raw_output = raw_code

        try:
            ctx = execute_z3_code(raw_code, timeout_seconds=code_exec_timeout)
        except CodeExecutionError as e:
            last_error = str(e)
            last_validation_feedback = []
            # Build a specific spelling-error hint for NameErrors so the retry
            # prompt gives the LLM a targeted "replace X with Y" instruction
            # rather than just the generic error string.
            last_name_error_hint = build_name_error_hint(raw_code, last_error)
            # Fall back to runtime error hint for non-NameError patterns
            # (e.g., index out of bounds, sort mismatch, arity errors).
            if last_name_error_hint is None:
                last_name_error_hint = build_runtime_error_hint(raw_code, last_error)
            attempt_record.code_exec_error = last_error
            attempts.append(attempt_record)
            logger.warning(
                "Phase 1 attempt %d/%d: code execution error: %s",
                attempt + 1, max_retries, e,
            )
            continue

        # Comprehensive output validation (Fix 6)
        validation_result = _validate_output(premises_nl, ctx)
        if validation_result:
            last_error, last_validation_feedback = validation_result
            attempt_record.validation_error = last_error
            attempts.append(attempt_record)
            logger.warning(
                "Phase 1 attempt %d/%d: %s", attempt + 1, max_retries, last_error
            )
            continue
        last_validation_feedback = []

        # Build Distinct() constraints for named entity constants (P0.2)
        bound_var_names: set[str] = ctx.get("bound_var_names", set())
        background_constraints = solver.build_distinct_constraints(
            ctx["namespace"], bound_var_names
        )
        # Pick up user-defined background_constraints from LLM code (RULE 10:
        # comparison predicate axioms like transitivity, antisymmetry).
        user_bg = ctx["namespace"].get("background_constraints")
        if user_bg and hasattr(user_bg, '__iter__'):
            for c in user_bg:
                if isinstance(c, z3.BoolRef):
                    background_constraints.append(c)
        # CWA axiom injection: detect and close semantic gaps under closed-world assumption
        if world_assumption == "cwa":
            cwa_axioms = build_cwa_constraints(
                namespace=ctx["namespace"],
                premises=list(ctx["premises"]),
                q=ctx["q"],
                bound_var_names=bound_var_names,
            )
            background_constraints.extend(cwa_axioms)
        # Solver uses NL premises + background constraints for entailment check
        solver_premises = list(ctx["premises"]) + background_constraints

        # Run solver
        try:
            if task_type == "three_class":
                verdict, model_info, model_info_q = solver.check_entailment_three_class(
                    solver_premises, ctx["q"]
                )
            else:
                verdict, model_info = solver.check_entailment(solver_premises, ctx["q"])
                model_info_q = None
        except Exception as e:
            last_error = str(e)
            last_validation_feedback = []
            attempt_record.solver_error = last_error
            attempts.append(attempt_record)
            logger.warning(
                "Phase 1 attempt %d/%d: solver error: %s", attempt + 1, max_retries, e
            )
            continue

        attempt_record.verdict = verdict
        attempts.append(attempt_record)

        # Vacuous model check: did Z3 find a world where every rule-derived,
        # conclusion-relevant predicate is False for every named entity?
        # This is the semantic signature of all dangling-conditional failures —
        # regardless of syntactic form (top-level ForAll, And-nested ForAll,
        # circular chains A→B/B→A, entity-mismatch groundings) — because they
        # all result in the proof-chain predicates being vacuously False.
        #
        # For Uncertain: check model_info_q (P∧q model).  q passing trivially
        # because everything is False is the telltale sign of weak formalization.
        # For other non-Entailed verdicts: check model_info (P∧¬q model).
        # Fire at most once, never on the last attempt.
        if not _vacuous_check_done and attempt < max_retries - 1:
            _model_to_check = (
                model_info_q
                if (verdict == VERDICT_UNCERTAIN and model_info_q is not None)
                else model_info
            )
            if _model_to_check is not None:
                try:
                    vacuous, always_false_preds = _check_model_vacuousness(
                        _model_to_check,
                        list(ctx["premises"]),
                        ctx["q"],
                        ctx["namespace"],
                        bound_var_names,
                    )
                except Exception as _vac_exc:
                    logger.debug("Vacuousness check skipped (exception): %s", _vac_exc)
                    vacuous = False
                    always_false_preds = []
                if vacuous:
                    _vacuous_check_done = True
                    pred_list = ", ".join(f"`{p}`" for p in sorted(always_false_preds))
                    last_error = (
                        "Vacuous world detected: the solver found a model where "
                        "every rule-derived predicate is False for every named "
                        f"entity ({pred_list or 'all rule predicates'}). "
                        "All conditional rules are trivially satisfied because "
                        "their antecedents are always False — the formalization "
                        "is too weak to constrain the domain."
                    )
                    last_validation_feedback = [
                        "This happens when a rule's antecedent predicate is never "
                        "instantiated by any ground fact. Z3 satisfies the rule by "
                        "setting the predicate False everywhere.",
                        "Fix: for each rule ForAll([v], Implies(A(v), B(v))), add "
                        "at least one ground fact asserting a specific named entity "
                        "has property A — e.g. A(entity_name) listed in `premises`.",
                        "Do NOT express membership or participation purely as a "
                        "conditional rule without also adding a ground fact that "
                        "names the specific entity satisfying the antecedent.",
                    ]
                    attempts[-1].validation_error = last_error
                    logger.warning(
                        "Phase 1 attempt %d/%d: vacuous model — rule-derived "
                        "conclusion-relevant predicates all False [%s]. "
                        "Triggering retry.",
                        attempt + 1, max_retries, pred_list[:120],
                    )
                    continue

        logger.info("Phase 1 success (attempt %d): verdict=%s", attempt + 1, verdict)

        result = Phase1Result(
            verdict=verdict,
            premises=list(ctx["premises"]),       # NL-only, aligned with sentences
            background_constraints=background_constraints,
            bound_var_names=bound_var_names,
            q=ctx["q"],
            model_info=model_info,
            model_info_q=model_info_q,
            namespace=ctx["namespace"],
            raw_code=ctx["raw_code"],
            verdict_pre_bridge=verdict,           # snapshot before Phase 1.5 may change verdict
            attempts=attempts,
        )

        # Phase 1.5: predicate connectivity bridge check (only for non-Entailed)
        if enable_bridge and verdict != VERDICT_ENTAILED:
            result = await _run_bridge_check(
                result=result,
                premises_nl=premises_nl,
                conclusion_nl=conclusion_nl,
                llm=llm,
                solver=solver,
                prompt_engine=prompt_engine,
                task_type=task_type,
                bridge_retries=bridge_retries,
            )

        return result

    return Phase1Result(
        verdict=VERDICT_UNKNOWN,
        premises=[],
        background_constraints=[],
        bound_var_names=set(),
        q=None,
        model_info=None,
        model_info_q=None,
        namespace={},
        raw_code=raw_code,
        attempts=attempts,
        error=f"Phase 1 failed after {max_retries} attempts. Last error: {last_error}",
    )


# ---------------------------------------------------------------------------
# Phase 1.5: Predicate connectivity bridge check
# ---------------------------------------------------------------------------

async def _run_bridge_check(
    result: Phase1Result,
    premises_nl: list[str],
    conclusion_nl: str,
    llm: LLMClient,
    solver: Z3Solver,
    prompt_engine: PromptEngine,
    task_type: str,
    bridge_retries: int = 2,
) -> Phase1Result:
    """
    Phase 1.5: Detect and repair structurally disconnected premises.

    Checks if any premise's predicate set is unreachable from the conclusion
    through the predicate connectivity graph.  A premise is "disconnected" when
    its predicates share nothing (directly or transitively) with the predicates
    in the conclusion and the universal/existential rules.  This is the hallmark
    of an under-formalization: the LLM chose a literal predicate that doesn't
    join the proof chain.

    For each disconnected premise, asks the LLM to write a universal linking
    axiom (ForAll form) that connects the premise's predicate vocabulary to the
    rest of the proof graph.  The axiom is appended to background_constraints —
    the original premise is preserved unchanged.  The solver is re-run to obtain
    an updated verdict.

    This runs at most once per premise (no retry loop) and is designed to be
    lightweight — a single LLM call per disconnected premise.
    """
    disconnected = _find_disconnected_premises(result.premises, result.q)
    if not disconnected:
        return result

    logger.info(
        "Phase 1.5: %d disconnected premise(s) detected: %s",
        len(disconnected), disconnected,
    )

    premises = list(result.premises)
    background_constraints = list(result.background_constraints)
    changed = False

    for idx in disconnected:
        connected_preds = _get_connected_predicates(idx, premises, result.q)
        linking_axiom = await _repair_bridge_premise(
            idx=idx,
            nl_text=premises_nl[idx],
            current_formula=premises[idx],
            all_premises=premises,
            premises_nl=premises_nl,
            q=result.q,
            conclusion_nl=conclusion_nl,
            namespace=result.namespace,
            connected_preds=connected_preds,
            llm=llm,
            prompt_engine=prompt_engine,
            raw_code=result.raw_code,
            bridge_retries=bridge_retries,
        )
        if linking_axiom is not None:
            # Add as background constraint instead of replacing the original
            # premise.  This preserves the original NL→FOL alignment (premises[i]
            # always corresponds to sentence i) and prevents semantic corruption
            # where the replacement silently drops part of the original meaning.
            background_constraints.append(linking_axiom)
            changed = True
            logger.info(
                "Phase 1.5 idx=%d: bridge axiom added: %.80s (original preserved: %.60s)",
                idx, str(linking_axiom), str(premises[idx]),
            )
        else:
            logger.warning(
                "Phase 1.5 idx=%d: bridge repair failed, keeping original formula", idx
            )

    if not changed:
        return result

    # Re-run solver with original premises + augmented background constraints
    solver_premises = premises + background_constraints
    try:
        if task_type == "three_class":
            verdict, model_info, model_info_q = solver.check_entailment_three_class(
                solver_premises, result.q
            )
        else:
            verdict, model_info = solver.check_entailment(solver_premises, result.q)
            model_info_q = None
    except Exception as e:
        logger.warning("Phase 1.5: re-solve failed (%s), keeping original result", e)
        return result

    logger.info(
        "Phase 1.5: re-solve complete: verdict %s → %s", result.verdict, verdict
    )

    # Append bridge axiom notes to raw_code so subsequent Phase 5 prompts
    # reflect the added linking axioms.
    new_axioms = background_constraints[len(result.background_constraints):]
    repair_notes = [
        f"# Phase 1.5 bridge axiom: {str(ax)}"
        for ax in new_axioms
    ]
    updated_raw_code = result.raw_code
    if repair_notes:
        updated_raw_code = (
            result.raw_code.rstrip()
            + "\n\n# ---- Phase 1.5 bridge axioms ----\n"
            + "\n".join(repair_notes)
        )

    return Phase1Result(
        verdict=verdict,
        premises=premises,
        background_constraints=background_constraints,
        bound_var_names=result.bound_var_names,
        q=result.q,
        model_info=model_info,
        model_info_q=model_info_q,
        namespace=result.namespace,
        raw_code=updated_raw_code,
        verdict_pre_bridge=result.verdict_pre_bridge,  # preserve original pre-bridge snapshot
        attempts=result.attempts,
        error=None,
    )


async def _repair_bridge_premise(
    idx: int,
    nl_text: str,
    current_formula: z3.ExprRef,
    all_premises: list[z3.ExprRef],
    premises_nl: list[str],
    q: z3.ExprRef,
    conclusion_nl: str,
    namespace: dict[str, Any],
    connected_preds: set[str],
    llm: LLMClient,
    prompt_engine: PromptEngine,
    raw_code: str,
    bridge_retries: int = 2,
) -> z3.ExprRef | None:
    """
    Ask the LLM to reformulate a single disconnected premise so its predicate
    vocabulary connects to the rest of the proof chain.

    Returns the new z3.ExprRef on success, or None if all attempts fail.
    """
    orphaned_preds = sorted(_extract_predicate_names(current_formula))
    other_premises = [
        (i, premises_nl[i], str(all_premises[i]))
        for i in range(len(all_premises))
        if i != idx
    ]

    messages: list[dict] = [{
        "role": "user",
        "content": prompt_engine.render(
            "phase1_bridge.j2",
            raw_code=raw_code,
            premise_index=idx,
            nl_text=nl_text,
            current_formula=str(current_formula),
            orphaned_preds=orphaned_preds,
            connected_preds=sorted(connected_preds),
            other_premises=other_premises,
            conclusion_nl=conclusion_nl,
            conclusion_formula=str(q),
        ),
    }]

    raw_output = ""
    for attempt in range(bridge_retries):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        f"Error: {last_error}\n\n"
                        "Output ONLY a valid Z3 boolean expression using identifiers "
                        "already declared in the code. No imports, no assignments."
                    ),
                },
            ]

        raw_output = await llm.complete_with_retry(messages)
        expr_str = _strip_fences(raw_output)
        if not expr_str:
            last_error = "Empty output"
            continue

        # P2: allow LLM to explicitly decline bridging
        if expr_str.strip().upper() == "NONE":
            logger.info("Phase 1.5 idx=%d: LLM returned NONE (no bridge needed)", idx)
            return None

        formula, error = _eval_bridge_expr(expr_str, namespace)
        if error or formula is None:
            last_error = error or "Returned None"
            continue

        # Guard: bridge output must be a universally/existentially quantified
        # linking axiom, not a ground fact.  Ground facts would only make sense
        # as premise replacements, but we now add bridge outputs as background
        # constraints — a ground fact there would be semantically wrong (asserting
        # an additional individual claim not in the original NL).
        if not z3.is_quantifier(formula):
            last_error = (
                "The bridge output must be a universally quantified linking axiom "
                "(ForAll/Exists), not a ground fact. Write a general rule connecting "
                "the orphaned predicate to the connected vocabulary, e.g. "
                "`ForAll([x], Implies(orphan_pred(x), connected_pred(x)))`."
            )
            logger.debug(
                "Phase 1.5 idx=%d attempt %d: rejected non-quantified bridge output",
                idx, attempt + 1,
            )
            continue

        return formula

    return None


from cgbv.core.gap_analysis import (
    extract_predicate_names as _extract_predicate_names,
    find_disconnected_premises as _find_disconnected_premises,
    get_connected_predicates as _get_connected_predicates,
)


# ---------------------------------------------------------------------------
# Phase 1 Targeted Re-formalization
# ---------------------------------------------------------------------------

async def run_phase1_targeted(
    original_code: str,
    failed_repairs: list[tuple[Mismatch, str | None]],
    premises_nl: list[str],
    conclusion_nl: str,
    llm: LLMClient,
    solver: Z3Solver,
    prompt_engine: PromptEngine,
    task_type: str = "entailment",
    code_exec_timeout: int = 30,
    world_assumption: str = "owa",
    max_retries: int = 2,
) -> Phase1Result | None:
    """
    Phase 1 targeted re-formalization: rewrite the entire Z3 theory when
    Phase 5 repair fails to fix detected mismatches.

    Unlike formula-level repair (Phase 5), this rewrites the whole theory,
    allowing structural changes like adding new sorts, predicates, or entity
    constants that the original formalization missed.

    Args:
        original_code: the current Z3-Python code (may include Phase 5 repair notes)
        failed_repairs: list of (Mismatch, repair_error_str | None) for mismatches
                       that Phase 5 could not fix
        premises_nl: original NL premises
        conclusion_nl: original NL conclusion
        llm: LLM client
        solver: Z3 solver
        prompt_engine: for rendering the reformalize template
        task_type: "entailment" or "three_class"
        code_exec_timeout: timeout for executing the re-formalized code
        world_assumption: "owa" or "cwa"
        max_retries: number of retry attempts on code execution errors

    Returns:
        Phase1Result on success, None if all retries fail.
    """
    # Build mismatch data for the template
    mismatch_data = []
    for m, repair_error in failed_repairs:
        mismatch_data.append({
            "sentence_index": m.sentence_index,
            "nl_sentence": m.nl_sentence,
            "mismatch_type": m.mismatch_type,
            "fol_truth": m.fol_truth,
            "grounded_truth": m.grounded_truth,
            "fol_formula": m.fol_formula_str,
            "grounded_formula": m.grounded_formula,
            "repair_error": repair_error,
        })

    user_content = prompt_engine.render(
        "phase1_reformalize.j2",
        original_code=original_code,
        failed_mismatches=mismatch_data,
    )
    messages: list[dict] = [{"role": "user", "content": user_content}]
    raw_code = ""
    last_error = ""

    for attempt in range(max_retries):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_code},
                {
                    "role": "user",
                    "content": (
                        f"Error: {last_error}\n\n"
                        "Fix the error and output the corrected COMPLETE Z3-Python code. "
                        "No markdown fences, no explanations."
                    ),
                },
            ]

        raw_code = await llm.complete_with_retry(messages)

        try:
            ctx = execute_z3_code(raw_code, timeout_seconds=code_exec_timeout)
        except CodeExecutionError as e:
            last_error = str(e)
            logger.warning(
                "Phase 1 targeted re-formalization attempt %d/%d: code error: %s",
                attempt + 1, max_retries, e,
            )
            continue

        # Validate output structure
        validation_result = _validate_output(premises_nl, ctx)
        if validation_result:
            last_error = validation_result[0]
            logger.warning(
                "Phase 1 targeted re-formalization attempt %d/%d: validation error: %s",
                attempt + 1, max_retries, last_error,
            )
            continue

        # Build constraints and solve
        bound_var_names: set[str] = ctx.get("bound_var_names", set())
        background_constraints = solver.build_distinct_constraints(
            ctx["namespace"], bound_var_names
        )
        # Pick up user-defined background_constraints from LLM code (RULE 10)
        user_bg = ctx["namespace"].get("background_constraints")
        if user_bg and hasattr(user_bg, '__iter__'):
            for c in user_bg:
                if isinstance(c, z3.BoolRef):
                    background_constraints.append(c)
        if world_assumption == "cwa":
            cwa_axioms = build_cwa_constraints(
                namespace=ctx["namespace"],
                premises=list(ctx["premises"]),
                q=ctx["q"],
                bound_var_names=bound_var_names,
            )
            background_constraints.extend(cwa_axioms)

        solver_premises = list(ctx["premises"]) + background_constraints

        try:
            if task_type == "three_class":
                verdict, model_info, model_info_q = solver.check_entailment_three_class(
                    solver_premises, ctx["q"]
                )
            else:
                verdict, model_info = solver.check_entailment(solver_premises, ctx["q"])
                model_info_q = None
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Phase 1 targeted re-formalization attempt %d/%d: solver error: %s",
                attempt + 1, max_retries, e,
            )
            continue

        logger.info(
            "Phase 1 targeted re-formalization success (attempt %d): verdict=%s",
            attempt + 1, verdict,
        )

        return Phase1Result(
            verdict=verdict,
            premises=list(ctx["premises"]),
            background_constraints=background_constraints,
            bound_var_names=bound_var_names,
            q=ctx["q"],
            model_info=model_info,
            model_info_q=model_info_q,
            namespace=ctx["namespace"],
            raw_code=ctx["raw_code"],
            verdict_pre_bridge="",  # not applicable for re-formalization
            attempts=[],
        )

    logger.warning(
        "Phase 1 targeted re-formalization failed after %d attempts", max_retries,
    )
    return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_messages(
    premises_nl: list[str],
    conclusion_nl: str,
    prompt_engine: PromptEngine,
    dataset: str,
    world_assumption: str = "owa",
) -> list[dict]:
    user_content = prompt_engine.render(
        "phase1_formalize.j2",
        dataset=dataset,
        premises=premises_nl,
        conclusion=conclusion_nl,
        world_assumption=world_assumption,
    )
    return [{"role": "user", "content": user_content}]


def _build_retry_message(
    premises_nl: list[str],
    conclusion_nl: str,
    raw_code: str,
    last_error: str,
    validation_feedback: list[str],
    attempt_num: int,
    max_retries: int,
    prompt_engine: PromptEngine,
    name_error_hint: str | None = None,
) -> dict[str, str]:
    user_content = prompt_engine.render(
        "phase1_retry.j2",
        premises=premises_nl,
        conclusion=conclusion_nl,
        previous_code=_normalise_code_for_prompt(raw_code),
        last_error=last_error,
        validation_feedback=validation_feedback,
        name_error_hint=name_error_hint,
        attempt_num=attempt_num,
        max_retries=max_retries,
    )
    return {"role": "user", "content": user_content}


# ---------------------------------------------------------------------------
# Validation (Fix 6)
# ---------------------------------------------------------------------------

def _validate_output(
    premises_nl: list[str],
    ctx: dict[str, Any],
) -> tuple[str, list[str]] | None:
    """
    Comprehensive validation of Phase 1 exec() output.

    Checks (in order):
    1. `premises` is a sized sequence with the correct length
    2. Every entry in `premises` is a z3.BoolRef
    3. `q` is a z3.BoolRef

    Returns (error_msg, feedback_list) on failure, None on success.
    """
    expected_count = len(premises_nl)
    premises = ctx.get("premises")
    q = ctx.get("q")

    # --- Check 1: Length ---
    try:
        actual_count = len(premises)
    except TypeError:
        return (
            "Validation error: `premises` must be a sized sequence with one top-level "
            "formula per numbered NL premise.",
            [
                f"The original task has {expected_count} numbered premises.",
                "Your code did not define `premises` as a sized sequence.",
                "Rewrite so `premises` is a list of formulas in the same order as the "
                "numbered premises.",
            ],
        )

    if actual_count != expected_count:
        return (
            f"Validation error: `premises` must contain exactly one top-level formula "
            f"per numbered NL premise (expected {expected_count}, got {actual_count}).",
            [
                f"The original task has {expected_count} numbered premises.",
                f"Your code defined {actual_count} top-level formulas inside `premises`.",
                "Rewrite only the premise formalization so `premises` contains exactly "
                "one formula per numbered premise, in the same order.",
                "If one numbered premise is internally conjunctive or disjunctive, keep "
                "that structure inside a single formula instead of splitting or omitting it.",
            ],
        )

    # --- Check 2: BoolRef for each premise ---
    for i, f in enumerate(premises):
        if not isinstance(f, z3.BoolRef):
            return (
                f"Validation error: premises[{i}] is not a Z3 boolean formula "
                f"(got {type(f).__name__}: {str(f)[:80]}). "
                "All entries in `premises` must be BoolRef (boolean Z3 expressions).",
                [
                    f"premises[{i}] evaluates to a non-boolean Z3 expression.",
                    "Non-boolean functions (e.g. cost(x), rent(x)) cannot be used "
                    "directly as formulas. Wrap them in an equality or comparison: "
                    "e.g. ForAll([x], Implies(is_apt(x), Eq(rent(x), low_rent))).",
                    "Ensure every entry in `premises` is a formula that Z3 treats as Bool.",
                ],
            )

    # --- Check 3: BoolRef for q ---
    if q is None:
        return (
            "Validation error: `q` is not defined.",
            ["Define `q` as the Z3 formula for the conclusion (a BoolRef expression)."],
        )
    if not isinstance(q, z3.BoolRef):
        return (
            f"Validation error: `q` is not a Z3 boolean formula "
            f"(got {type(q).__name__}: {str(q)[:80]}).",
            [
                "`q` must be a BoolRef formula for the conclusion.",
                "Non-boolean function applications cannot be used as `q` directly.",
            ],
        )

    return None


def _check_model_vacuousness(
    model: z3.ModelRef,
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
    namespace: dict[str, Any],
    bound_var_names: set[str],
) -> tuple[bool, list[str]]:
    """
    Check whether a Z3 model is *rule-vacuous* with respect to the proof task.

    A model is rule-vacuous when every predicate that is:
      (a) reachable from the conclusion via the predicate co-occurrence graph, AND
      (b) NOT directly grounded by a ground fact in the premises
    evaluates to False for every named entity combination in the model.

    This is the semantic signature shared by ALL dangling-conditional failure
    modes, regardless of syntactic form:
      - Simple ungrounded predicates (top-level ForAll)
      - And-nested ForAll: And(ground_fact, ForAll(Implies(A, B)))
      - Circular chains: A→B, B→A, neither grounded
      - Entity-mismatch: predicate grounded for a different entity than the rule needs

    Two-stage approach:
      Stage 1 (Z3 formula graph): BFS from conclusion predicates; exclude
        predicates already grounded by ground facts — avoids false positives
        from irrelevant or already-grounded predicates.
      Stage 2 (model evaluation): evaluate remaining predicates on every named
        entity combination; vacuous if all are always-False.

    Returns (is_vacuous, list_of_always_false_predicate_names).
    """
    import itertools

    # --- Stage 1: Scope to proof-relevant, rule-derived predicates ---

    # Predicates directly asserted by ground atoms are excluded — they are
    # already concretely True for specific entities and don't contribute to
    # vacuousness.  We walk each premise stopping at quantifier boundaries:
    # a Bool predicate call where every argument is a 0-arity named constant
    # (no '!' in name) is a ground atom, regardless of whether the containing
    # formula also has quantified sub-formulae (e.g. And(p(a), ForAll(...))).
    ground_pred_names: set[str] = set()

    def _ground_walk(expr: z3.ExprRef) -> None:
        if z3.is_quantifier(expr):
            return  # stop — do not enter rule bodies
        if z3.is_app(expr):
            decl = expr.decl()
            if (decl.arity() >= 1
                    and decl.range().kind() == z3.Z3_BOOL_SORT
                    and all(
                        z3.is_app(ch) and ch.decl().arity() == 0
                        and '!' not in ch.decl().name()
                        for ch in expr.children()
                    )):
                ground_pred_names.add(decl.name())
            for child in expr.children():
                _ground_walk(child)

    for f in premises:
        _ground_walk(f)

    # BFS from conclusion predicates through predicate co-occurrence graph.
    pred_graph: dict[str, set[str]] = {}
    for f in list(premises) + [q]:
        preds = _extract_predicate_names(f)
        for p in preds:
            pred_graph.setdefault(p, set()).update(preds - {p})

    conclusion_preds = _extract_predicate_names(q)
    reachable: set[str] = set(conclusion_preds)
    frontier: list[str] = list(conclusion_preds)
    while frontier:
        cur = frontier.pop()
        for nb in pred_graph.get(cur, set()):
            if nb not in reachable:
                reachable.add(nb)
                frontier.append(nb)

    # Target = conclusion-reachable AND not covered by a ground fact.
    target_pred_names: set[str] = reachable - ground_pred_names
    if not target_pred_names:
        return False, []   # all proof-relevant predicates are ground-fact-grounded

    # --- Stage 2: Evaluate target predicates in the model ---

    # Collect named entity constants by sort (exclude bound vars and Sort!val!N).
    entities_by_sort: dict[str, list] = {}
    for name, obj in namespace.items():
        if (
            isinstance(obj, z3.ExprRef)
            and z3.is_const(obj)
            and name not in bound_var_names
            and obj.sort().kind() == z3.Z3_UNINTERPRETED_SORT
            and '!' not in obj.decl().name()
        ):
            entities_by_sort.setdefault(obj.sort().name(), []).append(obj)

    if not entities_by_sort:
        return False, []

    # Collect FuncDeclRef objects for target predicates only.
    pred_funcs: dict[str, z3.FuncDeclRef] = {
        obj.name(): obj
        for obj in namespace.values()
        if (isinstance(obj, z3.FuncDeclRef)
            and obj.arity() >= 1
            and obj.range().kind() == z3.Z3_BOOL_SORT
            and obj.name() in target_pred_names)
    }
    if not pred_funcs:
        return False, []

    checked: list[str] = []
    always_false: list[str] = []

    for pred_name, pred_func in pred_funcs.items():
        entity_lists: list[list] = []
        all_covered = True
        for i in range(pred_func.arity()):
            sort_name = pred_func.domain(i).name()
            if sort_name not in entities_by_sort:
                all_covered = False
                break
            entity_lists.append(entities_by_sort[sort_name])
        if not all_covered:
            continue   # sort has no named entities — skip

        checked.append(pred_name)
        pred_is_always_false = True

        for entity_tuple in itertools.product(*entity_lists):
            try:
                val = model.eval(pred_func(*entity_tuple), model_completion=True)
                if z3.is_true(val):
                    pred_is_always_false = False
                    break
            except Exception:
                pass   # type mismatch or eval error — treat as False

        if pred_is_always_false:
            always_false.append(pred_name)

    if not checked:
        return False, []   # couldn't evaluate any target predicate

    return len(always_false) == len(checked), always_false


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _normalise_code_for_prompt(raw_code: str) -> str:
    code = raw_code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        code = "" if first_newline < 0 else code[first_newline + 1:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _strip_fences(raw: str) -> str:
    """Strip markdown fences and inline backticks from LLM output."""
    raw = re.sub(r'^```(?:python)?\s*\n', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\n```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    if raw.startswith('`') and raw.endswith('`') and len(raw) > 1:
        raw = raw[1:-1].strip()
    return raw


def _eval_bridge_expr(
    expr_str: str,
    namespace: dict[str, Any],
) -> tuple[z3.ExprRef | None, str | None]:
    """
    Evaluate a Z3-Python expression string in the Phase 1 namespace.
    Returns (formula, None) on success or (None, error_str) on failure.
    """
    if not expr_str:
        return None, "Empty expression"
    try:
        import z3 as _z3
        eval_ns = dict(namespace)
        for name in dir(_z3):
            if not name.startswith("_") and name not in eval_ns:
                eval_ns[name] = getattr(_z3, name)
        result = eval(expr_str, eval_ns)  # noqa: S307
        if not isinstance(result, _z3.BoolRef):
            return None, (
                f"Expression is not a Z3 boolean formula "
                f"(got {type(result).__name__})"
            )
        return result, None
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Evaluation error: {e}"


def _snapshot_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Copy chat messages so attempt traces remain stable after retries mutate the list."""
    return [
        {
            "role": str(m.get("role", "")),
            "content": str(m.get("content", "")),
        }
        for m in messages
    ]
