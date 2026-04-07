from __future__ import annotations

import json
import logging
import re
import ast
from dataclasses import dataclass, field, replace
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
class Phase1Diagnostic:
    """Best-effort structural diagnosis for a failed Phase 1 attempt."""
    raw_error: str = ""
    failure_stage: str = ""  # exec | validation | solver
    source_slots: list[str] = field(default_factory=list)
    offending_symbols: list[str] = field(default_factory=list)
    preserve_constraints: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    attempt_fingerprint: str = ""


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
    diagnostic: dict[str, Any] | None = None


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
    repeated_failure: bool = False     # True iff compile/validation failure repeated structurally
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
    last_diagnostic = Phase1Diagnostic()
    raw_code = ""
    attempts: list[Phase1Attempt] = []
    failure_fingerprints: list[str] = []
    repeated_failure_detected = False
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
                    diagnostic=last_diagnostic,
                    repeated_failure=repeated_failure_detected,
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

        # Fix D: AST-based sort consistency check (before execution).
        # Catches sort mismatches statically, providing precise diagnostics
        # that guide the retry instead of opaque Z3 runtime errors.
        sort_error = check_z3_sort_consistency(raw_code)
        if sort_error:
            last_error = sort_error
            last_validation_feedback = []
            last_name_error_hint = sort_error  # inject as specific diagnosis
            attempt_record.code_exec_error = sort_error
            last_diagnostic = _build_phase1_diagnostic(
                raw_code=raw_code,
                failure_stage="sort_check",
                raw_error=sort_error,
                validation_feedback=[],
                name_error_hint=sort_error,
                premises_nl=premises_nl,
                conclusion_nl=conclusion_nl,
            )
            repeated_failure_detected = bool(
                failure_fingerprints
                and failure_fingerprints[-1] == last_diagnostic.attempt_fingerprint
            )
            failure_fingerprints.append(last_diagnostic.attempt_fingerprint)
            attempt_record.diagnostic = _diagnostic_to_dict(last_diagnostic)
            attempts.append(attempt_record)
            logger.warning(
                "Phase 1 attempt %d/%d: static sort mismatch: %s",
                attempt + 1, max_retries, sort_error[:200],
            )
            continue

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
            last_diagnostic = _build_phase1_diagnostic(
                raw_code=raw_code,
                failure_stage="exec",
                raw_error=last_error,
                validation_feedback=[],
                name_error_hint=last_name_error_hint,
                premises_nl=premises_nl,
                conclusion_nl=conclusion_nl,
            )
            repeated_failure_detected = bool(
                failure_fingerprints
                and failure_fingerprints[-1] == last_diagnostic.attempt_fingerprint
            )
            failure_fingerprints.append(last_diagnostic.attempt_fingerprint)
            attempt_record.diagnostic = _diagnostic_to_dict(last_diagnostic)
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
            last_diagnostic = _build_phase1_diagnostic(
                raw_code=raw_code,
                failure_stage="validation",
                raw_error=last_error,
                validation_feedback=last_validation_feedback,
                name_error_hint=last_name_error_hint,
                premises_nl=premises_nl,
                conclusion_nl=conclusion_nl,
            )
            repeated_failure_detected = bool(
                failure_fingerprints
                and failure_fingerprints[-1] == last_diagnostic.attempt_fingerprint
            )
            failure_fingerprints.append(last_diagnostic.attempt_fingerprint)
            attempt_record.diagnostic = _diagnostic_to_dict(last_diagnostic)
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
            last_diagnostic = _build_phase1_diagnostic(
                raw_code=raw_code,
                failure_stage="solver",
                raw_error=last_error,
                validation_feedback=[],
                name_error_hint=last_name_error_hint,
                premises_nl=premises_nl,
                conclusion_nl=conclusion_nl,
            )
            repeated_failure_detected = bool(
                failure_fingerprints
                and failure_fingerprints[-1] == last_diagnostic.attempt_fingerprint
            )
            failure_fingerprints.append(last_diagnostic.attempt_fingerprint)
            attempt_record.diagnostic = _diagnostic_to_dict(last_diagnostic)
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
                        "This usually means the current code failed to preserve an "
                        "existing NL grounding or link, so a proof-relevant rule "
                        "chain never activates and Z3 sets the antecedent predicate "
                        "False everywhere.",
                        "Fix only if the original NL already licenses it: rewrite "
                        "the affected numbered premise or move a general rule into "
                        "`background_constraints`. Do NOT add a new premise, a new "
                        "ground fact, or an extra conjunct that is not stated in the NL.",
                        "If the original NL premise is a plain disjunction, keep it "
                        "a plain disjunction. Do NOT turn it into patterns like "
                        "`And(Or(...), extra_fact)` just to satisfy the solver.",
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
            repeated_failure=repeated_failure_detected or _has_repeated_phase1_failure(attempts),
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

        # Phase 1.6: identifier canonicalization
        result = await _run_canonicalization(
                result=result,
                premises_nl=premises_nl,
                llm=llm,
                solver=solver,
                prompt_engine=prompt_engine,
                task_type=task_type,
                code_exec_timeout=code_exec_timeout,
                world_assumption=world_assumption,
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
        repeated_failure=repeated_failure_detected or _has_repeated_phase1_failure(attempts),
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
            solver=solver,
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
        repeated_failure=result.repeated_failure,
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
    solver: "Z3Solver | None" = None,
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

    initial_render = prompt_engine.render(
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
        retry_context=None,
    )
    messages: list[dict] = [{"role": "user", "content": initial_render}]

    raw_output = ""
    last_error: str = ""
    _consistency_rejected = False
    _rejected_bridge_str: str = ""
    _rejected_bridge_reason: str = ""

    for attempt in range(bridge_retries):
        if attempt > 0:
            if _consistency_rejected:
                # Re-render the full bridge prompt with retry_context so the LLM
                # has complete context about why the previous bridge was rejected.
                retry_render = prompt_engine.render(
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
                    retry_context={
                        "rejected_bridge": _rejected_bridge_str,
                        "reason": _rejected_bridge_reason,
                    },
                )
                messages = messages + [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": retry_render},
                ]
                _consistency_rejected = False
            else:
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

        # Bridge consistency gate: reject axioms that make the premise set UNSAT.
        # A bridge causing UNSAT would allow Z3 to vacuously derive anything
        # (ex falso quodlibet), producing a spurious Entailed verdict.
        if solver is not None:
            s_test = z3.Solver()
            for p in all_premises:
                s_test.add(p)
            s_test.add(formula)
            if s_test.check() == z3.unsat:
                _rejected_bridge_str = str(formula)
                _rejected_bridge_reason = "inconsistent"
                _consistency_rejected = True
                last_error = (
                    f"CONSISTENCY GATE: bridge `{formula}` combined with existing "
                    f"premises is UNSAT — this bridge contradicts the premises. "
                    f"Generate a different bridge that does not conflict."
                )
                logger.warning(
                    "Phase 1.5 idx=%d attempt %d: bridge consistency gate REJECTED "
                    "(inconsistent with premises): %.80s",
                    idx, attempt + 1, str(formula),
                )
                continue

        return formula

    return None


from cgbv.core.gap_analysis import (
    extract_predicate_names as _extract_predicate_names,
    find_disconnected_premises as _find_disconnected_premises,
    get_connected_predicates as _get_connected_predicates,
    is_ground_fact as _is_ground_fact,
)


# ---------------------------------------------------------------------------
# Phase 1.6: Identifier canonicalization
# ---------------------------------------------------------------------------

async def _run_canonicalization(
    result: Phase1Result,
    premises_nl: list[str],
    llm: LLMClient,
    solver: Z3Solver,
    prompt_engine: PromptEngine,
    task_type: str,
    code_exec_timeout: int,
    world_assumption: str = "owa",
) -> Phase1Result:
    """
    Phase 1.6: LLM-driven identifier canonicalization.

    Asks the LLM to compare all symbolic identifiers in the Z3 code against the
    original NL premises and identify any pair of identifiers that refer to the
    same real-world concept but were assigned different names. If duplicates are
    found, applies whole-word substitutions, re-executes, and re-solves.

    If re-execution fails or no changes are needed, returns the original result.
    """
    user_content = prompt_engine.render(
        "phase1_canon.j2",
        premises=premises_nl,
        raw_code=result.raw_code,
    )
    raw = (await llm.complete_with_retry([{"role": "user", "content": user_content}])).strip()

    if not raw or raw.upper().startswith("OK") or not raw.startswith("{"):
        return result

    try:
        subst_map: dict[str, str] = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Phase 1.6: failed to parse substitution map: %.100s", raw)
        return result

    if not subst_map:
        return result

    # Apply whole-word substitutions to raw code
    new_code = result.raw_code
    for old, new_name in subst_map.items():
        if old == new_name:
            continue
        new_code = re.sub(r'\b' + re.escape(old) + r'\b', new_name, new_code)

    if new_code == result.raw_code:
        return result

    try:
        ctx = execute_z3_code(new_code, timeout_seconds=code_exec_timeout)
    except CodeExecutionError as e:
        logger.warning("Phase 1.6: re-exec failed after canonicalization, reverting: %s", e)
        return result

    # Tautology guard: reject substitution if any premise became trivially True.
    # This happens when a premise Implies(A(x), B(x)) exists and A→B was merged,
    # turning it into Implies(B(x), B(x)) = always True, destroying the inference.
    for i, prem in enumerate(ctx.get("premises", [])):
        try:
            s_check = z3.Solver()
            s_check.add(z3.Not(prem))
            if s_check.check() == z3.unsat:
                logger.warning(
                    "Phase 1.6: substitution %s created tautological premise[%d], reverting",
                    subst_map, i,
                )
                return result
        except Exception:
            pass  # if check fails, continue — don't block on it

    # Rebuild background constraints for the renamed namespace
    bound_var_names: set[str] = ctx.get("bound_var_names", set())
    bg = solver.build_distinct_constraints(ctx["namespace"], bound_var_names)
    user_bg = ctx["namespace"].get("background_constraints")
    if user_bg and hasattr(user_bg, '__iter__'):
        for c in user_bg:
            if isinstance(c, z3.BoolRef):
                bg.append(c)
    if world_assumption == "cwa":
        cwa_axioms = build_cwa_constraints(
            namespace=ctx["namespace"],
            premises=list(ctx["premises"]),
            q=ctx["q"],
            bound_var_names=bound_var_names,
        )
        bg.extend(cwa_axioms)

    solver_premises = list(ctx["premises"]) + bg

    try:
        if task_type == "three_class":
            new_verdict, new_model_info, new_model_info_q = solver.check_entailment_three_class(
                solver_premises, ctx["q"]
            )
        else:
            new_verdict, new_model_info = solver.check_entailment(solver_premises, ctx["q"])
            new_model_info_q = None
    except Exception as e:
        logger.warning("Phase 1.6: solver error after canonicalization, reverting: %s", e)
        return result

    logger.info(
        "Phase 1.6: unified %d identifier(s) %s; verdict %s → %s",
        len(subst_map), list(subst_map.keys()), result.verdict, new_verdict,
    )

    return replace(
        result,
        raw_code=ctx["raw_code"],
        premises=list(ctx["premises"]),
        q=ctx["q"],
        background_constraints=bg,
        bound_var_names=bound_var_names,
        namespace=ctx["namespace"],
        verdict=new_verdict,
        model_info=new_model_info,
        model_info_q=new_model_info_q,
    )


# ---------------------------------------------------------------------------
# Phase 1 Targeted Re-formalization
# ---------------------------------------------------------------------------

async def run_phase1_targeted(
    original_code: str,
    failed_repairs: list[tuple[Any, str | None]],
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
            "sentence_index": getattr(m, "sentence_index", -1),
            "nl_sentence": getattr(m, "nl_sentence", ""),
            "mismatch_type": getattr(m, "mismatch_type", "semantic_drift"),
            "fol_truth": getattr(m, "fol_truth", None),
            "grounded_truth": getattr(m, "grounded_truth", None),
            "fol_formula": getattr(m, "fol_formula_str", getattr(m, "current_formula_str", "")),
            "grounded_formula": getattr(m, "grounded_formula", getattr(m, "audited_formula_str", "")),
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
            repeated_failure=False,
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
    diagnostic: Phase1Diagnostic | None = None,
    repeated_failure: bool = False,
) -> dict[str, str]:
    user_content = prompt_engine.render(
        "phase1_retry.j2",
        premises=premises_nl,
        conclusion=conclusion_nl,
        previous_code=_normalise_code_for_prompt(raw_code),
        last_error=last_error,
        validation_feedback=validation_feedback,
        name_error_hint=name_error_hint,
        diagnostic=_diagnostic_to_dict(diagnostic),
        repeated_failure=repeated_failure,
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
    2. Every entry in `premises` is a z3.BoolRef without raw/trivial bool literals
    3. Ground disjunctive premises are not strengthened with extra conjuncts
    4. `q` is a z3.BoolRef without raw/trivial bool literals

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
        if isinstance(f, bool):
            return (
                f"Validation error: premises[{i}] is a raw Python boolean "
                f"(`{f}`), not a Z3 formula.",
                [
                    f"premises[{i}] must be a Z3 BoolRef built from declared predicates.",
                    "Do not assign plain Python `True`/`False` to any premise.",
                    "Translate the NL premise into a predicate or relation formula instead.",
                ],
            )
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
        if _contains_boolean_literal(f):
            return (
                f"Validation error: premises[{i}] contains a literal boolean "
                f"constant ({str(f)[:80]}).",
                [
                    "Do not use literal `True`/`False` inside premise formulas.",
                    "Do not encode a premise as a tautology, contradiction, or helper "
                    "literal such as `True`, `False`, `BoolVal(True)`, or `If(True, ..., ...)`.",
                    "Use predicates and relations that are explicitly supported by the NL.",
                ],
            )
        if _is_strengthened_disjunctive_premise(premises_nl[i], f):
            return (
                f"Validation error: premises[{i}] strengthens a disjunctive NL premise "
                f"with an extra conjunction ({str(f)[:80]}).",
                [
                    "This numbered NL premise is a plain disjunction, not a conjunction.",
                    "Keep it as a disjunction such as `Or(...)` if the NL only says `or`.",
                    "Do not repair vacuousness by wrapping a disjunction as "
                    "`And(Or(...), extra_fact)` or by smuggling extra ground facts into it.",
                ],
            )

    # --- Check 4: BoolRef for q ---
    if q is None:
        return (
            "Validation error: `q` is not defined.",
            ["Define `q` as the Z3 formula for the conclusion (a BoolRef expression)."],
        )
    if isinstance(q, bool):
        return (
            f"Validation error: `q` is a raw Python boolean (`{q}`), not a Z3 formula.",
            [
                "`q` must be a BoolRef formula for the conclusion.",
                "Do not assign plain Python `True`/`False` to `q`.",
                "Translate the conclusion into declared predicates and relations instead.",
            ],
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
    if _contains_boolean_literal(q):
        return (
            f"Validation error: `q` contains a literal boolean constant "
            f"({str(q)[:80]}).",
            [
                "Do not use literal `True`/`False` inside `q`.",
                "The conclusion must be expressed with declared predicates and relations.",
            ],
        )

    return None


def _contains_boolean_literal(expr: z3.ExprRef) -> bool:
    """Return True if a Z3 formula contains an explicit True/False literal node."""
    found = False

    def _walk(node: z3.ExprRef) -> None:
        nonlocal found
        if found:
            return
        if z3.is_true(node) or z3.is_false(node):
            found = True
            return
        if z3.is_app(node):
            for child in node.children():
                _walk(child)
        elif z3.is_quantifier(node):
            _walk(node.body())

    _walk(expr)
    return found


def _is_strengthened_disjunctive_premise(
    premise_nl: str,
    formula: z3.BoolRef,
) -> bool:
    """
    Detect the specific repair hack `And(Or(...), extra_fact)` for NL premises
    that are plain disjunctions.
    """
    nl = premise_nl.lower()
    has_disjunction = re.search(r"\b(or|either)\b", nl) is not None
    has_conjunction = re.search(
        r"\b(and|both|together with|as well as|but also)\b",
        nl,
    ) is not None
    if not has_disjunction or has_conjunction:
        return False
    if not _is_ground_fact(formula) or not z3.is_and(formula):
        return False
    return any(z3.is_or(child) for child in formula.children())


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

    # If any conclusion-reachable predicate is already grounded by a concrete
    # premise, a non-Entailed countermodel may be perfectly legitimate rather
    # than evidence of an under-formalized proof chain. In that case, skip the
    # vacuousness retry heuristic instead of pushing the model to invent facts.
    if reachable & ground_pred_names:
        return False, []

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
# Retry diagnostics
# ---------------------------------------------------------------------------

def _build_phase1_diagnostic(
    raw_code: str,
    failure_stage: str,
    raw_error: str,
    validation_feedback: list[str],
    name_error_hint: str | None,
    premises_nl: list[str],
    conclusion_nl: str,
) -> Phase1Diagnostic:
    """
    Build a best-effort retry contract for Phase 1 failures.

    This is intentionally soft-structured: only `failure_stage` is mandatory.
    All other fields may be empty when the error text offers no reliable signal.
    """
    source_slots: list[str] = []
    offending_symbols: list[str] = []
    forbidden_patterns: list[str] = []

    err = raw_error or ""
    if re.search(r"premises\[(\d+)\]", err):
        source_slots.extend(
            f"premises[{m}]"
            for m in re.findall(r"premises\[(\d+)\]", err)
        )
    if "`q`" in err or re.search(r"\bq\b", err):
        source_slots.append("q")
    if "premises must contain exactly one top-level formula" in err:
        source_slots.append("premises")

    offending_symbols.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", name_error_hint or ""))
    m_name = re.search(r"name '(\w+)' is not defined", err)
    if m_name:
        offending_symbols.append(m_name.group(1))

    err_lower = err.lower()
    if "sort mismatch" in err_lower:
        forbidden_patterns.append("Do not use Bool predicates as values inside equality/comparison predicates.")
    if "literal boolean" in err_lower or "raw python boolean" in err_lower:
        forbidden_patterns.append("Do not place literal True/False inside premises or q.")
    if "exactly one top-level formula" in err_lower:
        forbidden_patterns.append("Do not change the number of top-level formulas in premises.")
    if "not a z3 boolean formula" in err_lower:
        forbidden_patterns.append("Every premise entry and q must evaluate to a BoolRef.")
    if "tautolog" in err_lower:
        forbidden_patterns.append("Do not collapse a numbered premise into a tautology or empty semantic shell.")

    preserve_constraints = [
        f"Keep exactly {len(premises_nl)} top-level formulas in `premises`, in NL order.",
        f"Keep `q` aligned with the same conclusion sentence: {conclusion_nl}",
        "Preserve valid declarations, constants, sorts, and predicates unless the error proves one is wrong.",
        "Reuse the existing symbol table whenever possible; do not rename working identifiers.",
    ]
    preserve_constraints.extend(validation_feedback[:3])

    fingerprint = _fingerprint_phase1_attempt(
        raw_code=raw_code,
        failure_stage=failure_stage,
        raw_error=raw_error,
        source_slots=source_slots,
    )

    return Phase1Diagnostic(
        raw_error=raw_error,
        failure_stage=failure_stage,
        source_slots=sorted(set(source_slots)),
        offending_symbols=sorted(set(offending_symbols)),
        preserve_constraints=preserve_constraints,
        forbidden_patterns=forbidden_patterns,
        attempt_fingerprint=fingerprint,
    )


def _fingerprint_phase1_attempt(
    raw_code: str,
    failure_stage: str,
    raw_error: str,
    source_slots: list[str],
) -> str:
    """Return a stable-ish fingerprint for repeated structural failures."""
    code_key = raw_code.strip()
    try:
        tree = ast.parse(_normalise_code_for_prompt(raw_code))
        code_key = ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:
        code_key = re.sub(r"\s+", " ", code_key)
    error_key = re.sub(r"\s+", " ", (raw_error or "").strip().lower())
    error_key = re.sub(r"`[^`]+`", "`sym`", error_key)
    return "|".join([
        failure_stage,
        ",".join(sorted(source_slots)),
        error_key[:240],
        code_key[:1200],
    ])


def _diagnostic_to_dict(diagnostic: Phase1Diagnostic | None) -> dict[str, Any] | None:
    if diagnostic is None:
        return None
    return {
        "raw_error": diagnostic.raw_error,
        "failure_stage": diagnostic.failure_stage,
        "source_slots": list(diagnostic.source_slots),
        "offending_symbols": list(diagnostic.offending_symbols),
        "preserve_constraints": list(diagnostic.preserve_constraints),
        "forbidden_patterns": list(diagnostic.forbidden_patterns),
        "attempt_fingerprint": diagnostic.attempt_fingerprint,
    }


def _has_repeated_phase1_failure(attempts: list[Phase1Attempt]) -> bool:
    fingerprints = [
        (a.diagnostic or {}).get("attempt_fingerprint", "")
        for a in attempts
        if a.diagnostic
    ]
    return any(
        fingerprints[i] and fingerprints[i] == fingerprints[i - 1]
        for i in range(1, len(fingerprints))
    )


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


def check_z3_sort_consistency(code: str) -> str | None:
    """Parse Phase 1 code (AST only) and detect Z3 sort mismatches statically.

    Detects: a Function declared as returning BoolSort() being passed as an
    argument to another Function that expects a custom Sort parameter.

    Returns None if no error found, or a diagnostic string describing the
    sort mismatch.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # let execute_z3_code handle syntax errors

    # Pass 1: collect Function declarations and their sort signatures.
    # Function('name', SortA, SortB, BoolSort()) → name: [SortA, SortB, BoolSort]
    func_sorts: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            # Function('name', Sort1, Sort2, ..., BoolSort())
            if (isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and _ast_call_name(value) == "Function"):
                args = value.args
                if len(args) >= 2:
                    sort_names = []
                    for a in args[1:]:
                        sort_names.append(_ast_sort_name(a))
                    func_sorts[target.id] = sort_names

    if not func_sorts:
        return None

    # Pass 2: find Function calls where a BoolSort-returning function is
    # passed as argument to a slot that expects a custom Sort.
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        caller_name = _ast_call_name(node)
        if caller_name not in func_sorts:
            continue
        caller_sig = func_sorts[caller_name]
        # caller_sig[-1] is the return sort; caller_sig[:-1] are parameter sorts
        param_sorts = caller_sig[:-1]
        for i, arg in enumerate(node.args):
            if i >= len(param_sorts):
                break
            expected_sort = param_sorts[i]
            if expected_sort == "BoolSort":
                continue  # BoolSort parameter — any bool expression is fine
            # Check if the arg is a call to a function that returns BoolSort
            if isinstance(arg, ast.Call):
                arg_func_name = _ast_call_name(arg)
                if (arg_func_name in func_sorts
                    and func_sorts[arg_func_name][-1] == "BoolSort"):
                    lineno = getattr(node, 'lineno', '?')
                    errors.append(
                        f"line {lineno}: `{arg_func_name}(...)` returns BoolSort, "
                        f"but is passed to `{caller_name}(...)` parameter {i+1} "
                        f"which expects {expected_sort}"
                    )

    if errors:
        return (
            "Z3 Sort mismatch detected (static check):\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nFunctions returning BoolSort (predicates) cannot be used "
            "as arguments where a custom Sort (entity) is expected. "
            "Use a separate predicate or restructure the formula."
        )
    return None


def _ast_call_name(node: ast.Call) -> str:
    """Extract the function name from an AST Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _ast_sort_name(node: ast.expr) -> str:
    """Extract a sort name from an AST node used in Function() declaration.

    Handles: SortName (ast.Name), BoolSort() (ast.Call), IntSort() (ast.Call).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        name = _ast_call_name(node)
        return name if name else "Unknown"
    return "Unknown"
