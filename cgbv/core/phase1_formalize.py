from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.code_executor import execute_z3_code, CodeExecutionError, build_name_error_hint
from cgbv.solver.cwa_axioms import build_cwa_constraints
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNKNOWN, VERDICT_REFUTED, VERDICT_UNCERTAIN, VERDICT_ENTAILED

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
        if verdict != VERDICT_ENTAILED:
            result = await _run_bridge_check(
                result=result,
                premises_nl=premises_nl,
                conclusion_nl=conclusion_nl,
                llm=llm,
                solver=solver,
                prompt_engine=prompt_engine,
                task_type=task_type,
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
    for attempt in range(2):
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


# ---------------------------------------------------------------------------
# Predicate connectivity analysis (Phase 1.5 helpers)
# ---------------------------------------------------------------------------

# Z3 built-in function/operator names to exclude from predicate extraction
_Z3_BUILTINS: frozenset[str] = frozenset({
    'and', 'or', 'not', 'implies', '=>', '=', 'distinct', 'ite',
    'true', 'false', 'xor', 'nand', 'nor',
    '+', '-', '*', '/', 'mod', 'div', 'rem',
    '<', '>', '<=', '>=', 'to_int', 'to_real', 'is_int',
})


def _extract_predicate_names(formula: z3.ExprRef) -> set[str]:
    """
    Recursively collect user-defined function/predicate names (arity ≥ 1) from
    a Z3 formula, including quantifier bodies.  Z3 built-in operator names
    (``and``, ``or``, ``not``, ``=``, etc.) are excluded.
    """
    names: set[str] = set()

    def _walk(expr: z3.ExprRef) -> None:
        if z3.is_app(expr):
            decl = expr.decl()
            name = decl.name()
            # Only Bool-range functions are logical predicates that participate
            # in the proof chain.  Non-Bool functions (e.g. rent(x) → Int)
            # are value-returning and should not be treated as connectivity
            # links — including them could create false connections or hide
            # real orphan premises.
            if (decl.arity() > 0
                    and name not in _Z3_BUILTINS
                    and decl.range().kind() == z3.Z3_BOOL_SORT):
                names.add(name)
            for child in expr.children():
                _walk(child)
        elif z3.is_quantifier(expr):
            _walk(expr.body())

    _walk(formula)
    return names


def _is_ground_fact(formula: z3.ExprRef) -> bool:
    """
    Return True if *formula* is a ground (non-quantified) formula — i.e. it
    contains no top-level or nested ForAll/Exists quantifier.
    We detect this by walking the formula and checking for quantifier nodes.
    """
    def _has_quantifier(expr: z3.ExprRef) -> bool:
        if z3.is_quantifier(expr):
            return True
        if z3.is_app(expr):
            return any(_has_quantifier(c) for c in expr.children())
        return False

    return not _has_quantifier(formula)


def _extract_antecedent_predicates(formula: z3.ExprRef) -> set[str]:
    """
    For a quantified implication ForAll([x…], Implies(A, B)) (possibly nested),
    return the predicate names in the antecedent A.  Returns an empty set for
    all other formula shapes.
    """
    inner = formula
    while z3.is_quantifier(inner):
        inner = inner.body()
    if not z3.is_implies(inner):
        return set()
    antecedent = inner.children()[0]
    return _extract_predicate_names(antecedent)


def _extract_constant_names(formula: z3.ExprRef) -> set[str]:
    """
    Collect all 0-arity constant names from a Z3 formula (named entities, not
    sort elements like Thing!val!0).  Used for entity-connectivity filtering.
    """
    names: set[str] = set()

    def _walk(expr: z3.ExprRef) -> None:
        if z3.is_app(expr):
            decl = expr.decl()
            if decl.arity() == 0 and decl.range().kind() != z3.Z3_BOOL_SORT:
                name = decl.name()
                # Skip Z3 internal names (containing '!')
                if '!' not in name:
                    names.add(name)
            for child in expr.children():
                _walk(child)
        elif z3.is_quantifier(expr):
            _walk(expr.body())

    _walk(formula)
    return names


def _extract_rule_consequent_predicates(formula: z3.ExprRef) -> set[str]:
    """
    For a quantified implication ForAll([x…], Implies(A, B)), return predicates
    in B (the consequent).  Rules whose consequents are existentially quantified
    are also handled.
    """
    inner = formula
    while z3.is_quantifier(inner):
        inner = inner.body()
    if not z3.is_implies(inner):
        return set()
    return _extract_predicate_names(inner.children()[1])


def _find_disconnected_premises(
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
) -> list[int]:
    """
    Return indices of ground-fact premises that are structurally disconnected
    from the proof chain AND whose presence signals a real bridge gap.

    Three-stage filter:

    Stage 1 — Predicate reachability:
      BFS from conclusion predicates through shared predicate sets.  Any ground
      fact whose predicate set is disjoint from the reachable set is a candidate.
      Universal/existential rules are excluded (they are proof-skeleton, not links).

    Stage 2 — Grounding gap:
      Check that at least one quantified rule's antecedent predicate is not
      present in any ground fact and is not derivable as the consequent of
      another rule.  Without an actual grounding gap, disconnected facts are
      harmless background material.

    Stage 3 — Entity connectivity:
      A background fact like "costs(gre, $205)" is structurally disconnected
      but involves entities (gre, $205) that are unrelated to the proof chain
      entities (tom in sample 560).  We require that a bridge candidate shares
      at least one entity constant with the "connected" part of the formula
      (ground facts that ARE reachable).  This prevents false-positive repair
      attempts on genuinely unrelated background premises.
    """
    pred_sets = [_extract_predicate_names(f) for f in premises]
    conclusion_preds = _extract_predicate_names(q)

    # BFS: expand reachable predicates from conclusion
    reachable: set[str] = set(conclusion_preds)
    changed = True
    while changed:
        changed = False
        for ps in pred_sets:
            if ps & reachable:
                new_preds = ps - reachable
                if new_preds:
                    reachable.update(new_preds)
                    changed = True

    # Stage 1+: disconnected ground facts
    candidates = [
        i for i, (f, ps) in enumerate(zip(premises, pred_sets))
        if ps and not (ps & reachable) and _is_ground_fact(f)
    ]
    if not candidates:
        return []

    # Stage 2: grounding-gap check
    # Collect predicates provided by ground facts
    ground_preds: set[str] = set()
    for i, f in enumerate(premises):
        if _is_ground_fact(f):
            ground_preds.update(pred_sets[i])

    # Predicates derivable as rule consequents (can be "inferred", not needing a ground fact)
    rule_consequent_preds: set[str] = set()
    for f in premises:
        rule_consequent_preds.update(_extract_rule_consequent_predicates(f))

    rule_antecedent_preds: set[str] = set()
    for f in premises:
        rule_antecedent_preds.update(_extract_antecedent_predicates(f))

    # Truly ungrounded: rule antecedent predicates not in ground facts and not derivable
    truly_ungrounded = rule_antecedent_preds - ground_preds - rule_consequent_preds
    if not truly_ungrounded:
        return []

    # Stage 3: entity-connectivity filter
    # Collect entity constants appearing in connected (reachable) ground facts and conclusion
    connected_entities: set[str] = _extract_constant_names(q)
    for i, f in enumerate(premises):
        if i not in candidates and _is_ground_fact(f) and pred_sets[i] & reachable:
            connected_entities.update(_extract_constant_names(f))

    # A candidate is a genuine bridge gap only if it shares at least one entity
    # with the connected portion of the formula, OR if connected_entities is empty
    # (degenerate case — include all candidates)
    if connected_entities:
        candidates = [
            i for i in candidates
            if _extract_constant_names(premises[i]) & connected_entities
        ]

    return candidates


def _get_connected_predicates(
    disconnected_idx: int,
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
) -> set[str]:
    """
    Return the set of predicates reachable from the conclusion through all
    premises *except* the disconnected one.  This is the "connected vocabulary"
    we want the bridge repair to use.
    """
    other_pred_sets = [
        _extract_predicate_names(f)
        for i, f in enumerate(premises)
        if i != disconnected_idx
    ]
    conclusion_preds = _extract_predicate_names(q)

    reachable: set[str] = set(conclusion_preds)
    changed = True
    while changed:
        changed = False
        for ps in other_pred_sets:
            if ps & reachable:
                new_preds = ps - reachable
                if new_preds:
                    reachable.update(new_preds)
                    changed = True
    return reachable


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
