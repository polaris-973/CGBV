from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.core.phase4_check import Mismatch
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.finite_evaluator import FiniteModelEvaluator
from cgbv.solver.model_extractor import format_domain_desc

logger = logging.getLogger(__name__)

_evaluator = FiniteModelEvaluator()


@dataclass
class RepairAttempt:
    """One LLM repair attempt, including eval/local-validation feedback."""
    attempt_num: int
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    extracted_expression: str = ""
    eval_error: str | None = None
    local_validation_error: str | None = None
    local_validation_truth: bool | None = None
    accepted: bool = False


@dataclass
class RepairEntry:
    """Record of a single formula repair attempt."""
    sentence_index: int
    mismatch_type: str
    original_formula_str: str
    grounded_formula: str
    fol_truth_before: bool
    grounded_truth_expected: bool
    repaired_expr_str: str         # raw LLM output (expression string)
    repaired_formula: object       # z3.ExprRef if successful, None otherwise
    success: bool
    witness_index: int = 0
    witness_side: str = "not_q"
    local_validated: bool = False  # True if repaired formula resolved the mismatch locally
    attempts: list[RepairAttempt] = field(default_factory=list)
    error: str | None = None


@dataclass
class Phase5Result:
    # Repaired premises list (same length as original), z3.ExprRef per entry
    repaired_premises: list                     # list[z3.ExprRef]
    # Repaired conclusion, z3.ExprRef
    repaired_q: object                          # z3.ExprRef
    # Per-mismatch repair records
    repairs: list[RepairEntry] = field(default_factory=list)
    # True if all targeted mismatches were repaired successfully
    all_repaired: bool = False
    # Count of repairs that passed local mismatch verification
    num_local_validated: int = 0
    error: str | None = None


async def run_phase5(
    mismatches: list[Mismatch],
    premises: list,                   # current z3.ExprRef list
    q: object,                        # current z3.ExprRef for conclusion
    namespace: dict[str, Any],        # Phase 1 exec namespace (sorts, predicates, constants)
    raw_code: str,                    # original Phase 1 generated code (for context)
    domain: dict,                     # fallback domain (used when per-witness domain unavailable)
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    models: dict[int, Any] | None = None,          # current-round: witness_index → model
    domains: dict[int, dict] | None = None,        # current-round: witness_index → domain
    mismatch_models: dict[int, Any] | None = None,  # carried override: sentence_index → model
    mismatch_domains: dict[int, dict] | None = None, # carried override: sentence_index → domain
    solver: Any = None,               # Z3Solver for local acceptance check (P0.4)
    world_assumption: str = "owa",
) -> Phase5Result:
    """
    Phase 5: Diagnosis & Targeted Repair.

    For each mismatch, ask the LLM to fix the corresponding FOL formula.
    The repaired expression is eval()d back into the Phase 1 namespace to
    produce a new z3.ExprRef that replaces the original.

    Local acceptance (P0.4): after eval, re-evaluate the repaired formula on the
    correct witness model. Model/domain lookup priority:
      1. mismatch_models/mismatch_domains[sentence_index]  — carried-issue overrides
         (keyed by sentence_index to avoid witness_index renumbering collisions)
      2. models/domains[witness_index]  — current-round witnesses
    """
    if not mismatches:
        return Phase5Result(
            repaired_premises=list(premises),
            repaired_q=q,
            all_repaired=True,
        )

    n = len(premises)
    repaired_premises: list = list(premises)
    repaired_q: object = q

    # Precompute fallback description (used when per-witness domain lookup misses)
    fallback_desc = format_domain_desc(domain)
    repairs: list[RepairEntry] = []

    for mismatch in mismatches:
        idx = mismatch.sentence_index
        is_conclusion = (idx == n)

        # Resolve model/domain for this mismatch.
        # Priority: per-sentence-index carried overrides → per-witness-index current-round.
        # The sentence_index key avoids collisions from witness_index being renumbered
        # 0..N-1 each round (a carried issue's witness_index from a prior round would
        # alias to a different current-round witness under a witness_index keyed map).
        local_domain: dict | None = (
            mismatch_domains.get(idx)
            if mismatch_domains is not None and idx in mismatch_domains
            else (domains.get(mismatch.witness_index) if domains is not None else None)
        )
        if local_domain is None:
            if domains is not None or mismatch_domains is not None:
                logger.warning(
                    "Phase 5 idx=%d: no domain found (witness_index=%d); "
                    "falling back to first-witness domain",
                    idx, mismatch.witness_index,
                )
            witness_desc = fallback_desc
        else:
            witness_desc = format_domain_desc(local_domain)

        local_model = (
            mismatch_models.get(idx)
            if mismatch_models is not None and idx in mismatch_models
            else (models.get(mismatch.witness_index) if models is not None else None)
        )
        if local_model is None and (models is not None or mismatch_models is not None):
            logger.warning(
                "Phase 5 idx=%d: no model found (witness_index=%d); "
                "local acceptance check disabled for this repair",
                idx, mismatch.witness_index,
            )

        repaired_entry = await _repair_one(
            mismatch=mismatch,
            witness_desc=witness_desc,
            raw_code=raw_code,
            namespace=namespace,
            llm=llm,
            prompt_engine=prompt_engine,
            max_retries=max_retries,
            model=local_model,
            solver=solver,
            world_assumption=world_assumption,
        )

        repairs.append(repaired_entry)

        if repaired_entry.success and repaired_entry.repaired_formula is not None:
            if is_conclusion:
                repaired_q = repaired_entry.repaired_formula
            else:
                repaired_premises[idx] = repaired_entry.repaired_formula
            logger.info(
                "Phase 5 idx=%d: repair success (local_validated=%s) (%s → %s)",
                idx, repaired_entry.local_validated,
                str(premises[idx] if not is_conclusion else q)[:60],
                repaired_entry.repaired_expr_str[:60],
            )
        else:
            logger.warning(
                "Phase 5 idx=%d: repair failed: %s", idx, repaired_entry.error
            )

    all_repaired = all(r.success for r in repairs)
    num_local_validated = sum(1 for r in repairs if r.local_validated)
    return Phase5Result(
        repaired_premises=repaired_premises,
        repaired_q=repaired_q,
        repairs=repairs,
        all_repaired=all_repaired,
        num_local_validated=num_local_validated,
    )


async def _repair_one(
    mismatch: Mismatch,
    witness_desc: str,
    raw_code: str,
    namespace: dict[str, Any],
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int,
    model: Any = None,
    solver: Any = None,
    world_assumption: str = "owa",
) -> RepairEntry:
    """Ask LLM to repair one mismatched formula, with retry on eval or local-check failure."""
    messages = _build_messages(mismatch, witness_desc, raw_code, prompt_engine, world_assumption)
    last_error: str | None = None
    raw_output = ""
    attempts: list[RepairAttempt] = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        f"Error: {last_error}\n\n"
                        "Output ONLY a valid Z3-Python expression using the same "
                        "variable names, sort names, predicate names, and constant names "
                        "as the original code. No imports, no assignments, no prose."
                    ),
                },
            ]

        attempt_record = RepairAttempt(
            attempt_num=attempt + 1,
            messages=_snapshot_messages(messages),
        )
        raw_output = await llm.complete_with_retry(messages)
        attempt_record.raw_output = raw_output
        expr_str = _extract_expression(raw_output)
        attempt_record.extracted_expression = expr_str

        formula, error = _eval_expression(expr_str, namespace)
        if error is not None or formula is None:
            last_error = error
            attempt_record.eval_error = error
            attempts.append(attempt_record)
            logger.debug(
                "Phase 5 idx=%d attempt %d: eval error: %s",
                mismatch.sentence_index, attempt + 1, error,
            )
            continue

        # --- Guard 1: Triviality check ---
        # Reject repairs that simplify to the tautology True — these are hollow
        # fixes that always hold and carry no semantic content.
        simplified = z3.simplify(formula)
        if z3.is_true(simplified):
            last_error = (
                "Repaired formula simplifies to True (always satisfied). "
                "This is a hollow fix that carries no semantic content. "
                "Rewrite the formula to accurately capture the NL sentence's meaning "
                "without making it vacuously True."
            )
            attempt_record.local_validation_error = last_error
            attempts.append(attempt_record)
            logger.debug(
                "Phase 5 idx=%d attempt %d: triviality guard triggered",
                mismatch.sentence_index, attempt + 1,
            )
            continue

        # --- Guard 2a: Quantifier-promotion guard ---
        # Two-condition check that prevents bridging-axiom injection while
        # preserving legitimate ground → Exists/ForAll repairs.
        #
        # A quantifier in the repaired formula is only acceptable when:
        #   (a) the original formula already contained a quantifier, OR
        #   (b) the grounded formula is existential/universal (any(...)/all(...))
        #       and the model is asked to lift the repair to match.
        #
        # Even when justified, the quantifier must be the TOP-LEVEL connective.
        # Burying a quantifier inside And/Or/Not (e.g. And(fact, ForAll(...)))
        # still constitutes structural smuggling — it passes local validation but
        # produces semantically over/under-constrained formulas.
        if _contains_quantifier(formula):
            # Use the pre-computed structural label (set from the actual Z3 object
            # in phase4_check._quantifier_layout) rather than string heuristics.
            orig_layout = mismatch.fol_quantifier_layout   # "none" | "top" | "buried"
            orig_has_quantifier = orig_layout in ("top", "buried")
            orig_buried_quantifier = orig_layout == "buried"
            grounded_requires_quantifier = (
                "any(" in mismatch.grounded_formula
                or "all(" in mismatch.grounded_formula
            )
            justified = orig_has_quantifier or grounded_requires_quantifier

            if not justified:
                # Neither the original formula nor the grounded semantics require
                # a quantifier — any quantifier is unjustified structural injection.
                last_error = (
                    "Structure guard: the repair introduced a quantifier (ForAll/Exists) "
                    "but neither the original formula nor the grounded semantics require one. "
                    "Phase 5 must only refine the formula's existing logical structure — "
                    "do NOT embed universal rules or existential witnesses inside the repair. "
                    "Rewrite using only ground predicates and boolean connectives "
                    "(And, Or, Not, direct predicate calls)."
                )
            elif z3.is_quantifier(formula):
                if orig_buried_quantifier:
                    # Preserve the original scope shape: promoting a buried
                    # quantifier to the top level changes quantifier scope and
                    # can alter the sentence's semantics.
                    last_error = (
                        "Structure guard: the original formula used a buried quantifier, "
                        "but the repair promoted it to the top level. "
                        "Preserve the original quantifier scope shape — keep the quantified "
                        "subformula inside its surrounding And/Or/Not structure."
                    )
                else:
                    last_error = None  # justified and top-level → allow
            elif orig_buried_quantifier:
                last_error = None  # original already had buried quantifier; repair inherits the structure
            else:
                # Justified (via grounded semantics or top-level-orig) but the repair
                # buries the quantifier inside And/Or/Not — this is new structural
                # smuggling that was not present in the original.
                last_error = (
                    "Structure guard: a quantifier (ForAll/Exists) is hidden inside a "
                    "compound expression (And/Or/Not). When the repair requires quantification, "
                    "the quantifier must be the top-level connective — not a sub-term. "
                    "Rewrite so that the outermost constructor is ForAll or Exists."
                )

            if last_error is not None:
                attempt_record.local_validation_error = last_error
                attempts.append(attempt_record)
                logger.debug(
                    "Phase 5 idx=%d attempt %d: quantifier-promotion guard triggered",
                    mismatch.sentence_index, attempt + 1,
                )
                continue

        # --- Guard 2b: Shape guard for quantified implications ---
        # If the original formula is ForAll/Exists(Implies(A, B)), the repair
        # must not move B's core predicates into the antecedent or replace B
        # with True, as that produces a vacuous implication.
        shape_error = _check_repair_shape(mismatch.fol_formula_str, formula)
        if shape_error:
            last_error = shape_error
            attempt_record.local_validation_error = last_error
            attempts.append(attempt_record)
            logger.debug(
                "Phase 5 idx=%d attempt %d: shape guard: %s",
                mismatch.sentence_index, attempt + 1, shape_error,
            )
            continue

        # --- Guard 3: Local acceptance check (P0.4) ---
        # The repaired formula must resolve the mismatch on the current witness:
        # FiniteModelEvaluator(model, repaired) must equal grounded_truth.
        # A None result (unverifiable) is treated as a failed check so the LLM
        # must produce a formula that is evaluable on this finite world.
        local_validated = False
        if model is not None and solver is not None:
            new_fol_truth = _evaluator.evaluate(model, formula, namespace=namespace)
            attempt_record.local_validation_truth = new_fol_truth
            if new_fol_truth is None:
                last_error = (
                    "Local validation inconclusive: repaired formula could not be "
                    "evaluated on the witness model (quantifier universe unavailable). "
                    "Use ground facts or a predicate whose universe is fully specified."
                )
                logger.debug(
                    "Phase 5 idx=%d attempt %d: local check returned None",
                    mismatch.sentence_index, attempt + 1,
                )
                attempt_record.local_validation_error = last_error
                attempts.append(attempt_record)
                continue
            elif new_fol_truth == mismatch.grounded_truth:
                local_validated = True
            else:
                last_error = (
                    f"Local validation failed: repaired formula evaluates to "
                    f"{new_fol_truth} on witness, but expected {mismatch.grounded_truth} "
                    f"(mismatch type: {mismatch.mismatch_type}). "
                    "The repair did not resolve the mismatch — please try again."
                )
                logger.debug(
                    "Phase 5 idx=%d attempt %d: local check failed: %s",
                    mismatch.sentence_index, attempt + 1, last_error,
                )
                attempt_record.local_validation_error = last_error
                attempts.append(attempt_record)
                continue

        attempt_record.accepted = True
        attempts.append(attempt_record)
        return RepairEntry(
            sentence_index=mismatch.sentence_index,
            mismatch_type=mismatch.mismatch_type,
            original_formula_str=mismatch.fol_formula_str,
            grounded_formula=mismatch.grounded_formula,
            fol_truth_before=mismatch.fol_truth,
            grounded_truth_expected=mismatch.grounded_truth,
            repaired_expr_str=expr_str,
            repaired_formula=formula,
            success=True,
            witness_index=mismatch.witness_index,
            witness_side=mismatch.witness_side,
            local_validated=local_validated,
            attempts=attempts,
        )

    return RepairEntry(
        sentence_index=mismatch.sentence_index,
        mismatch_type=mismatch.mismatch_type,
        original_formula_str=mismatch.fol_formula_str,
        grounded_formula=mismatch.grounded_formula,
        fol_truth_before=mismatch.fol_truth,
        grounded_truth_expected=mismatch.grounded_truth,
        repaired_expr_str=raw_output,
        repaired_formula=None,
        success=False,
        witness_index=mismatch.witness_index,
        witness_side=mismatch.witness_side,
        local_validated=False,
        attempts=attempts,
        error=f"Repair failed after {max_retries + 1} attempts. Last error: {last_error}",
    )


def _build_messages(
    mismatch: Mismatch,
    witness_desc: str,
    raw_code: str,
    prompt_engine: PromptEngine,
    world_assumption: str = "owa",
) -> list[dict]:
    user_content = prompt_engine.render(
        "phase5_repair.j2",
        sentence_index=mismatch.sentence_index,
        nl_sentence=mismatch.nl_sentence,
        fol_formula=mismatch.fol_formula_str,
        mismatch_type=mismatch.mismatch_type,
        witness_desc=witness_desc,
        fol_truth=mismatch.fol_truth,
        grounded_truth=mismatch.grounded_truth,
        grounded_formula=mismatch.grounded_formula,
        original_code=raw_code,
        world_assumption=world_assumption,
    )
    return [{"role": "user", "content": user_content}]


def _snapshot_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Copy chat messages so repair traces preserve the exact retry context."""
    return [
        {
            "role": str(m.get("role", "")),
            "content": str(m.get("content", "")),
        }
        for m in messages
    ]


def _extract_expression(raw: str) -> str:
    """Strip markdown fences and single backticks if present."""
    raw = re.sub(r'^```(?:python)?\s*\n', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\n```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    if raw.startswith('`') and raw.endswith('`') and len(raw) > 1:
        raw = raw[1:-1].strip()
    return raw


def _extract_pred_names(expr: z3.ExprRef) -> set[str]:
    """Recursively extract all function/predicate names from a Z3 expression."""
    names: set[str] = set()

    def _walk(e: z3.ExprRef) -> None:
        if z3.is_app(e):
            decl = e.decl()
            if decl.arity() > 0:   # exclude 0-arity constants
                names.add(decl.name())
            for child in e.children():
                _walk(child)
        elif z3.is_quantifier(e):
            _walk(e.body())

    _walk(expr)
    return names


def _contains_quantifier(formula: z3.ExprRef) -> bool:
    """
    Return True if `formula` contains a ForAll or Exists quantifier
    *anywhere* in its AST — including buried inside And/Or/Not/etc.
    """
    if z3.is_quantifier(formula):
        return True
    if z3.is_app(formula):
        return any(_contains_quantifier(c) for c in formula.children())
    return False


def _consequent_preds(formula: z3.ExprRef) -> set[str]:
    """
    For a formula of the shape [ForAll/Exists*](Implies(A, B)), return the
    predicate names appearing in B (the consequent).  Returns an empty set for
    any other top-level structure (no shape constraint to enforce).
    """
    inner = formula
    while z3.is_quantifier(inner):
        inner = inner.body()
    if not z3.is_implies(inner):
        return set()
    consequent = inner.children()[1]
    return _extract_pred_names(consequent)


def _check_repair_shape(original_formula_str: str, repaired: z3.ExprRef) -> str | None:
    """
    Two-layer structural validity check for repaired formulas.

    Layer 1 — Triviality guard (belt-and-suspenders, already checked before
    this call but cheap to re-verify):
        Reject if simplify(repaired) is the boolean True.

    Layer 2 — Shape guard for quantified implications:
        If the original formula is ForAll/Exists(Implies(A, B)):
          - The consequent's core predicates must still appear in the repaired
            formula's consequent.
          - They must not have been moved into the antecedent or replaced by True.

    Returns an error string if the shape is violated, None if OK.
    """
    # Layer 1 (redundant but cheap)
    if z3.is_true(z3.simplify(repaired)):
        return (
            "Repaired formula simplifies to True — hollow fix rejected. "
            "The repair must faithfully capture the sentence's semantic content."
        )

    # Layer 2: parse the original formula string to extract consequent predicates.
    # We work from the *string* because the original formula object may not be
    # available at call-site (we pass the string stored in the Mismatch record).
    # A lightweight heuristic: check whether the repaired formula's consequent
    # still contains all the predicates that were in the original's consequent.
    orig_consequent_preds = _consequent_preds_from_str(original_formula_str)
    if not orig_consequent_preds:
        return None   # original is not a quantified implication — no shape constraint

    repair_consequent_preds = _consequent_preds(repaired)

    if not (orig_consequent_preds & repair_consequent_preds):
        return (
            f"Shape guard: the repaired formula's consequent has lost all original "
            f"target predicates {sorted(orig_consequent_preds)}. "
            f"Found in repaired consequent: {sorted(repair_consequent_preds) or 'none'}. "
            "Do not move goal predicates into the antecedent or replace the consequent with True."
        )

    return None


def _consequent_preds_from_str(formula_str: str) -> set[str]:
    """
    Heuristic extraction of consequent predicate names from a Z3 formula *string*
    without re-parsing into a Z3 AST.

    Looks for the pattern ``Implies(..., CONSEQUENT)`` possibly wrapped in
    ``ForAll`` / ``Exists``.  Only used to produce the set of predicate names
    the shape guard should protect; false negatives (empty set) are safe.
    """
    import re as _re
    # Strip leading quantifiers
    s = formula_str.strip()
    s = _re.sub(r'^(ForAll|Exists)\s*\([^\]]+\]\s*,\s*', '', s)

    m = _re.search(r'Implies\s*\(', s)
    if not m:
        return set()

    # Find the matching closing paren for Implies(
    start = m.end()
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1

    # The content inside Implies(...) — split at the top-level comma
    inner = s[start: i - 1]
    # Find top-level comma separating antecedent from consequent
    depth = 0
    split_pos = -1
    for j, ch in enumerate(inner):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            split_pos = j
            break

    if split_pos < 0:
        return set()

    consequent_str = inner[split_pos + 1:].strip()
    # Extract identifiers that look like predicate names (followed by '(')
    pred_names = set(_re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', consequent_str))
    # Filter out Z3 keywords
    _z3_kw = {'And', 'Or', 'Not', 'Implies', 'ForAll', 'Exists', 'If', 'Const',
               'Function', 'BoolSort', 'DeclareSort', 'BoolVal'}
    return pred_names - _z3_kw


def _eval_expression(
    expr_str: str,
    namespace: dict[str, Any],
) -> tuple[z3.ExprRef | None, str | None]:
    """
    Evaluate a Z3-Python expression string inside the Phase 1 namespace.

    Returns (formula, None) on success, or (None, error_str) on failure.
    """
    if not expr_str:
        return None, "Empty expression"

    try:
        eval_ns = dict(namespace)
        import z3 as _z3
        for name in dir(_z3):
            if not name.startswith("_") and name not in eval_ns:
                eval_ns[name] = getattr(_z3, name)

        result = eval(expr_str, eval_ns)  # noqa: S307

        if not isinstance(result, _z3.ExprRef):
            return None, f"Expression did not evaluate to a z3.ExprRef (got {type(result).__name__})"

        return result, None
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Evaluation error: {e}"
