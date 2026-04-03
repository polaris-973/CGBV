from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.core.phase4_check import Mismatch
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.model_extractor import format_domain_desc

logger = logging.getLogger(__name__)


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
    models: dict[int, Any] | None = None,   # P0.4: per-witness models keyed by witness_index
    domains: dict[int, dict] | None = None, # per-witness domains for accurate prompts
    solver: Any = None,               # Z3Solver for local acceptance check (P0.4)
) -> Phase5Result:
    """
    Phase 5: Diagnosis & Targeted Repair.

    For each mismatch, ask the LLM to fix the corresponding FOL formula.
    The repaired expression is eval()d back into the Phase 1 namespace to
    produce a new z3.ExprRef that replaces the original.

    Local acceptance (P0.4): after eval, re-evaluate the repaired formula on the
    correct witness model (keyed by mismatch.witness_index). If fol_truth still
    differs from grounded_truth the repair is rejected and retried.
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

        # Use the domain and model from the witness that produced this mismatch
        local_domain: dict | None = (
            domains.get(mismatch.witness_index) if domains is not None else None
        )
        if local_domain is None:
            if domains is not None:
                # domains was provided but had no entry for this witness_index —
                # log a warning so wiring bugs are visible rather than silently swallowed
                logger.warning(
                    "Phase 5 idx=%d: no domain found for witness_index=%d "
                    "(domains keys: %s); falling back to first-witness domain",
                    mismatch.sentence_index, mismatch.witness_index,
                    sorted(domains.keys()),
                )
            witness_desc = fallback_desc
        else:
            witness_desc = format_domain_desc(local_domain)

        local_model = models.get(mismatch.witness_index) if models is not None else None
        if local_model is None and models is not None:
            logger.warning(
                "Phase 5 idx=%d: no model found for witness_index=%d "
                "(models keys: %s); local acceptance check disabled for this repair",
                mismatch.sentence_index, mismatch.witness_index,
                sorted(models.keys()),
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
) -> RepairEntry:
    """Ask LLM to repair one mismatched formula, with retry on eval or local-check failure."""
    messages = _build_messages(mismatch, witness_desc, raw_code, prompt_engine)
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

        # --- P0.4 Local acceptance check ---
        # The repaired formula must resolve the mismatch on the current witness:
        # evaluate_formula(model, repaired) must equal grounded_truth.
        local_validated = False
        if model is not None and solver is not None:
            try:
                new_fol_truth = solver.evaluate_formula(model, formula)
                attempt_record.local_validation_truth = new_fol_truth
                if new_fol_truth == mismatch.grounded_truth:
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
            except Exception as e:
                # Formula could not be evaluated on the witness model.
                # Treat as a failed local check — retry so the LLM can produce
                # a formula that is actually evaluable in this world.
                last_error = (
                    f"Local validation error: repaired formula could not be "
                    f"evaluated on the witness model: {e}"
                )
                logger.debug(
                    "Phase 5 idx=%d attempt %d: local validation exception (retry): %s",
                    mismatch.sentence_index, attempt + 1, e,
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
