from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.code_executor import execute_z3_code, CodeExecutionError
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNKNOWN, VERDICT_REFUTED, VERDICT_UNCERTAIN

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
) -> Phase1Result:
    """
    Phase 1: Formalize & Solve.

    LLM translates NL → Z3-Python code.
    CodeExecutor exec()s the code.
    Z3Solver checks entailment with Unique Name Assumption (Distinct constraints).

    On code errors: feed error back to LLM and retry (up to max_retries).
    """
    messages = _build_messages(premises_nl, conclusion_nl, prompt_engine, dataset)
    last_error: str | None = None
    last_validation_feedback: list[str] = []
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
                    attempt_num=attempt + 1,
                    max_retries=max_retries,
                    prompt_engine=prompt_engine,
                ),
            ]

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
            attempt_record.code_exec_error = last_error
            attempts.append(attempt_record)
            logger.warning("Phase 1 attempt %d/%d: code execution error: %s", attempt + 1, max_retries, e)
            continue

        validation_result = _validate_premises_alignment(premises_nl, ctx["premises"])
        if validation_result:
            last_error, last_validation_feedback = validation_result
            attempt_record.validation_error = last_error
            attempts.append(attempt_record)
            logger.warning("Phase 1 attempt %d/%d: %s", attempt + 1, max_retries, last_error)
            continue
        last_validation_feedback = []

        # Build Distinct() constraints for named entity constants (P0.2)
        bound_var_names: set[str] = ctx.get("bound_var_names", set())
        background_constraints = solver.build_distinct_constraints(
            ctx["namespace"], bound_var_names
        )
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
            logger.warning("Phase 1 attempt %d/%d: solver error: %s", attempt + 1, max_retries, e)
            continue

        attempt_record.verdict = verdict
        attempts.append(attempt_record)
        logger.info("Phase 1 success (attempt %d): verdict=%s", attempt + 1, verdict)
        return Phase1Result(
            verdict=verdict,
            premises=list(ctx["premises"]),       # NL-only, aligned with sentences
            background_constraints=background_constraints,
            bound_var_names=bound_var_names,
            q=ctx["q"],
            model_info=model_info,
            model_info_q=model_info_q,
            namespace=ctx["namespace"],
            raw_code=ctx["raw_code"],
            attempts=attempts,
        )

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


def _build_messages(
    premises_nl: list[str],
    conclusion_nl: str,
    prompt_engine: PromptEngine,
    dataset: str,
) -> list[dict]:
    user_content = prompt_engine.render(
        "phase1_formalize.j2",
        dataset=dataset,
        premises=premises_nl,
        conclusion=conclusion_nl,
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
) -> dict[str, str]:
    user_content = prompt_engine.render(
        "phase1_retry.j2",
        premises=premises_nl,
        conclusion=conclusion_nl,
        previous_code=_normalise_code_for_prompt(raw_code),
        last_error=last_error,
        validation_feedback=validation_feedback,
        attempt_num=attempt_num,
        max_retries=max_retries,
    )
    return {"role": "user", "content": user_content}


def _normalise_code_for_prompt(raw_code: str) -> str:
    code = raw_code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        code = "" if first_newline < 0 else code[first_newline + 1:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _validate_premises_alignment(
    premises_nl: list[str],
    premises: Any,
) -> tuple[str, list[str]] | None:
    expected_count = len(premises_nl)
    try:
        actual_count = len(premises)
    except TypeError:
        return (
            "Validation error: `premises` must be a sized sequence with one top-level "
            "formula per numbered NL premise.",
            [
                f"The original task has {expected_count} numbered premises.",
                "Your code did not define `premises` as a sized sequence that can be aligned to those premises.",
                "Rewrite only the premise formalization so `premises` is a list of formulas in the same order as the numbered premises.",
            ],
        )

    if actual_count == expected_count:
        return None
    return (
        "Validation error: `premises` must contain exactly one top-level formula per "
        f"numbered NL premise (expected {expected_count}, got {actual_count}).",
        [
            f"The original task has {expected_count} numbered premises.",
            f"Your code defined {actual_count} top-level formulas inside `premises`.",
            "Rewrite only the premise formalization so `premises` contains exactly one formula per numbered premise, in the same order.",
            "If one numbered premise is internally conjunctive or disjunctive, keep that structure inside a single formula instead of splitting or omitting it.",
        ],
    )


def _snapshot_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Copy chat messages so attempt traces remain stable after retries mutate the list."""
    return [
        {
            "role": str(m.get("role", "")),
            "content": str(m.get("content", "")),
        }
        for m in messages
    ]
