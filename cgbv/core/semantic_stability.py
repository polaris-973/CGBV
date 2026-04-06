from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.core.gap_analysis import compute_gap_analysis
from cgbv.core.multi_witness import WitnessCheckResult
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.finite_evaluator import FiniteModelEvaluator

logger = logging.getLogger(__name__)
_evaluator = FiniteModelEvaluator()


@dataclass
class SemanticAuditIssue:
    sentence_index: int
    nl_sentence: str
    current_formula_str: str
    audited_formula_str: str = ""
    differing_witnesses: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class SemanticAuditResult:
    stable: bool = True
    checked_indices: list[int] = field(default_factory=list)
    issues: list[SemanticAuditIssue] = field(default_factory=list)


async def audit_semantic_stability(
    *,
    sentences: list[str],
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
    namespace: dict[str, Any],
    raw_code: str,
    witness_results: list[WitnessCheckResult],
    llm: LLMClient,
    prompt_engine: PromptEngine,
    indices: list[int] | None = None,
) -> SemanticAuditResult:
    """
    Audit a query-relevant formula slice under a frozen symbol table.

    The auditor may either confirm the current formula (`MATCH`) or provide an
    independently re-formalized alternative using the same namespace. A formula
    is flagged unstable only when the alternative is evaluable and diverges on
    at least one existing witness world.
    """
    selected = list(indices) if indices is not None else _default_indices(premises, q)
    if not selected:
        return SemanticAuditResult(stable=True)

    issues: list[SemanticAuditIssue] = []
    checked: list[int] = []
    all_formulas = list(premises) + [q]

    for idx in selected:
        if idx < 0 or idx >= len(all_formulas):
            continue
        checked.append(idx)
        current_formula = all_formulas[idx]
        rendered = prompt_engine.render(
            "phase1_semantic_audit.j2",
            raw_code=raw_code,
            sentence_index=idx,
            sentence_role=("conclusion" if idx == len(premises) else "premise"),
            nl_sentence=sentences[idx],
            current_formula=str(current_formula),
        )
        raw = (await llm.complete_with_retry([{"role": "user", "content": rendered}])).strip()
        candidate = _strip_fences(raw)
        if not candidate or candidate.upper() == "MATCH":
            continue

        audited_formula, error = _eval_formula(candidate, namespace)
        if error or audited_formula is None:
            logger.debug(
                "Semantic audit sentence=%d ignored malformed candidate: %s",
                idx, error or "unknown error",
            )
            continue

        differing_witnesses: list[dict[str, Any]] = []
        for wr in witness_results:
            if wr.phase2.model is None:
                continue
            current_truth = _evaluator.evaluate(wr.phase2.model, current_formula, namespace=namespace)
            audited_truth = _evaluator.evaluate(wr.phase2.model, audited_formula, namespace=namespace)
            if current_truth is None or audited_truth is None or current_truth == audited_truth:
                continue
            differing_witnesses.append({
                "witness_index": wr.witness_index,
                "witness_side": wr.phase2.witness_side,
                "current_truth": current_truth,
                "audited_truth": audited_truth,
            })

        if differing_witnesses:
            issues.append(SemanticAuditIssue(
                sentence_index=idx,
                nl_sentence=sentences[idx],
                current_formula_str=str(current_formula),
                audited_formula_str=str(audited_formula),
                differing_witnesses=differing_witnesses,
            ))

    return SemanticAuditResult(
        stable=(len(issues) == 0),
        checked_indices=checked,
        issues=issues,
    )


def _default_indices(premises: list[z3.ExprRef], q: z3.ExprRef) -> list[int]:
    gap = compute_gap_analysis(premises, q, mismatches=None, background_constraints=None)
    return sorted(set(gap.query_relevant_premise_indices + [len(premises)]))


def _eval_formula(expr_str: str, namespace: dict[str, Any]) -> tuple[z3.ExprRef | None, str | None]:
    if not expr_str:
        return None, "empty expression"
    try:
        import z3 as _z3
        eval_ns = dict(namespace)
        for name in dir(_z3):
            if not name.startswith("_") and name not in eval_ns:
                eval_ns[name] = getattr(_z3, name)
        result = eval(expr_str, eval_ns)  # noqa: S307
        if not isinstance(result, _z3.BoolRef):
            return None, f"expression is not a BoolRef (got {type(result).__name__})"
        return result, None
    except SyntaxError as e:
        return None, f"syntax error: {e}"
    except Exception as e:
        return None, f"evaluation error: {e}"


def _strip_fences(raw: str) -> str:
    raw = re.sub(r'^```(?:python)?\s*\n', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\n```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    if raw.startswith('`') and raw.endswith('`') and len(raw) > 1:
        raw = raw[1:-1].strip()
    return raw
