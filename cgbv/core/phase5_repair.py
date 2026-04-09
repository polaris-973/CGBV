from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.core.logic_compiler import (
    SymbolTable,
    canonicalize_logic_obj,
    compile_sentence_logic,
    to_compact_dsl_payload_safe,
)
from cgbv.core.phase4_check import Mismatch
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.finite_evaluator import FiniteModelEvaluator
from cgbv.solver.model_extractor import format_domain_desc, format_filtered_domain_desc, format_sparse_domain_desc

logger = logging.getLogger(__name__)

_evaluator = FiniteModelEvaluator()


def _gap_analysis_field(gap_analysis: Any, name: str, default: Any) -> Any:
    """Read a gap-analysis field from either an object or a dict-like payload."""
    if gap_analysis is None:
        return default
    if isinstance(gap_analysis, dict):
        return gap_analysis.get(name, default)
    return getattr(gap_analysis, name, default)


def _gap_analysis_prompt_data(gap_analysis: Any) -> dict[str, Any] | None:
    """Normalize gap analysis into a stable prompt contract."""
    if gap_analysis is None:
        return None

    ungrounded_predicates = list(_gap_analysis_field(gap_analysis, "ungrounded_predicates", []) or [])
    missing_links = list(_gap_analysis_field(gap_analysis, "missing_links", []) or [])
    obligation_hints = list(_gap_analysis_field(gap_analysis, "obligation_hints", []) or [])

    if not (ungrounded_predicates or missing_links or obligation_hints):
        return None

    return {
        "ungrounded_predicates": ungrounded_predicates,
        "missing_links": missing_links,
        "obligation_hints": obligation_hints,
    }


# ---------------------------------------------------------------------------
# f_is() normalization: Phase 3 grounded formulas use truth["f_is(e, v)"]
# keys for booleanized functions, but Phase 1 namespace only has f(e).
# This normalizer rewrites f_is references so the LLM sees Phase-1-compatible
# expressions using direct function-value comparisons.
# ---------------------------------------------------------------------------

def _normalize_fis_references(grounded_formula: str) -> str:
    """Replace f_is references with value-comparison notation.

    The model_extractor booleanizes non-Bool functions as f_is(args, val)
    predicates, but Phase 1 namespace only has the original f(args) function.
    This ensures the repair prompt shows f(entity) == value instead of
    f_is(entity, value), matching the compiler-backed Phase 1 symbol surface.
    """
    def _rewrite_inner(inner: str) -> str:
        m = re.match(r'^(\w+)_is\((.+)\)$', inner)
        if not m:
            return ""
        func_name = m.group(1)
        all_args = [a.strip() for a in m.group(2).split(',')]
        if len(all_args) < 2:
            return ""
        entity_args = ', '.join(all_args[:-1])
        value_arg = all_args[-1]
        return f'(value["{func_name}({entity_args})"] == "{value_arg}")'

    def _rewrite_bracket(match: re.Match) -> str:
        rewritten = _rewrite_inner(match.group(2))
        return rewritten or match.group(0)

    out = re.sub(
        r'truth\[(["\'])(\w+_is\([^)]+\))\1\]',
        _rewrite_bracket,
        grounded_formula,
    )
    # IR debug_render form: truth(f_is(arg, value))
    def _rewrite_call(match: re.Match) -> str:
        rewritten = _rewrite_inner(match.group(1))
        return rewritten or match.group(0)

    out = re.sub(
        r'truth\(\s*(\w+_is\([^)]+\))\s*\)',
        _rewrite_call,
        out,
    )
    return out


_Z3_KEYWORDS = frozenset({
    'And', 'Or', 'Not', 'Implies', 'ForAll', 'Exists', 'If',
    'Const', 'Function', 'BoolSort', 'DeclareSort', 'BoolVal',
})


def _build_function_value_helpers(namespace: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Build (helper_name, original_func, arity_description) for non-Bool functions.

    These helpers describe the normalized f_is(entity, value) → f(entity) == value
    convention so the repair prompt can inform the LLM about it.

    Returns:
        List of (helper_name, original_func_name, arg_description) tuples.
        Example: [("monthly_rent_is", "monthly_rent", "entity, value")]
    """
    helpers = []
    for name, obj in sorted(namespace.items()):
        if (isinstance(obj, z3.FuncDeclRef)
                and obj.range().kind() != z3.Z3_BOOL_SORT
                and obj.arity() > 0):
            n_args = obj.arity()
            entity_args = ", ".join(f"arg{i}" for i in range(n_args))
            helpers.append((f"{name}_is", name, f"{entity_args}, value"))
    return helpers


def _extract_relevant_predicates(
    fol_formula_str: str,
    grounded_formula: str,
) -> set[str]:
    """Extract predicate/function names referenced by the FOL and grounded formulas.

    Used to filter the witness description to only show relevant ground atoms,
    reducing prompt noise for the repair LLM.
    """
    preds: set[str] = set()
    # From FOL formula string: identifiers followed by '('
    fol_ids = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', fol_formula_str))
    preds.update(fol_ids - _Z3_KEYWORDS)

    # From grounded formula:
    # - legacy: truth["pred(...)"], value["func(...)"]
    # - IR debug_render: truth(pred(...)), value(func(...))
    truth_preds = re.findall(r'truth\[["\'](\w+)\(', grounded_formula)
    preds.update(truth_preds)
    truth_preds_ir = re.findall(r'truth\(\s*(\w+)\(', grounded_formula)
    preds.update(truth_preds_ir)
    value_preds = re.findall(r'value\[["\'](\w+)\(', grounded_formula)
    preds.update(value_preds)
    value_preds_ir = re.findall(r'value\(\s*(\w+)\(', grounded_formula)
    preds.update(value_preds_ir)

    # Include _is variants for booleanized functions
    for p in list(preds):
        preds.add(f"{p}_is")

    return preds


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
    repaired_expr_str: str         # raw LLM output or JSON-serialized replacement logic
    repaired_formula: object       # z3.ExprRef if successful, None otherwise
    success: bool
    repaired_logic: Any | None = None
    witness_index: int = 0
    witness_side: str = "not_q"
    local_validated: bool = False  # True if repaired formula resolved the mismatch on its seed witness
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
    # Bridge axioms produced by unified repair (z3.ExprRef list)
    bridge_axioms: list = field(default_factory=list)
    bridge_logic_payloads: list[Any] = field(default_factory=list)
    # Per-attempt trace for unified repair path (observability)
    unified_attempts: list[dict] = field(default_factory=list)
    error: str | None = None


async def run_phase5(
    mismatches: list[Mismatch],
    premises: list,                   # current z3.ExprRef list
    q: object,                        # current z3.ExprRef for conclusion
    namespace: dict[str, Any],        # Phase 1 exec namespace (sorts, predicates, constants)
    dsl_payload: dict[str, Any],
    symbol_table: SymbolTable | None,
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
    batch_repair_threshold: int = 2,  # deprecated, ignored
    gap_analysis: Any = None,         # GapAnalysisResult from gap_analysis.py
    sparse_witness_format: bool = False,  # use sparse positive witness format (True atoms only)
) -> Phase5Result:
    """
    Phase 5: Diagnosis & Targeted Repair (Unified).

    Always presents all mismatches jointly to the LLM for mutually consistent
    repairs. When gap analysis detects ungrounded predicates, the LLM may also
    output bridge axioms via the [BRIDGE] prefix.

    For a single mismatch without gap analysis, falls back to the lighter
    per-mismatch path (_repair_one) to avoid unnecessary overhead.

    Local acceptance (P0.4): after eval, re-evaluate the repaired formula on the
    correct witness model. Model/domain lookup priority:
      1. mismatch_models/mismatch_domains[sentence_index]  — carried-issue overrides
         (keyed by sentence_index to avoid witness_index renumbering collisions)
      2. models/domains[witness_index]  — current-round witnesses
    """
    gap_data = _gap_analysis_prompt_data(gap_analysis)

    if not mismatches:
        has_gap = bool(gap_data and gap_data["missing_links"])
        if not has_gap:
            return Phase5Result(
                repaired_premises=list(premises),
                repaired_q=q,
                all_repaired=True,
            )

    has_gap = bool(gap_data)

    for m in mismatches:
        if m.fol_truth is None or m.grounded_truth is None:
            raise ValueError(
                "Phase 5 received a non-concrete mismatch. "
                "Only concrete truth disagreements are repair targets."
            )

    # Single mismatch without gap signals → lightweight per-mismatch path
    if len(mismatches) == 1 and not has_gap:
        return await _run_single_mismatch(
            mismatch=mismatches[0],
            premises=premises,
            q=q,
            namespace=namespace,
            dsl_payload=dsl_payload,
            symbol_table=symbol_table,
            raw_code=raw_code,
            domain=domain,
            llm=llm,
            prompt_engine=prompt_engine,
            max_retries=max_retries,
            models=models,
            domains=domains,
            mismatch_models=mismatch_models,
            mismatch_domains=mismatch_domains,
            solver=solver,
            world_assumption=world_assumption,
            sparse_witness_format=sparse_witness_format,
        )

    # Unified path: present all mismatches jointly with gap analysis
    return await _run_unified_repair(
        mismatches=mismatches,
        premises=premises,
        q=q,
        namespace=namespace,
        dsl_payload=dsl_payload,
        symbol_table=symbol_table,
        raw_code=raw_code,
        domain=domain,
        llm=llm,
        prompt_engine=prompt_engine,
        max_retries=max_retries,
        models=models,
        domains=domains,
        mismatch_models=mismatch_models,
        mismatch_domains=mismatch_domains,
        solver=solver,
        world_assumption=world_assumption,
        gap_analysis=gap_data,
        sparse_witness_format=sparse_witness_format,
    )


async def _run_single_mismatch(
    mismatch: Mismatch,
    premises: list,
    q: object,
    namespace: dict[str, Any],
    dsl_payload: dict[str, Any],
    symbol_table: SymbolTable | None,
    raw_code: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int,
    models: dict[int, Any] | None,
    domains: dict[int, dict] | None,
    mismatch_models: dict[int, Any] | None,
    mismatch_domains: dict[int, dict] | None,
    solver: Any,
    world_assumption: str,
    sparse_witness_format: bool = False,
) -> Phase5Result:
    """Lightweight single-mismatch path (delegates to _repair_one)."""
    n = len(premises)
    idx = mismatch.sentence_index
    is_conclusion = (idx == n)

    repaired_premises: list = list(premises)
    repaired_q: object = q
    fallback_desc = format_sparse_domain_desc(domain) if sparse_witness_format else format_domain_desc(domain)

    local_domain, local_model = _resolve_domain_model(
        mismatch, domain, models, domains, mismatch_models, mismatch_domains,
    )
    if local_domain is None:
        witness_desc = fallback_desc
    else:
        relevant_preds = _extract_relevant_predicates(
            mismatch.fol_formula_str, mismatch.grounded_formula,
        )
        witness_desc = format_filtered_domain_desc(local_domain, relevant_preds)

    repaired_entry = await _repair_one(
        mismatch=mismatch,
        witness_desc=witness_desc,
        dsl_payload=dsl_payload,
        symbol_table=symbol_table,
        raw_code=raw_code,
        namespace=namespace,
        llm=llm,
        prompt_engine=prompt_engine,
        max_retries=max_retries,
        model=local_model,
        solver=solver,
        world_assumption=world_assumption,
    )

    if repaired_entry.success and repaired_entry.repaired_formula is not None:
        if is_conclusion:
            logger.warning(
                "Phase 5: mismatch on conclusion index (idx=%d) should have been "
                "routed to run_phase1_targeted upstream — skipping conclusion modification.",
                idx,
            )
        else:
            repaired_premises[idx] = repaired_entry.repaired_formula
        logger.info(
            "Phase 5 idx=%d: repair success (local_validated=%s) (%s → %s)",
            idx, repaired_entry.local_validated,
            str(premises[idx] if not is_conclusion else q)[:60],
            repaired_entry.repaired_expr_str[:60],
        )
    else:
        logger.warning("Phase 5 idx=%d: repair failed: %s", idx, repaired_entry.error)

    return Phase5Result(
        repaired_premises=repaired_premises,
        repaired_q=repaired_q,
        repairs=[repaired_entry],
        all_repaired=repaired_entry.success,
        num_local_validated=1 if repaired_entry.local_validated else 0,
    )


def _resolve_domain_model(
    mismatch: Mismatch,
    domain: dict,
    models: dict[int, Any] | None,
    domains: dict[int, dict] | None,
    mismatch_models: dict[int, Any] | None,
    mismatch_domains: dict[int, dict] | None,
) -> tuple[dict | None, Any]:
    """Resolve domain and model for a mismatch with fallback chain."""
    idx = mismatch.sentence_index
    local_domain: dict | None = (
        mismatch_domains.get(idx)
        if mismatch_domains is not None and idx in mismatch_domains
        else (domains.get(mismatch.witness_index) if domains is not None else None)
    )
    if local_domain is None and (domains is not None or mismatch_domains is not None):
        logger.warning(
            "Phase 5 idx=%d: no domain found (witness_index=%d); "
            "falling back to first-witness domain",
            idx, mismatch.witness_index,
        )

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
    return local_domain, local_model


async def _run_unified_repair(
    mismatches: list[Mismatch],
    premises: list,
    q: object,
    namespace: dict[str, Any],
    dsl_payload: dict[str, Any],
    symbol_table: SymbolTable | None,
    raw_code: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    models: dict[int, Any] | None = None,
    domains: dict[int, dict] | None = None,
    mismatch_models: dict[int, Any] | None = None,
    mismatch_domains: dict[int, dict] | None = None,
    solver: Any = None,
    world_assumption: str = "owa",
    gap_analysis: Any = None,
    sparse_witness_format: bool = False,
) -> Phase5Result:
    """Unified repair: present all mismatches + gap analysis jointly to the LLM."""
    n = len(premises)
    repaired_premises: list = list(premises)
    repaired_q: object = q

    # Build per-mismatch witness descriptions and detect if all share the same domain
    mismatch_witness_descs: dict[int, str] = {}
    mismatch_domains_resolved: dict[int, int] = {}  # idx → id(domain) for dedup
    for m in mismatches:
        local_domain, _ = _resolve_domain_model(
            m, domain, None, domains, None, mismatch_domains,
        )
        if local_domain is None:
            local_domain = domain
        relevant_preds = _extract_relevant_predicates(m.fol_formula_str, m.grounded_formula)
        mismatch_witness_descs[m.sentence_index] = format_filtered_domain_desc(local_domain, relevant_preds)
        mismatch_domains_resolved[m.sentence_index] = id(local_domain)

    # If all mismatches share the same domain object, render a single shared witness desc
    unique_domains = set(mismatch_domains_resolved.values())
    shared_witness = len(unique_domains) == 1

    if shared_witness:
        # Shared: build a union-filtered desc from the single domain
        all_relevant: set[str] = set()
        for m in mismatches:
            all_relevant |= _extract_relevant_predicates(m.fol_formula_str, m.grounded_formula)
        first_key = next(iter(mismatch_domains_resolved))
        first_m = next(m for m in mismatches if m.sentence_index == first_key)
        shared_local, _ = _resolve_domain_model(
            first_m, domain, None, domains, None, mismatch_domains,
        )
        if shared_local is None:
            shared_local = domain
        witness_desc = format_filtered_domain_desc(shared_local, all_relevant)
    else:
        witness_desc = ""  # will be empty; per-mismatch descs are inline

    # Normalize grounded formulas for the prompt
    mismatch_data = []
    for m in mismatches:
        entry: dict[str, Any] = {
            "sentence_index": m.sentence_index,
            "nl_sentence": m.nl_sentence,
            "fol_formula": m.fol_formula_str,
            "mismatch_type": m.mismatch_type,
            "fol_truth": m.fol_truth,
            "grounded_truth": m.grounded_truth,
            "grounded_formula": _normalize_fis_references(m.grounded_formula),
        }
        if not shared_witness:
            entry["witness_desc"] = mismatch_witness_descs[m.sentence_index]
        mismatch_data.append(entry)

    gap_data = _gap_analysis_prompt_data(gap_analysis)

    function_value_helpers = _build_function_value_helpers(namespace)
    compact_dsl_json = _dsl_payload_for_prompt(dsl_payload)

    user_content = prompt_engine.render(
        "phase5_unified_repair.j2",
        mismatches=mismatch_data,
        current_dsl=compact_dsl_json,
        original_code=raw_code,
        witness_desc=witness_desc,
        world_assumption=world_assumption,
        gap_analysis=gap_data,
        function_value_helpers=function_value_helpers,
    )
    messages: list[dict] = [{"role": "user", "content": user_content}]

    # Build per-mismatch model/domain for guard evaluation
    mismatch_model_map: dict[int, Any] = {}
    for m in mismatches:
        _, local_model = _resolve_domain_model(
            m, domain, models, domains, mismatch_models, mismatch_domains,
        )
        if local_model is not None:
            mismatch_model_map[m.sentence_index] = local_model

    # Retry loop
    expected_indices = {m.sentence_index for m in mismatches}
    mismatch_by_idx: dict[int, Mismatch] = {m.sentence_index: m for m in mismatches}
    best_parsed: dict[int, tuple[Any, z3.ExprRef, bool]] = {}
    best_bridges: list[tuple[Any, z3.ExprRef]] = []
    seen_bridge_keys: set[str] = set()
    raw_output = ""
    attempt_errors: dict[int, str] = {}  # idx → last rejection reason
    unified_attempts: list[dict] = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Build feedback for failed parses — include rejection reason
            feedback_lines = ["Some repairs failed or were rejected. Fix the following:"]
            for m in mismatches:
                idx = m.sentence_index
                if idx not in best_parsed:
                    reason = attempt_errors.get(idx, "no output parsed for this index")
                    feedback_lines.append(f"  [{idx}] — rejected: {reason}")
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": "\n".join(feedback_lines)},
            ]

        raw_output = await llm.complete_with_retry(messages)
        parsed, bridge_logic_objs = _parse_unified_output(raw_output, expected_indices)
        attempt_errors.clear()

        # Per-attempt trace
        attempt_trace: dict = {
            "attempt_num": attempt + 1,
            "raw_output": raw_output,
            "parsed_indices": list(parsed.keys()),
            "bridges": bridge_logic_objs,
            "per_idx": {},
        }

        # Process per-mismatch repairs
        for idx in parsed:
            if idx in best_parsed:
                attempt_trace["per_idx"][idx] = {"status": "already_accepted"}
                continue  # already accepted from prior attempt
            m = mismatch_by_idx[idx]
            local_model = mismatch_model_map.get(idx)

            passed, formula, local_validated, _err = _apply_guards(
                parsed[idx], m, symbol_table, local_model, solver,
            )
            if passed and formula is not None:
                canonical_logic = canonicalize_logic_obj(parsed[idx])
                best_parsed[idx] = (canonical_logic, formula, local_validated)
                attempt_trace["per_idx"][idx] = {
                    "status": "accepted",
                    "logic": canonical_logic,
                    "local_validated": local_validated,
                }
            else:
                attempt_errors[idx] = _err or "parse/eval error"
                attempt_trace["per_idx"][idx] = {
                    "status": "rejected",
                    "logic": parsed[idx],
                    "error": _err,
                }
                logger.debug("Phase 5 unified idx=%d attempt %d: %s", idx, attempt + 1, _err)

        unified_attempts.append(attempt_trace)

        # Process bridge axioms (accumulate across retries, dedup by string)
        for bridge_logic in bridge_logic_objs:
            try:
                canonical_bridge = canonicalize_logic_obj(bridge_logic)
            except Exception as exc:
                logger.debug("Phase 5 bridge canonicalize error: %s", exc)
                continue
            bridge_key = json.dumps(canonical_bridge, sort_keys=True, ensure_ascii=False)
            if bridge_key in seen_bridge_keys:
                continue
            if symbol_table is None:
                continue
            try:
                formula = compile_sentence_logic(canonical_bridge, symbol_table, dict(symbol_table.variables))
            except Exception as exc:
                logger.debug("Phase 5 bridge compile error: %s", exc)
                continue
            # Bridge must be a quantified formula
            if not _contains_quantifier(formula):
                logger.debug("Phase 5 bridge rejected: not quantified")
                continue
            seen_bridge_keys.add(bridge_key)
            best_bridges.append((canonical_bridge, formula))

        # Early exit if all parsed
        if len(best_parsed) == len(mismatches):
            break

    # Build RepairEntry for each mismatch and commit to premises/q
    repairs: list[RepairEntry] = []
    for m in mismatches:
        idx = m.sentence_index
        is_conclusion = (idx == n)

        if idx in best_parsed:
            logic_obj, formula, local_validated = best_parsed[idx]
            if is_conclusion:
                logger.warning(
                    "Phase 5: mismatch on conclusion index (idx=%d) should have been "
                    "routed to run_phase1_targeted upstream — skipping conclusion modification.",
                    idx,
                )
            else:
                repaired_premises[idx] = formula

            repairs.append(RepairEntry(
                sentence_index=idx,
                mismatch_type=m.mismatch_type,
                original_formula_str=m.fol_formula_str,
                grounded_formula=m.grounded_formula,
                fol_truth_before=m.fol_truth,
                grounded_truth_expected=m.grounded_truth,
                repaired_expr_str=json.dumps(logic_obj, ensure_ascii=False),
                repaired_formula=formula,
                success=True,
                repaired_logic=logic_obj,
                witness_index=m.witness_index,
                witness_side=m.witness_side,
                local_validated=local_validated,
            ))
            logger.info("Phase 5 unified idx=%d: success (local_validated=%s)", idx, local_validated)
        else:
            repairs.append(RepairEntry(
                sentence_index=idx,
                mismatch_type=m.mismatch_type,
                original_formula_str=m.fol_formula_str,
                grounded_formula=m.grounded_formula,
                fol_truth_before=m.fol_truth,
                grounded_truth_expected=m.grounded_truth,
                repaired_expr_str="",
                repaired_formula=None,
                success=False,
                repaired_logic=None,
                witness_index=m.witness_index,
                witness_side=m.witness_side,
                local_validated=False,
                error="Unified repair: failed to produce valid repair expression",
            ))
            logger.warning("Phase 5 unified idx=%d: failed", idx)

    all_repaired = bool(repairs) and all(r.success for r in repairs)
    num_local_validated = sum(1 for r in repairs if r.local_validated)
    return Phase5Result(
        repaired_premises=repaired_premises,
        repaired_q=repaired_q,
        repairs=repairs,
        all_repaired=all_repaired,
        num_local_validated=num_local_validated,
        bridge_axioms=[formula for _, formula in best_bridges],
        bridge_logic_payloads=[logic for logic, _ in best_bridges],
        unified_attempts=unified_attempts,
    )


def _parse_unified_output(
    raw: str,
    expected_indices: set[int],
) -> tuple[dict[int, Any], list[Any]]:
    payload = _extract_json_payload(raw)
    if payload is None:
        return {}, []

    repairs: dict[int, Any] = {}
    for item in payload.get("repairs", []) or []:
        if not isinstance(item, dict):
            continue
        idx = item.get("sentence_index")
        logic = item.get("logic")
        if isinstance(idx, int) and idx in expected_indices and isinstance(logic, (dict, str)):
            repairs[idx] = logic

    bridges: list[Any] = []
    for item in payload.get("bridges", []) or []:
        if isinstance(item, (dict, str)):
            bridges.append(item)
    return repairs, bridges


def _apply_guards(
    logic_obj: Any,
    mismatch: Mismatch,
    symbol_table: SymbolTable | None,
    model: Any = None,
    solver: Any = None,
) -> tuple[bool, z3.ExprRef | None, bool, str | None]:
    """Apply guards to a repair replacement logic object.

    Returns (passed, formula, local_validated, error_msg).

    Only Guard 3 (local acceptance) is applied: the repaired formula must
    evaluate to the expected truth value on the mismatch witness model.
    """
    if symbol_table is None:
        return False, None, False, "No symbol table available for repair compilation."
    try:
        formula = compile_sentence_logic(logic_obj, symbol_table, dict(symbol_table.variables))
    except Exception as exc:
        return False, None, False, f"Repair logic did not compile: {exc}"

    # Guard 3: Local acceptance check (P0.4)
    local_validated = False
    if model is not None and solver is not None:
        new_fol_truth = _evaluator.evaluate(model, formula, namespace=symbol_table.namespace())
        if new_fol_truth is None:
            return False, formula, False, (
                "Local validation inconclusive: repaired formula could not be "
                "evaluated on the witness model."
            )
        elif new_fol_truth == mismatch.grounded_truth:
            local_validated = True
        else:
            return False, formula, False, (
                f"Local validation failed: repaired formula evaluates to "
                f"{new_fol_truth} on witness, but expected {mismatch.grounded_truth}."
            )

    return True, formula, local_validated, None


async def _repair_one(
    mismatch: Mismatch,
    witness_desc: str,
    dsl_payload: dict[str, Any],
    symbol_table: SymbolTable | None,
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
    messages = _build_messages(
        mismatch,
        witness_desc,
        dsl_payload,
        raw_code,
        prompt_engine,
        world_assumption,
        namespace,
    )
    last_error: str | None = None
    raw_output = ""
    attempts: list[RepairAttempt] = []
    seen_logic_keys: set[str] = set()

    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        f"Error: {last_error}\n\n"
                        "Carefully re-read the grounded formula and the boundary witness world above. "
                        "Identify which specific predicate or condition in the logic object "
                        "causes it to evaluate differently from the grounded formula. "
                        "Then output ONLY a corrected JSON repair payload. "
                        "No prose."
                    ),
                },
            ]

        attempt_record = RepairAttempt(
            attempt_num=attempt + 1,
            messages=_snapshot_messages(messages),
        )
        raw_output = await llm.complete_with_retry(messages)
        attempt_record.raw_output = raw_output
        parsed_repairs, _ = _parse_unified_output(raw_output, {mismatch.sentence_index})
        logic_obj = parsed_repairs.get(mismatch.sentence_index)
        attempt_record.extracted_expression = json.dumps(logic_obj, ensure_ascii=False) if logic_obj else ""
        if logic_obj is None:
            last_error = "Repair output did not contain a valid JSON logic replacement."
            attempt_record.eval_error = last_error
            attempts.append(attempt_record)
            continue

        logic_key = json.dumps(logic_obj, sort_keys=True, ensure_ascii=False)
        if logic_key in seen_logic_keys:
            last_error = (
                "You proposed the same repair logic as a previous attempt. "
                "Try a different structural repair."
            )
            attempt_record.eval_error = "duplicate_expression"
            attempts.append(attempt_record)
            continue
        seen_logic_keys.add(logic_key)

        passed, formula, local_validated, guard_error = _apply_guards(
            logic_obj, mismatch, symbol_table, model, solver,
        )

        if passed and formula is not None:
            canonical_logic = canonicalize_logic_obj(logic_obj)
            attempt_record.accepted = True
            attempts.append(attempt_record)
            return RepairEntry(
                sentence_index=mismatch.sentence_index,
                mismatch_type=mismatch.mismatch_type,
                original_formula_str=mismatch.fol_formula_str,
                grounded_formula=mismatch.grounded_formula,
                fol_truth_before=mismatch.fol_truth,
                grounded_truth_expected=mismatch.grounded_truth,
                repaired_expr_str=json.dumps(canonical_logic, ensure_ascii=False),
                repaired_formula=formula,
                success=True,
                repaired_logic=canonical_logic,
                witness_index=mismatch.witness_index,
                witness_side=mismatch.witness_side,
                local_validated=local_validated,
                attempts=attempts,
            )

        last_error = guard_error
        attempt_record.local_validation_error = guard_error
        attempts.append(attempt_record)
        logger.debug(
            "Phase 5 idx=%d attempt %d: guard rejected: %s",
            mismatch.sentence_index, attempt + 1, guard_error,
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
        repaired_logic=None,
        witness_index=mismatch.witness_index,
        witness_side=mismatch.witness_side,
        local_validated=False,
        attempts=attempts,
        error=f"Repair failed after {max_retries + 1} attempts. Last error: {last_error}",
    )


def _build_messages(
    mismatch: Mismatch,
    witness_desc: str,
    dsl_payload: dict[str, Any],
    raw_code: str,
    prompt_engine: PromptEngine,
    world_assumption: str = "owa",
    namespace: dict[str, Any] | None = None,
) -> list[dict]:
    # Normalize f_is() references so the LLM sees Phase-1-namespace-compatible names
    normalized_grounded = _normalize_fis_references(mismatch.grounded_formula)

    # 9.5: add persistence hint for recurring mismatches
    persist_hint = ""
    if getattr(mismatch, "persist_rounds", 0) > 0:
        persist_hint = (
            f"\n\n**Note:** This mismatch has persisted for "
            f"{mismatch.persist_rounds} repair round(s). "
            f"Previous repair attempts failed. Try a fundamentally different "
            f"approach to the formula structure.\n"
        )

    # Use unified template even for single mismatch
    mismatch_data = [{
        "sentence_index": mismatch.sentence_index,
        "nl_sentence": mismatch.nl_sentence,
        "fol_formula": mismatch.fol_formula_str,
        "mismatch_type": mismatch.mismatch_type,
        "fol_truth": mismatch.fol_truth,
        "grounded_truth": mismatch.grounded_truth,
        "grounded_formula": normalized_grounded,
    }]

    function_value_helpers = _build_function_value_helpers(namespace) if namespace else []
    compact_dsl_json = _dsl_payload_for_prompt(dsl_payload)

    user_content = prompt_engine.render(
        "phase5_unified_repair.j2",
        mismatches=mismatch_data,
        current_dsl=compact_dsl_json,
        original_code=raw_code,
        witness_desc=witness_desc,
        world_assumption=world_assumption,
        gap_analysis=None,
        function_value_helpers=function_value_helpers,
    )
    if persist_hint:
        user_content += persist_hint
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


def _dsl_payload_for_prompt(dsl_payload: dict[str, Any]) -> str:
    compact = to_compact_dsl_payload_safe(dsl_payload)
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _extract_expression(raw: str) -> str:
    """Strip markdown fences and single backticks if present."""
    raw = re.sub(r'^```(?:python)?\s*\n', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\n```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    if raw.startswith('`') and raw.endswith('`') and len(raw) > 1:
        raw = raw[1:-1].strip()
    return raw


def _extract_json_payload(raw: str) -> dict[str, Any] | None:
    text = _extract_expression(raw)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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
