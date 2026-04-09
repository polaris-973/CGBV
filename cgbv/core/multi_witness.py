from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import cgbv.logging as cgbv_log
from cgbv.core.phase2_witness import Phase2Result, run_phase2
from cgbv.core.phase3_grounded import (
    GroundedFormula, GroundingTemplate, Phase3Result,
    generate_templates, generate_templates_partial,
)
from cgbv.core.phase4_check import Mismatch, Phase4Result, run_phase4
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.model_extractor import format_domain_schema
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNCERTAIN

logger = logging.getLogger(__name__)


@dataclass
class WitnessCheckResult:
    """Result of Phase 3+4 for a single witness."""
    witness_index: int
    phase2: Phase2Result
    phase3: Phase3Result
    phase4: Phase4Result


@dataclass
class MultiWitnessResult:
    """Aggregated result across K witnesses."""
    witness_results: list[WitnessCheckResult] = field(default_factory=list)
    # Union of mismatches across all witnesses (deduplicated by sentence_index)
    mismatches: list[Mismatch] = field(default_factory=list)
    # True if NO mismatch found in any witness AND all sentences verified
    all_passed: bool = True
    # Number of witnesses successfully constructed
    num_witnesses_constructed: int = 0
    # Templates used for this round (for pipeline template cache)
    templates: list[GroundingTemplate] = field(default_factory=list)


async def run_multi_witness(
    verdict: str,
    model_info: Any,                    # z3.ModelRef or None (¬q side from Phase 1)
    model_info_q: Any,                  # z3.ModelRef or None (q side, Uncertain only, P1.1)
    premises: list,                     # z3.ExprRef list (NL-only, aligned with sentences)
    q: object,                          # z3.ExprRef
    namespace: dict[str, Any],          # Phase 1 exec namespace
    sentences: list[str],               # [P1, ..., Pn, C] NL sentences
    solver: Z3Solver,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    dsl_payload: dict[str, Any] | None = None,
    symbol_table: Any | None = None,
    bound_var_names: set[str] | None = None,
    background_constraints: list | None = None,
    num_witnesses: int = 1,
    grounding_retries: int = 2,
    world_assumption: str = "owa",
    batch_grounding_size: int = 0,      # Phase 3 batch size (0 = all sentences in one call)
    prev_templates: list[GroundingTemplate] | None = None,
    regenerate_indices: set[int] | None = None,  # None or empty → reuse all prev_templates
) -> MultiWitnessResult:
    """
    Multi-Witness coordinator: Phase 2 + Phase 3 + Phase 4 across K witnesses.

    Template-Once design (Proposal-v2 improvement):
      Phase 3 generates formula templates ONCE from the domain schema (no truth values),
      then instantiates them on each witness via cheap Python eval — no per-witness LLM calls.
      This eliminates cross-witness inconsistency (sample 547) and truth-value contamination,
      and reduces Phase 3 LLM calls from O(K×N) to O(N/batch_size).

    For Uncertain verdict (P1.1): constructs K witnesses on the ¬q side PLUS exactly
      1 additional witness on the q side, regardless of K.
    For other verdicts: all K witnesses use the standard ¬q / MaxSAT path.

    Phase 2 (witness construction) is serial (block clause chain per side).
    Phase 3 template generation happens ONCE after all witnesses are constructed.
    Phase 4 runs in parallel across all witnesses via asyncio.gather.
    """
    fol_formulas = list(premises) + [q]

    # Determine witness counts (P1.1)
    if verdict == VERDICT_UNCERTAIN and model_info_q is not None:
        not_q_count = num_witnesses   # full K witnesses on ¬q side
        q_count = 1                   # always 1 q-side witness for Uncertain
    else:
        not_q_count = num_witnesses
        q_count = 0

    # ----------------------------------------------------------------
    # Phase 2: Construct witnesses serially (block clause chain per side)
    # ----------------------------------------------------------------
    witnesses: list[Phase2Result] = []
    # ¬q side
    not_q_block_clauses: list = []
    for k in range(not_q_count):
        p2 = run_phase2(
            verdict=verdict,
            model_info=model_info,
            premises=premises,
            q=q,
            namespace=namespace,
            solver=solver,
            block_clauses=not_q_block_clauses if not_q_block_clauses else None,
            background_constraints=background_constraints,
            bound_var_names=bound_var_names,
            use_q_side=False,
        )

        if p2.error or p2.domain is None:
            logger.warning(
                "Multi-Witness: failed to construct ¬q witness %d/%d: %s",
                k + 1, not_q_count, p2.error,
            )
            break

        witnesses.append(p2)
        logger.debug("Multi-Witness: constructed ¬q witness %d/%d", k + 1, not_q_count)

        if k < not_q_count - 1:
            block_clause = solver.make_block_clause(p2.domain, namespace)
            not_q_block_clauses.append(block_clause)

    # q side (Uncertain only, P1.1)
    if q_count > 0:
        q_block_clauses: list = []
        for k in range(q_count):
            # First q witness uses model_info_q directly; subsequent ones need a new SAT call
            q_model = model_info_q if k == 0 else None
            p2_q = run_phase2(
                verdict=verdict,
                model_info=q_model,
                premises=premises,
                q=q,
                namespace=namespace,
                solver=solver,
                block_clauses=q_block_clauses if q_block_clauses else None,
                background_constraints=background_constraints,
                bound_var_names=bound_var_names,
                use_q_side=True,
            )

            if p2_q.error or p2_q.domain is None:
                logger.warning(
                    "Multi-Witness: failed to construct q witness %d/%d: %s",
                    k + 1, q_count, p2_q.error,
                )
                break

            witnesses.append(p2_q)
            logger.debug("Multi-Witness: constructed q witness %d/%d", k + 1, q_count)

            if k < q_count - 1:
                block_clause = solver.make_block_clause(p2_q.domain, namespace)
                q_block_clauses.append(block_clause)

    if not witnesses:
        return MultiWitnessResult(
            all_passed=False,
            mismatches=[],
            num_witnesses_constructed=0,
        )

    # ----------------------------------------------------------------
    # Phase 3 Step 1: Template Generation (once, using first witness schema)
    # Templates are generated from domain schema WITHOUT truth values,
    # ensuring all witnesses use the same logical structure (Template-Once design).
    #
    # Template cache: if prev_templates is provided, only regenerate the
    # indices in regenerate_indices (e.g. sentences whose FOL was repaired).
    # This eliminates stochastic regression on already-correct templates.
    # ----------------------------------------------------------------
    cgbv_log.update_phase("phase3")
    first_witness = witnesses[0]
    domain_schema_str = format_domain_schema(first_witness.domain)

    if prev_templates is not None:
        # Schema compatibility check: same entity set and predicate signatures
        prev_schema = getattr(prev_templates[0], '_source_schema', None) if prev_templates else None
        schema_compatible = (prev_schema == domain_schema_str) if prev_schema else True

        if schema_compatible and len(prev_templates) == len(sentences):
            if regenerate_indices:
                logger.info(
                    "Multi-Witness: template cache hit — regenerating %d/%d indices: %s",
                    len(regenerate_indices), len(sentences), sorted(regenerate_indices),
                )
                partial = await generate_templates_partial(
                    indices=regenerate_indices,
                    sentences=sentences,
                    domain_schema_str=domain_schema_str,
                    domain=first_witness.domain,
                    llm=llm,
                    prompt_engine=prompt_engine,
                    max_retries=grounding_retries,
                    world_assumption=world_assumption,
                    solver=solver,
                    dsl_payload=dsl_payload,
                    symbol_table=symbol_table,
                )
                # Merge: replace only regenerated indices, keep rest from cache
                partial_by_idx = {t.sentence_index: t for t in partial}
                merged_templates = [
                    partial_by_idx.get(t.sentence_index, t)
                    for t in prev_templates
                ]
                templates = merged_templates
            else:
                logger.info(
                    "Multi-Witness: template cache hit — no indices to regenerate, reusing all %d templates",
                    len(prev_templates),
                )
                templates = list(prev_templates)
        else:
            logger.info(
                "Multi-Witness: template cache invalidated — schema mismatch or length change "
                "(prev=%d, curr=%d); regenerating all",
                len(prev_templates), len(sentences),
            )
            templates_result = await generate_templates(
                sentences=sentences,
                domain_schema_str=domain_schema_str,
                domain=first_witness.domain,
                llm=llm,
                prompt_engine=prompt_engine,
                max_retries=grounding_retries,
                world_assumption=world_assumption,
                solver=solver,
                dsl_payload=dsl_payload,
                symbol_table=symbol_table,
                batch_size=batch_grounding_size,
            )
            templates = templates_result.templates
    else:
        # First round or no cache: generate all templates
        templates_result = await generate_templates(
            sentences=sentences,
            domain_schema_str=domain_schema_str,
            domain=first_witness.domain,
            llm=llm,
            prompt_engine=prompt_engine,
            max_retries=grounding_retries,
            world_assumption=world_assumption,
            solver=solver,
            dsl_payload=dsl_payload,
            symbol_table=symbol_table,
            batch_size=batch_grounding_size,
        )
        templates = templates_result.templates

    # Stamp source schema on templates for future cache compatibility checks
    for t in templates:
        t._source_schema = domain_schema_str  # type: ignore[attr-defined]

    logger.debug(
        "Multi-Witness: %d templates ready (batch_size=%d, %d witnesses)",
        len(templates), batch_grounding_size, len(witnesses),
    )

    # ----------------------------------------------------------------
    # Phase 3 Step 2 + Phase 4: Parallel eval + check across all witnesses
    # eval() is cheap Python — no LLM calls per witness.
    # ----------------------------------------------------------------
    tasks = [
        _run_eval_and_phase4(
            witness_index=k,
            p2=witnesses[k],
            templates=templates,
            sentences=sentences,
            fol_formulas=fol_formulas,
            namespace=namespace,
            solver=solver,
        )
        for k in range(len(witnesses))
    ]
    witness_results: list[WitnessCheckResult] = await asyncio.gather(*tasks)

    # ----------------------------------------------------------------
    # Aggregate: union of mismatches (deduplicate by sentence_index,
    # keeping the first occurrence per sentence)
    # ----------------------------------------------------------------
    seen_indices: set[int] = set()
    all_mismatches: list[Mismatch] = []

    for wr in witness_results:
        for m in wr.phase4.mismatches:
            if m.sentence_index not in seen_indices:
                seen_indices.add(m.sentence_index)
                all_mismatches.append(m)

    return MultiWitnessResult(
        witness_results=list(witness_results),
        mismatches=all_mismatches,
        all_passed=(len(all_mismatches) == 0 and all(wr.phase4.all_passed for wr in witness_results)),
        num_witnesses_constructed=len(witnesses),
        templates=templates,
    )


async def _run_eval_and_phase4(
    witness_index: int,
    p2: Phase2Result,
    templates: list[GroundingTemplate],
    sentences: list[str],
    fol_formulas: list,
    namespace: dict,
    solver: Z3Solver,
) -> WitnessCheckResult:
    """Phase 3 Step 2 + Phase 4 for a single witness (no LLM calls).

    Converts witness-independent GroundingTemplates to GroundedFormula objects,
    then runs Phase 4 which evaluates each template IR on this witness's domain.
    """
    cgbv_log.update_phase("phase4")

    # Convert templates to GroundedFormula for Phase 4 compatibility.
    # formula_code stays as debug_render for logs/results; execution uses template_ir.
    grounded_formulas = [
        GroundedFormula(
            sentence_index=tmpl.sentence_index,
            nl_sentence=tmpl.nl_sentence,
            formula_code=tmpl.debug_render,
            template_ir=tmpl.template_ir,
            failed=tmpl.failed,
            attempts=tmpl.attempts,
            error=tmpl.error,
        )
        for tmpl in templates
    ]
    p3 = Phase3Result(grounded=grounded_formulas)

    p4 = run_phase4(
        sentences=sentences,
        fol_formulas=fol_formulas,
        model=p2.model,
        domain=p2.domain,
        grounded_formulas=p3.grounded,
        solver=solver,
        namespace=namespace,
        witness_index=witness_index,
        witness_side=p2.witness_side,
    )

    return WitnessCheckResult(
        witness_index=witness_index,
        phase2=p2,
        phase3=p3,
        phase4=p4,
    )
