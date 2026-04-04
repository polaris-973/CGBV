from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import cgbv.logging as cgbv_log
from cgbv.core.phase2_witness import Phase2Result, run_phase2
from cgbv.core.phase3_grounded import Phase3Result, run_phase3
from cgbv.core.phase4_check import Mismatch, Phase4Result, run_phase4
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNCERTAIN
from cgbv.solver.model_extractor import format_domain_desc

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
    bound_var_names: set[str] | None = None,
    background_constraints: list | None = None,
    num_witnesses: int = 1,
    grounding_retries: int = 2,
    world_assumption: str = "owa",
) -> MultiWitnessResult:
    """
    Multi-Witness coordinator: Phase 2 + Phase 3 + Phase 4 across K witnesses.

    For Uncertain verdict (P1.1): constructs K witnesses on the ¬q side PLUS exactly
      1 additional witness on the q side, regardless of K. This guarantees q-side
      coverage even when K=1 (the common default), without reducing ¬q coverage.
      Total witnesses = K + 1 for Uncertain with model_info_q available.
    For other verdicts: all K witnesses use the standard ¬q / MaxSAT path.

    Phase 2 (witness construction) is serial (block clause chain per side).
    Phase 3+4 runs in parallel across all witnesses via asyncio.gather.
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
    # Phase 3+4: Parallel execution across all witnesses
    # ----------------------------------------------------------------
    tasks = [
        _run_phase3_and_4(
            witness_index=k,
            p2=witnesses[k],
            sentences=sentences,
            fol_formulas=fol_formulas,
            namespace=namespace,
            solver=solver,
            llm=llm,
            prompt_engine=prompt_engine,
            grounding_retries=grounding_retries,
            world_assumption=world_assumption,
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
    )


async def _run_phase3_and_4(
    witness_index: int,
    p2: Phase2Result,
    sentences: list[str],
    fol_formulas: list,
    namespace: dict,
    solver: Z3Solver,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    grounding_retries: int,
    world_assumption: str = "owa",
) -> WitnessCheckResult:
    """Run Phase 3 + Phase 4 for a single witness."""
    cgbv_log.update_phase("phase3")
    p3 = await run_phase3(
        sentences=sentences,
        domain_desc_str=p2.domain_desc_str,
        domain=p2.domain,
        llm=llm,
        prompt_engine=prompt_engine,
        max_retries=grounding_retries,
        world_assumption=world_assumption,
    )

    cgbv_log.update_phase("phase4")
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
