from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import z3

from cgbv.solver.z3_solver import Z3Solver, VERDICT_NOT_ENTAILED, VERDICT_ENTAILED
from cgbv.solver.model_extractor import format_domain_desc

logger = logging.getLogger(__name__)


@dataclass
class Phase2Result:
    domain: dict | None          # structured domain description (truth table)
    domain_desc_str: str         # formatted string for Phase 3 LLM prompts
    model: object | None         # z3.ModelRef — needed by Phase 4 to evaluate FOL formulas
    witness_side: str = "not_q"  # "not_q" or "q"
    error: str | None = None     # set if witness construction failed


def run_phase2(
    verdict: str,
    model_info: Any,              # z3.ModelRef or None (from Phase 1 / q-side)
    premises: list,               # z3.ExprRef list (NL-only)
    q: object,                    # z3.ExprRef
    namespace: dict[str, Any],
    solver: Z3Solver,
    block_clauses: list | None = None,
    background_constraints: list | None = None,  # Distinct() etc. (P0.2)
    bound_var_names: set[str] | None = None,     # quantifier variable names (P0.1)
    use_q_side: bool = False,                    # construct q-side witness (P1.1 Uncertain)
) -> Phase2Result:
    """
    Phase 2: Boundary Witness Construction.

    - Not Entailed / Refuted / Uncertain (¬q): use the countermodel directly.
    - Entailed: MaxSAT near-countermodel (Z3 Optimize).
    - Uncertain (q-side): use the P∧q model directly.
    - background_constraints: system constraints (Distinct) included in solver calls but
      NOT in the NL-premises list used for Phase 3+4.
    - bound_var_names: excluded from entity extraction in model_extractor.
    """
    # Solver sees NL premises + background constraints
    solver_premises = list(premises) + (background_constraints or [])

    if use_q_side:
        # P1.1: q-side witness for Uncertain
        if model_info is not None:
            # Direct use of the P∧q model from Phase 1
            result = solver.construct_witness_not_entailed(model_info, namespace, bound_var_names)
        else:
            # Need to find a P∧q model (for additional q-side witnesses beyond the first)
            result = solver.construct_q_side_witness(
                solver_premises, q, namespace, bound_var_names, block_clauses
            )
    else:
        result = solver.construct_boundary_witness(
            verdict=verdict,
            model_info=model_info,
            premises=solver_premises,
            q=q,
            namespace=namespace,
            bound_var_names=bound_var_names,
            block_clauses=block_clauses,
        )

    if result is None:
        return Phase2Result(
            domain=None,
            domain_desc_str="",
            model=None,
            witness_side="q" if use_q_side else "not_q",
            error="Could not construct boundary witness",
        )

    domain, z3_model = result
    desc_str = format_domain_desc(domain)
    logger.debug("Phase 2 domain:\n%s", desc_str)
    return Phase2Result(
        domain=domain,
        domain_desc_str=desc_str,
        model=z3_model,
        witness_side="q" if use_q_side else "not_q",
    )
