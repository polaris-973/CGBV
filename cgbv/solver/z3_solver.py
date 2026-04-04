from __future__ import annotations

import logging
from typing import Any

import z3

from cgbv.solver.model_extractor import extract_model_description, format_domain_desc

logger = logging.getLogger(__name__)

VERDICT_ENTAILED = "Entailed"
VERDICT_NOT_ENTAILED = "Not Entailed"
VERDICT_UNKNOWN = "Unknown"
VERDICT_REFUTED = "Refuted"
VERDICT_UNCERTAIN = "Uncertain"


class Z3Solver:
    """Z3-based solver for entailment checking, boundary witness construction, and formula evaluation."""

    def __init__(self, timeout_ms: int = 60_000):
        self.timeout_ms = timeout_ms

    def _make_solver(self) -> z3.Solver:
        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        return s

    # ------------------------------------------------------------------
    # Phase 1: Entailment Check
    # ------------------------------------------------------------------

    def check_entailment(
        self,
        premises: list[z3.ExprRef],
        q: z3.ExprRef,
    ) -> tuple[str, z3.ModelRef | None]:
        """
        Check whether premises ⊨ q.

        Returns:
            (verdict, model_info)
            verdict ∈ {"Entailed", "Not Entailed", "Unknown"}
            model_info: Z3 model if Not Entailed, else None
        """
        s = self._make_solver()
        for f in premises:
            s.add(f)
        s.add(z3.Not(q))

        result = s.check()
        if result == z3.unsat:
            return VERDICT_ENTAILED, None
        elif result == z3.sat:
            return VERDICT_NOT_ENTAILED, s.model()
        else:
            return VERDICT_UNKNOWN, None

    def check_entailment_three_class(
        self,
        premises: list[z3.ExprRef],
        q: z3.ExprRef,
    ) -> tuple[str, z3.ModelRef | None, z3.ModelRef | None]:
        """
        Three-class entailment check.

        Returns:
            (verdict, model_not_q, model_q)
            verdict ∈ {Entailed, Refuted, Uncertain, Unknown}
            model_not_q: P∧¬q countermodel (Not Entailed / Refuted / Uncertain)
            model_q: P∧q model (only for Uncertain, else None)
        """
        verdict1, model1 = self.check_entailment(premises, q)
        if verdict1 != VERDICT_NOT_ENTAILED:
            return verdict1, model1, None
        # Check if premises ⊨ ¬q (i.e., P ∧ q is UNSAT)
        s2 = self._make_solver()
        for f in premises:
            s2.add(f)
        s2.add(q)
        result2 = s2.check()
        if result2 == z3.unsat:
            return VERDICT_REFUTED, model1, None
        elif result2 == z3.sat:
            return VERDICT_UNCERTAIN, model1, s2.model()
        else:
            return VERDICT_UNKNOWN, model1, None

    # ------------------------------------------------------------------
    # P0.2: Unique Name Assumption (Distinct constraints)
    # ------------------------------------------------------------------

    def build_distinct_constraints(
        self,
        namespace: dict[str, Any],
        bound_var_names: set[str] | None = None,
    ) -> list[z3.ExprRef]:
        """
        Build Distinct() constraints for named entity constants grouped by sort.

        For each sort with ≥2 named constants, adds Distinct(*constants) to enforce
        the Unique Name Assumption (UNA): distinct names refer to distinct entities.
        Bound variable names are excluded.

        Returns a list of z3.ExprRef constraints (empty if no sort has ≥2 constants).
        """
        if bound_var_names is None:
            bound_var_names = set()

        sort_to_consts: dict[str, list[z3.ExprRef]] = {}
        for name, obj in namespace.items():
            if name.startswith("_") or name in bound_var_names:
                continue
            if isinstance(obj, z3.ExprRef) and not isinstance(obj, z3.BoolRef):
                sort_key = str(obj.sort())
                if sort_key not in sort_to_consts:
                    sort_to_consts[sort_key] = []
                sort_to_consts[sort_key].append(obj)

        constraints: list[z3.ExprRef] = []
        for consts in sort_to_consts.values():
            if len(consts) >= 2:
                constraints.append(z3.Distinct(*consts))
        return constraints

    # ------------------------------------------------------------------
    # Phase 2: Boundary Witness Construction
    # ------------------------------------------------------------------

    def construct_witness_not_entailed(
        self,
        model_info: z3.ModelRef,
        namespace: dict[str, Any],
        bound_var_names: set[str] | None = None,
    ) -> tuple[dict, z3.ModelRef]:
        """
        Case: verdict = Not Entailed / Refuted / Uncertain (¬q side).
        Directly use the countermodel from Phase 1.

        Returns (domain_dict, z3_model).
        """
        domain = extract_model_description(model_info, namespace, bound_var_names)
        return domain, model_info

    def construct_witness_entailed(
        self,
        premises: list[z3.ExprRef],
        q: z3.ExprRef,
        namespace: dict[str, Any],
        bound_var_names: set[str] | None = None,
        block_clauses: list[z3.ExprRef] | None = None,
    ) -> tuple[dict, z3.ModelRef] | None:
        """
        Case: verdict = Entailed.
        Construct near-countermodel via MaxSAT:
            maximize Σ 1[M ⊨ f_i]  s.t. M ⊨ ¬q

        Returns (domain_dict, z3_model), or None if infeasible.
        """
        opt = z3.Optimize()
        opt.set("timeout", self.timeout_ms)

        opt.add(z3.Not(q))

        if block_clauses:
            for clause in block_clauses:
                opt.add(clause)

        for i, f in enumerate(premises):
            opt.add_soft(f, weight=1, id=f"premise_{i}")

        result = opt.check()
        if result == z3.sat:
            model = opt.model()
            domain = extract_model_description(model, namespace, bound_var_names)
            return domain, model
        elif result == z3.unsat:
            logger.info("Conclusion is a tautology; cannot construct boundary witness")
            return None
        else:
            logger.warning("MaxSAT solver returned unknown")
            return None

    def construct_q_side_witness(
        self,
        premises: list[z3.ExprRef],
        q: z3.ExprRef,
        namespace: dict[str, Any],
        bound_var_names: set[str] | None = None,
        block_clauses: list[z3.ExprRef] | None = None,
    ) -> tuple[dict, z3.ModelRef] | None:
        """
        Construct a witness where P∧q holds (for Uncertain verdict, q-side).

        Used to verify formalization from the "conclusion is true" perspective.
        Returns (domain_dict, z3_model), or None if infeasible.
        """
        s = self._make_solver()
        for f in premises:
            s.add(f)
        s.add(q)
        if block_clauses:
            for clause in block_clauses:
                s.add(clause)
        result = s.check()
        if result == z3.sat:
            model = s.model()
            domain = extract_model_description(model, namespace, bound_var_names)
            return domain, model
        return None

    def construct_boundary_witness(
        self,
        verdict: str,
        model_info: z3.ModelRef | None,
        premises: list[z3.ExprRef],
        q: z3.ExprRef,
        namespace: dict[str, Any],
        bound_var_names: set[str] | None = None,
        block_clauses: list[z3.ExprRef] | None = None,
    ) -> tuple[dict, z3.ModelRef] | None:
        """
        Unified witness construction dispatcher (¬q side).

        Returns (domain_dict, z3_model), or None if construction failed.
        """
        if verdict in (VERDICT_NOT_ENTAILED, VERDICT_REFUTED, VERDICT_UNCERTAIN):
            if block_clauses:
                return self._alternative_countermodel(
                    premises, q, namespace, block_clauses, bound_var_names
                )
            return self.construct_witness_not_entailed(model_info, namespace, bound_var_names)
        elif verdict == VERDICT_ENTAILED:
            return self.construct_witness_entailed(
                premises, q, namespace, bound_var_names, block_clauses
            )
        else:
            return None

    def _alternative_countermodel(
        self,
        premises: list[z3.ExprRef],
        q: z3.ExprRef,
        namespace: dict[str, Any],
        block_clauses: list[z3.ExprRef],
        bound_var_names: set[str] | None = None,
    ) -> tuple[dict, z3.ModelRef] | None:
        """Find a countermodel different from already-seen ones."""
        s = self._make_solver()
        for f in premises:
            s.add(f)
        s.add(z3.Not(q))
        for clause in block_clauses:
            s.add(clause)
        result = s.check()
        if result == z3.sat:
            model = s.model()
            domain = extract_model_description(model, namespace, bound_var_names)
            return domain, model
        return None

    # ------------------------------------------------------------------
    # Phase 4: Formula Evaluation on Model
    # ------------------------------------------------------------------

    def evaluate_formula(self, model: z3.ModelRef, formula: z3.ExprRef) -> bool:
        """
        Evaluate a Z3 FOL formula on a given model.

        Returns True/False.
        """
        val = model.evaluate(formula, model_completion=True)
        return z3.is_true(val)

    def evaluate_grounded_formula(
        self,
        domain: dict,
        grounded_code: str,
    ) -> bool | None:
        """
        Evaluate a grounded propositional formula on a domain truth table.

        Returns True/False, or None on evaluation error.
        """
        truth_table: dict[str, bool] = {}
        for pred_name, interp in domain.get("predicates", {}).items():
            for args, val in interp.items():
                key = f"{pred_name}({', '.join(args)})"
                truth_table[key] = bool(val)

        # value["fname(entity)"] → actual Python value (int/float/str).
        # Enables direct comparisons: value["score(alice)"] < value["score(bob)"].
        value_table: dict[str, Any] = dict(domain.get("function_values", {}))

        safe_globals: dict[str, Any] = {
            "__builtins__": {},
            "truth": truth_table,
            "value": value_table,
            "all": all,
            "any": any,
            "True": True,
            "False": False,
        }

        try:
            result = eval(grounded_code, safe_globals)  # noqa: S307
            return bool(result)
        except Exception as e:
            logger.warning("Failed to evaluate grounded formula %r: %s", grounded_code, e)
            return None

    # ------------------------------------------------------------------
    # Multi-Witness: Model Blocking
    # ------------------------------------------------------------------

    def make_block_clause(
        self,
        domain: dict,
        namespace: dict[str, Any],
    ) -> z3.ExprRef:
        """
        Generate a blocking clause that excludes the given model.
        """
        atoms: list[z3.ExprRef] = []
        for pred_name, interp in domain["predicates"].items():
            pred_ref = namespace.get(pred_name)
            if pred_ref is None:
                continue
            decl_map: dict[str, z3.ExprRef] = {}
            for decl in namespace.values():
                if isinstance(decl, z3.ExprRef) and not isinstance(decl, z3.BoolRef):
                    decl_map[str(decl)] = decl

            for args_tuple, truth_val in interp.items():
                arg_exprs = []
                ok = True
                for entity_name in args_tuple:
                    if entity_name in decl_map:
                        arg_exprs.append(decl_map[entity_name])
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                try:
                    atom = pred_ref(*arg_exprs)
                    atoms.append(atom if truth_val else z3.Not(atom))
                except Exception:
                    continue

        if not atoms:
            return z3.BoolVal(True)
        return z3.Not(z3.And(*atoms))
