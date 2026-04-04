from __future__ import annotations

import logging
from itertools import product
from typing import Any

import z3

logger = logging.getLogger(__name__)


class FiniteModelEvaluator:
    """
    Tri-state evaluator for Z3 FOL formulas on finite models.

    Z3's ``model.evaluate(formula, model_completion=True)`` can return a
    partially-simplified quantified expression (e.g. ``Exists(x, Not(x ==
    Thing!val!3))``) rather than a concrete ``True``/``False`` when the model's
    universe for an uninterpreted sort is not fully elaborated by the solver.
    This evaluator falls back to explicit finite instantiation over the model's
    universe when that happens.

    Return semantics
    ----------------
    * ``True``  — formula is definitely satisfied in the model
    * ``False`` — formula is definitely refuted in the model
    * ``None``  — unverifiable: the model's universe for a required sort is
                  unavailable, or a sub-expression could not be reduced to a
                  concrete boolean.  Callers should treat this as "unverifiable"
                  (not as False) to avoid introducing silent bias.

    Shared by Phase 4 (FOL formula checking) and Phase 5 (local acceptance
    check) so that both phases use identical evaluation semantics.
    """

    def evaluate(
        self,
        model: z3.ModelRef,
        formula: z3.ExprRef,
        namespace: dict | None = None,
    ) -> bool | None:
        """
        Evaluate *formula* on *model*.

        Tries ``model.evaluate`` first; falls back to recursive finite
        instantiation when the result is not a concrete boolean.

        namespace: optional Phase 1 exec namespace (sorts, predicates, constants).
        When provided, named constants in the namespace are used as a third-tier
        universe fallback after model.get_universe() and model.decls() both fail.
        This covers cases where the domain entities exist in the namespace but
        are not reflected in the model's universe or declaration list.
        """
        # Store namespace as instance state for use during recursive evaluation.
        # This avoids threading it through every recursive call signature.
        # Safe for asyncio (no threading interleave within a single coroutine).
        self._eval_namespace: dict | None = namespace
        try:
            val = model.evaluate(formula, model_completion=True)
        except Exception as exc:
            logger.debug("FiniteModelEvaluator.evaluate: model.evaluate raised %s", exc)
            self._eval_namespace = None
            return None

        if z3.is_true(val):
            self._eval_namespace = None
            return True
        if z3.is_false(val):
            self._eval_namespace = None
            return False

        # model.evaluate returned a non-concrete expression — recurse
        result = self._eval_recursive(model, formula)
        self._eval_namespace = None
        return result

    # ------------------------------------------------------------------
    # Internal recursive evaluator
    # ------------------------------------------------------------------

    def _eval_recursive(self, model: z3.ModelRef, expr: z3.ExprRef) -> bool | None:
        """
        Recursively evaluate *expr*, expanding quantifiers over the model's
        finite universe.  Returns ``True``, ``False``, or ``None``.
        """
        # Fast path: re-try model.evaluate for atomic / already-simplified nodes
        try:
            val = model.evaluate(expr, model_completion=True)
        except Exception:
            val = expr

        if z3.is_true(val):
            return True
        if z3.is_false(val):
            return False

        # --- Structural decomposition ---

        if z3.is_quantifier(expr):
            return self._eval_quantifier(model, expr)

        if z3.is_and(expr):
            result: bool | None = True
            for child in expr.children():
                cv = self._eval_recursive(model, child)
                if cv is False:
                    return False          # short-circuit
                if cv is None:
                    result = None         # propagate uncertainty; keep checking
            return result

        if z3.is_or(expr):
            result = False
            for child in expr.children():
                cv = self._eval_recursive(model, child)
                if cv is True:
                    return True           # short-circuit
                if cv is None:
                    result = None
            return result

        if z3.is_not(expr):
            inner = self._eval_recursive(model, expr.children()[0])
            return None if inner is None else not inner

        if z3.is_implies(expr):
            children = expr.children()
            ante = self._eval_recursive(model, children[0])
            if ante is False:
                return True               # False → anything = True
            cons = self._eval_recursive(model, children[1])
            if ante is True:
                return cons               # True → cons = cons
            # ante is None
            if cons is True:
                return True               # None → True = True (vacuously acceptable)
            return None                   # uncertain antecedent, unknown consequent

        # Atomic formula that model.evaluate could not reduce
        logger.debug(
            "FiniteModelEvaluator: irreducible atom %s (kind=%s) → None",
            expr, expr.decl().name() if z3.is_app(expr) else "?",
        )
        return None

    def _eval_quantifier(self, model: z3.ModelRef, expr: z3.ExprRef) -> bool | None:
        """
        Evaluate a quantified formula by instantiating over the model's finite
        universe.

        De Bruijn convention in Z3
        --------------------------
        For ``ForAll([x, y], phi)``:
        * ``expr.var_sort(0)`` = sort of ``x`` (first / outermost bound variable)
        * ``expr.var_sort(1)`` = sort of ``y`` (second / innermost)
        * In the body: ``Var(0)`` refers to *y*, ``Var(1)`` refers to *x*
        * ``z3.substitute_vars(body, a, b)`` replaces ``Var(0)``→``a``,
          ``Var(1)``→``b``

        Therefore, for a combo ``(x_val, y_val)`` drawn from
        ``product(universe_x, universe_y)`` we call
        ``substitute_vars(body, *combo[::-1])``, which maps
        ``Var(0)``→``y_val`` and ``Var(1)``→``x_val`` — the correct order.

        Returns ``None`` if the universe for any bound variable's sort cannot
        be determined (no bias toward True or False).
        """
        is_forall = expr.is_forall()
        num_vars = expr.num_vars()
        body = expr.body()

        # Collect universe per bound variable (in declaration order)
        universes: list[list[z3.ExprRef]] = []
        for i in range(num_vars):
            sort = expr.var_sort(i)
            universe = model.get_universe(sort)
            if not universe:
                # Tier-2 fallback: recover sort members from model.decls()
                universe = self._collect_universe_fallback(model, sort)
            if not universe:
                # Tier-3 fallback: named constants of this sort from the namespace
                universe = self._collect_namespace_fallback(model, sort)
            if not universe:
                logger.debug(
                    "FiniteModelEvaluator._eval_quantifier: empty universe for "
                    "sort %s — returning None (unverifiable)",
                    sort,
                )
                return None          # cannot enumerate → unverifiable, no bias
            universes.append(list(universe))

        has_uncertainty = False
        # Default: ForAll=True (all pass), Exists=False (none found)
        for combo in product(*universes):
            # combo[i] = value for var_sort(i) = i-th declared bound variable
            # substitute_vars maps Var(j) → combo[...]; reverse combo to align
            # De Bruijn indices with declaration order (see docstring).
            substituted = z3.substitute_vars(body, *combo[::-1])
            cv = self._eval_recursive(model, substituted)

            if cv is None:
                has_uncertainty = True
                continue              # can't determine this instance — keep going

            if is_forall and not cv:
                return False          # ForAll: found a counterexample
            if not is_forall and cv:
                return True           # Exists: found a witness

        # Finished iterating without a definitive counter/witness
        if has_uncertainty:
            # ForAll: couldn't verify all instances
            # Exists:  couldn't rule out existence
            return None

        # ForAll: all instances True → True
        # Exists: all instances False → False
        return is_forall

    def _collect_namespace_fallback(
        self, model: z3.ModelRef, sort: z3.SortRef
    ) -> list[z3.ExprRef]:
        """
        Tier-3 universe fallback using ``self._eval_namespace`` (the namespace
        passed to ``evaluate()``).

        Evaluates each named constant in the namespace against the model to
        obtain its concrete Z3 value, then collects unique values whose sort
        matches *sort*.  Covers scenarios where model.get_universe() is empty
        and model.decls() is incomplete, but the Phase 2 domain entities are
        reachable as named constants in the Phase 1 exec namespace.
        """
        ns = getattr(self, "_eval_namespace", None)
        if not ns:
            return []
        exprs: list[z3.ExprRef] = []
        seen: set[str] = set()
        try:
            for name, obj in ns.items():
                if name.startswith("_"):
                    continue
                if not isinstance(obj, z3.ExprRef) or isinstance(obj, z3.BoolRef):
                    continue
                if obj.sort() != sort:
                    continue
                try:
                    val = model.evaluate(obj, model_completion=True)
                except Exception:
                    continue
                s = str(val)
                if s not in seen:
                    seen.add(s)
                    exprs.append(val)
        except Exception as exc:
            logger.debug(
                "FiniteModelEvaluator._collect_namespace_fallback: error for "
                "sort %s: %s", sort, exc,
            )
        return exprs

    def _collect_universe_fallback(
        self, model: z3.ModelRef, sort: z3.SortRef
    ) -> list[z3.ExprRef]:
        """
        Fallback universe collection using model.decls() when
        model.get_universe() returns an empty sequence.

        Z3 sometimes populates model declarations rather than the sort universe
        (observed with uninterpreted sorts in certain model configurations).
        Collects all 0-arity constants in the model whose declared range matches
        *sort*.  Does not require namespace access.
        """
        exprs: list[z3.ExprRef] = []
        seen: set[str] = set()
        try:
            for decl in model.decls():
                if decl.arity() == 0 and decl.range() == sort:
                    expr = decl()
                    s = str(expr)
                    if s not in seen:
                        seen.add(s)
                        exprs.append(expr)
        except Exception as exc:
            logger.debug(
                "FiniteModelEvaluator._collect_universe_fallback: error for "
                "sort %s: %s", sort, exc,
            )
        return exprs
