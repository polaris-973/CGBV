"""
CWA (Closed-World Assumption) axiom detection and injection.

Analyses the Z3 formula structure produced by Phase 1 and generates additional
axioms that enforce closed-world semantics.  Two gap detectors:

1. Type subsumption gaps: ensure transitive closure of Implies(A(x), B(x)) chains
2. Comparison predicate chains: inject transitivity and antisymmetry for ordering predicates

XOR (exclusive partition) injection is intentionally NOT implemented here.
Distinguishing inclusive-or from exclusive-or requires reading the NL text, which
the LLM does in Phase 1 via the CWA Rule 10 prompt.  Programmatic XOR injection
would match *any* Or(P, Q) pattern regardless of NL intent, causing systematic
semantic regression on inclusive-or samples.
"""
from __future__ import annotations

import logging
from typing import Any

import z3

logger = logging.getLogger(__name__)

# Names of comparison predicates that model arithmetic ordering
_COMPARISON_PREDS = frozenset({
    "more_than", "less_than", "greater_than",
    "equal_to", "no_more_than", "no_less_than",
    "at_most", "at_least",
})

# Z3 built-in names to ignore
_Z3_BUILTINS = frozenset({
    'and', 'or', 'not', 'implies', '=>', '=', 'distinct', 'ite',
    'true', 'false', 'xor', 'nand', 'nor',
    '+', '-', '*', '/', 'mod', 'div', 'rem',
    '<', '>', '<=', '>=', 'to_int', 'to_real', 'is_int',
})


def build_cwa_constraints(
    namespace: dict[str, Any],
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
    bound_var_names: set[str] | None = None,
) -> list[z3.ExprRef]:
    """
    Detect CWA gaps in the premises and return Z3 axioms to close them.

    Called after Phase 1 formalization when world_assumption == "cwa".
    The returned axioms are appended to background_constraints.

    Only premises are analysed — not q.  Including q would derive axioms
    from the conclusion's structure and inject them as premises, which is
    semantically unsound (circular: the conclusion's predicates would influence
    what is assumed before the proof starts).
    """
    if bound_var_names is None:
        bound_var_names = set()

    # Analyse premises only — not q — to avoid circular reasoning.
    axioms: list[z3.ExprRef] = []
    axioms += _detect_type_subsumption_gaps(namespace, premises, bound_var_names)
    # Comparison chain axioms are general mathematical facts; scanning both
    # premises and q to detect *which predicates are used* is safe because
    # the injected axioms encode universal arithmetic laws, not assumptions
    # derived from the conclusion's truth.
    all_formulas = list(premises) + [q]
    axioms += _detect_comparison_chains(namespace, all_formulas, bound_var_names)

    if axioms:
        logger.info("CWA: injected %d axiom(s): %s",
                     len(axioms), [str(a)[:60] for a in axioms])
    return axioms


# ---------------------------------------------------------------------------
# Detector 1: Type subsumption gaps
# ---------------------------------------------------------------------------

def _detect_type_subsumption_gaps(
    namespace: dict[str, Any],
    formulas: list[z3.ExprRef],
    bound_var_names: set[str],
) -> list[z3.ExprRef]:
    """
    Collect ForAll([x], Implies(P(x), Q(x))) patterns to build a subsumption
    graph.  Compute transitive closure and inject any missing links.

    Example: if we have P→Q and Q→R but not P→R, inject ForAll([x], Implies(P(x), R(x))).
    """
    # Extract all simple Implies(P(x), Q(x)) edges
    edges: list[tuple[str, str, z3.SortRef]] = []

    for formula in formulas:
        edge = _extract_simple_implication(formula)
        if edge is not None:
            edges.append(edge)

    if not edges:
        return []

    # Build adjacency graph
    graph: dict[str, set[str]] = {}
    sort_map: dict[str, z3.SortRef] = {}
    for src, dst, sort in edges:
        if src not in graph:
            graph[src] = set()
        graph[src].add(dst)
        sort_map[src] = sort
        sort_map[dst] = sort

    # Compute transitive closure
    direct_edges = {(src, dst) for src, dsts in graph.items() for dst in dsts}
    closure: dict[str, set[str]] = {k: set(v) for k, v in graph.items()}
    changed = True
    while changed:
        changed = False
        for src in list(closure):
            new_reach = set()
            for mid in list(closure.get(src, set())):
                for dst in closure.get(mid, set()):
                    if dst not in closure.get(src, set()):
                        new_reach.add(dst)
            if new_reach:
                closure.setdefault(src, set()).update(new_reach)
                changed = True

    # Inject missing edges
    axioms: list[z3.ExprRef] = []
    for src, dsts in closure.items():
        for dst in dsts:
            if (src, dst) not in direct_edges:
                sort = sort_map.get(src)
                if sort is None:
                    continue
                src_func = _find_function(namespace, src)
                dst_func = _find_function(namespace, dst)
                if src_func is None or dst_func is None:
                    continue
                x = z3.Const('_cwa_x', sort)
                axiom = z3.ForAll([x], z3.Implies(src_func(x), dst_func(x)))
                axioms.append(axiom)
                logger.debug("CWA subsumption gap: %s → %s: %s",
                             src, dst, str(axiom)[:80])

    return axioms


def _extract_simple_implication(
    formula: z3.ExprRef,
) -> tuple[str, str, z3.SortRef] | None:
    """
    Check if formula is ForAll([x], Implies(P(x), Q(x))) where P and Q are
    unary predicates applied to the same variable.  Returns (P_name, Q_name, sort)
    or None.
    """
    if not z3.is_quantifier(formula):
        return None
    if not formula.is_forall():
        return None
    if formula.num_vars() != 1:
        return None

    body = formula.body()
    if not z3.is_implies(body):
        return None

    antecedent = body.arg(0)
    consequent = body.arg(1)

    # Both must be simple unary predicate applications
    a_name = _simple_unary_pred(antecedent)
    c_name = _simple_unary_pred(consequent)
    if a_name is None or c_name is None:
        return None

    sort = formula.var_sort(0)
    return (a_name, c_name, sort)


def _simple_unary_pred(expr: z3.ExprRef) -> str | None:
    """If expr is P(var) for a single unary predicate, return P's name."""
    if not z3.is_app(expr):
        return None
    decl = expr.decl()
    if decl.arity() != 1:
        return None
    if decl.name() in _Z3_BUILTINS:
        return None
    if decl.range().kind() != z3.Z3_BOOL_SORT:
        return None
    return decl.name()


def _find_function(
    namespace: dict[str, Any], name: str
) -> z3.FuncDeclRef | None:
    """Look up a Z3 Function by name in the namespace."""
    obj = namespace.get(name)
    if obj is not None and isinstance(obj, z3.FuncDeclRef):
        return obj
    # Search for FuncDeclRef objects
    for k, v in namespace.items():
        if isinstance(v, z3.FuncDeclRef) and v.name() == name:
            return v
    return None


# ---------------------------------------------------------------------------
# Detector 3: Comparison predicate chains
# ---------------------------------------------------------------------------

def _detect_comparison_chains(
    namespace: dict[str, Any],
    formulas: list[z3.ExprRef],
    bound_var_names: set[str],
) -> list[z3.ExprRef]:
    """
    Find comparison predicates (more_than, equal_to, no_more_than, etc.) used in
    the formulas and inject transitivity + antisymmetry axioms.
    """
    # Collect comparison predicates actually used in formulas
    used_comparisons: dict[str, z3.FuncDeclRef] = {}

    for formula in formulas:
        _collect_comparison_preds(formula, used_comparisons)

    if not used_comparisons:
        return []

    axioms: list[z3.ExprRef] = []

    # Group by sort (comparison predicates take two arguments of the same sort)
    sort_groups: dict[str, dict[str, z3.FuncDeclRef]] = {}
    for name, func in used_comparisons.items():
        if func.arity() != 2:
            continue
        sort_key = str(func.domain(0))
        sort_groups.setdefault(sort_key, {})[name] = func

    for sort_key, preds in sort_groups.items():
        # Get the sort from one of the predicates
        any_func = next(iter(preds.values()))
        sort = any_func.domain(0)
        x = z3.Const('_cwa_cx', sort)
        y = z3.Const('_cwa_cy', sort)
        z_var = z3.Const('_cwa_cz', sort)

        # Use explicit None checks — Z3 FuncDeclRef overrides __bool__ and would raise
        more = preds.get("more_than") if "more_than" in preds else preds.get("greater_than")
        equal = preds.get("equal_to")
        no_more = preds.get("no_more_than") if "no_more_than" in preds else preds.get("at_most")

        # Use 'is not None' throughout — z3.FuncDeclRef raises on implicit bool()

        # Transitivity of more_than
        if more is not None:
            axiom = z3.ForAll(
                [x, y, z_var],
                z3.Implies(
                    z3.And(more(x, y), more(y, z_var)),
                    more(x, z_var)
                )
            )
            axioms.append(axiom)

        # more_than → not equal_to
        if more is not None and equal is not None:
            axiom = z3.ForAll(
                [x, y],
                z3.Implies(more(x, y), z3.Not(equal(x, y)))
            )
            axioms.append(axiom)

        # more_than → not no_more_than
        if more is not None and no_more is not None:
            axiom = z3.ForAll(
                [x, y],
                z3.Implies(more(x, y), z3.Not(no_more(x, y)))
            )
            axioms.append(axiom)

        # equal_to → no_more_than (both directions)
        if equal is not None and no_more is not None:
            axiom = z3.ForAll(
                [x, y],
                z3.Implies(equal(x, y), z3.And(no_more(x, y), no_more(y, x)))
            )
            axioms.append(axiom)

        # Symmetry of equal_to
        if equal is not None:
            axiom = z3.ForAll(
                [x, y],
                z3.Implies(equal(x, y), equal(y, x))
            )
            axioms.append(axiom)

    return axioms


def _collect_comparison_preds(
    formula: z3.ExprRef,
    result: dict[str, z3.FuncDeclRef],
) -> None:
    """Recursively collect comparison predicate declarations from a formula."""
    if z3.is_app(formula):
        decl = formula.decl()
        name = decl.name()
        if name in _COMPARISON_PREDS and decl.arity() >= 2:
            result[name] = decl
        for i in range(formula.num_args()):
            _collect_comparison_preds(formula.arg(i), result)
    elif z3.is_quantifier(formula):
        _collect_comparison_preds(formula.body(), result)
