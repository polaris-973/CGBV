"""
Gap analysis utilities for CGBV Phase 5 unified repair.

Provides predicate connectivity analysis (extracted from Phase 1.5) and
a new `compute_gap_analysis()` that generates structured diagnostic signals
for the repair prompt — identifying ungrounded predicates, disconnected
premises, and missing inter-predicate links.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import z3

from cgbv.core.phase4_check import Mismatch


# Z3 built-in function/operator names to exclude from predicate extraction
Z3_BUILTINS: frozenset[str] = frozenset({
    'and', 'or', 'not', 'implies', '=>', '=', 'distinct', 'ite',
    'true', 'false', 'xor', 'nand', 'nor',
    '+', '-', '*', '/', 'mod', 'div', 'rem',
    '<', '>', '<=', '>=', 'to_int', 'to_real', 'is_int',
})


# ---------------------------------------------------------------------------
# Predicate / formula analysis (moved from phase1_formalize.py)
# ---------------------------------------------------------------------------

def extract_predicate_names(formula: z3.ExprRef) -> set[str]:
    """
    Recursively collect user-defined function/predicate names (arity >= 1) from
    a Z3 formula, including quantifier bodies.  Z3 built-in operator names
    (``and``, ``or``, ``not``, ``=``, etc.) are excluded.

    Only Bool-range functions are included — non-Bool functions (e.g. rent(x) -> Int)
    are value-returning and should not be treated as connectivity links.
    """
    names: set[str] = set()

    def _walk(expr: z3.ExprRef) -> None:
        if z3.is_app(expr):
            decl = expr.decl()
            name = decl.name()
            if (decl.arity() > 0
                    and name not in Z3_BUILTINS
                    and decl.range().kind() == z3.Z3_BOOL_SORT):
                names.add(name)
            for child in expr.children():
                _walk(child)
        elif z3.is_quantifier(expr):
            _walk(expr.body())

    _walk(formula)
    return names


def is_ground_fact(formula: z3.ExprRef) -> bool:
    """
    Return True if *formula* is a ground (non-quantified) formula — i.e. it
    contains no top-level or nested ForAll/Exists quantifier.
    """
    def _has_quantifier(expr: z3.ExprRef) -> bool:
        if z3.is_quantifier(expr):
            return True
        if z3.is_app(expr):
            return any(_has_quantifier(c) for c in expr.children())
        return False

    return not _has_quantifier(formula)


def extract_antecedent_predicates(formula: z3.ExprRef) -> set[str]:
    """
    For a quantified implication ForAll([x...], Implies(A, B)) (possibly nested),
    return the predicate names in the antecedent A.  Returns an empty set for
    all other formula shapes.
    """
    inner = formula
    while z3.is_quantifier(inner):
        inner = inner.body()
    if not z3.is_implies(inner):
        return set()
    antecedent = inner.children()[0]
    return extract_predicate_names(antecedent)


def extract_rule_consequent_predicates(formula: z3.ExprRef) -> set[str]:
    """
    For a quantified implication ForAll([x...], Implies(A, B)), return predicates
    in B (the consequent).  Rules whose consequents are existentially quantified
    are also handled.
    """
    inner = formula
    while z3.is_quantifier(inner):
        inner = inner.body()
    if not z3.is_implies(inner):
        return set()
    return extract_predicate_names(inner.children()[1])


def extract_constant_names(formula: z3.ExprRef) -> set[str]:
    """
    Collect all 0-arity constant names from a Z3 formula (named entities, not
    sort elements like Thing!val!0).  Used for entity-connectivity filtering.
    """
    names: set[str] = set()

    def _walk(expr: z3.ExprRef) -> None:
        if z3.is_app(expr):
            decl = expr.decl()
            if decl.arity() == 0 and decl.range().kind() != z3.Z3_BOOL_SORT:
                name = decl.name()
                if '!' not in name:
                    names.add(name)
            for child in expr.children():
                _walk(child)
        elif z3.is_quantifier(expr):
            _walk(expr.body())

    _walk(formula)
    return names


def find_disconnected_premises(
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
) -> list[int]:
    """
    Return indices of ground-fact premises that are structurally disconnected
    from the proof chain AND whose presence signals a real bridge gap.

    Three-stage filter:

    Stage 1 - Predicate reachability:
      BFS from conclusion predicates through shared predicate sets.  Any ground
      fact whose predicate set is disjoint from the reachable set is a candidate.

    Stage 2 - Grounding gap:
      Check that at least one quantified rule's antecedent predicate is not
      present in any ground fact and is not derivable as the consequent of
      another rule.

    Stage 3 - Entity connectivity:
      A candidate must share at least one entity constant with the connected
      part of the formula.
    """
    pred_sets = [extract_predicate_names(f) for f in premises]
    conclusion_preds = extract_predicate_names(q)

    # BFS: expand reachable predicates from conclusion
    reachable: set[str] = set(conclusion_preds)
    changed = True
    while changed:
        changed = False
        for ps in pred_sets:
            if ps & reachable:
                new_preds = ps - reachable
                if new_preds:
                    reachable.update(new_preds)
                    changed = True

    # Stage 1+: disconnected ground facts
    candidates = [
        i for i, (f, ps) in enumerate(zip(premises, pred_sets))
        if ps and not (ps & reachable) and is_ground_fact(f)
    ]
    if not candidates:
        return []

    # Stage 2: grounding-gap check
    ground_preds: set[str] = set()
    for i, f in enumerate(premises):
        if is_ground_fact(f):
            ground_preds.update(pred_sets[i])

    rule_consequent_preds: set[str] = set()
    for f in premises:
        rule_consequent_preds.update(extract_rule_consequent_predicates(f))

    rule_antecedent_preds: set[str] = set()
    for f in premises:
        rule_antecedent_preds.update(extract_antecedent_predicates(f))

    truly_ungrounded = rule_antecedent_preds - ground_preds - rule_consequent_preds
    if not truly_ungrounded:
        return []

    # Stage 3: entity-connectivity filter
    connected_entities: set[str] = extract_constant_names(q)
    for i, f in enumerate(premises):
        if i not in candidates and is_ground_fact(f) and pred_sets[i] & reachable:
            connected_entities.update(extract_constant_names(f))

    if connected_entities:
        candidates = [
            i for i in candidates
            if extract_constant_names(premises[i]) & connected_entities
        ]

    return candidates


def get_connected_predicates(
    disconnected_idx: int,
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
) -> set[str]:
    """
    Return the set of predicates reachable from the conclusion through all
    premises *except* the disconnected one.
    """
    other_pred_sets = [
        extract_predicate_names(f)
        for i, f in enumerate(premises)
        if i != disconnected_idx
    ]
    conclusion_preds = extract_predicate_names(q)

    reachable: set[str] = set(conclusion_preds)
    changed = True
    while changed:
        changed = False
        for ps in other_pred_sets:
            if ps & reachable:
                new_preds = ps - reachable
                if new_preds:
                    reachable.update(new_preds)
                    changed = True
    return reachable


# ---------------------------------------------------------------------------
# Gap analysis for Phase 5 unified repair
# ---------------------------------------------------------------------------

@dataclass
class GapAnalysisResult:
    """Structured gap analysis for Phase 5 repair prompt injection."""
    ungrounded_predicates: set[str] = field(default_factory=set)
    disconnected_premise_indices: list[int] = field(default_factory=list)
    predicate_graph: dict[str, set[str]] = field(default_factory=dict)
    missing_links: list[tuple[str, str]] = field(default_factory=list)


def _build_predicate_graph(premises: list[z3.ExprRef], q: z3.ExprRef) -> dict[str, set[str]]:
    """
    Build an adjacency graph where predicates that co-occur in the same
    formula (premise or conclusion) are connected.
    """
    graph: dict[str, set[str]] = {}
    all_formulas = list(premises) + [q]
    for f in all_formulas:
        preds = extract_predicate_names(f)
        for p in preds:
            if p not in graph:
                graph[p] = set()
            graph[p].update(preds - {p})
    return graph


def _find_closest_grounded(
    pred: str,
    graph: dict[str, set[str]],
    ground_preds: set[str],
) -> str | None:
    """
    BFS from `pred` through the predicate graph to find the nearest
    predicate that has a ground fact instantiation.
    """
    if pred in ground_preds:
        return pred
    visited = {pred}
    queue = deque([pred])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            if neighbor in ground_preds:
                return neighbor
            queue.append(neighbor)
    return None


def compute_gap_analysis(
    premises: list[z3.ExprRef],
    q: z3.ExprRef,
    mismatches: list[Mismatch] | None = None,
    background_constraints: list[z3.ExprRef] | None = None,
) -> GapAnalysisResult:
    """
    Analyze formula gap structure for Phase 5 repair.

    Goes beyond Phase 1.5's island detection:
    1. Computes predicate co-occurrence graph
    2. Identifies ungrounded rule antecedent predicates
    3. For each ungrounded predicate, finds the closest grounded predicate
       via BFS to suggest missing linking axioms

    background_constraints includes Phase 1.5 bridge axioms and previously
    committed Phase 5 bridges — these are counted as derivable rules so that
    already-bridged gaps are not re-reported.
    """
    # Include background constraints (bridge axioms etc.) in the analysis
    all_formulas = list(premises)
    if background_constraints:
        all_formulas.extend(background_constraints)

    pred_sets = [extract_predicate_names(f) for f in all_formulas]

    # Collect predicates provided by ground facts
    ground_preds: set[str] = set()
    for i, f in enumerate(all_formulas):
        if is_ground_fact(f):
            ground_preds.update(pred_sets[i])

    # Rule antecedent and consequent predicates
    rule_antecedent_preds: set[str] = set()
    rule_consequent_preds: set[str] = set()
    for f in all_formulas:
        rule_antecedent_preds.update(extract_antecedent_predicates(f))
        rule_consequent_preds.update(extract_rule_consequent_predicates(f))

    # Truly ungrounded: rule antecedent predicates not in ground facts or derivable
    ungrounded = rule_antecedent_preds - ground_preds - rule_consequent_preds

    # Disconnected premises
    disconnected = find_disconnected_premises(premises, q)

    # Build predicate graph (includes background constraints for bridge visibility)
    graph = _build_predicate_graph(all_formulas, q)

    # Find missing links: scope to predicates relevant to current mismatches.
    # Only compute for ungrounded preds in the mismatch formula or 1-hop away,
    # avoiding noisy global BFS suggestions for unrelated predicates.
    if mismatches:
        mismatch_preds: set[str] = set()
        n_prem = len(premises)
        for m in mismatches:
            formula = premises[m.sentence_index] if m.sentence_index < n_prem else q
            mismatch_preds.update(extract_predicate_names(formula))

        # 1-hop neighborhood in predicate graph
        one_hop: set[str] = set()
        for p in mismatch_preds:
            one_hop.update(graph.get(p, set()))

        relevant_ungrounded = ungrounded & (mismatch_preds | one_hop)
    else:
        relevant_ungrounded = ungrounded

    # Exclude predicates already explicitly bridged in background_constraints
    # (i.e., already appear as antecedents of committed bridge axioms)
    bg_antecedent_preds: set[str] = set()
    if background_constraints:
        for f in background_constraints:
            bg_antecedent_preds.update(extract_antecedent_predicates(f))

    missing_links: list[tuple[str, str]] = []
    for pred in sorted(relevant_ungrounded - bg_antecedent_preds):
        closest = _find_closest_grounded(pred, graph, ground_preds)
        if closest and closest != pred:
            missing_links.append((closest, pred))

    return GapAnalysisResult(
        ungrounded_predicates=ungrounded,
        disconnected_premise_indices=disconnected,
        predicate_graph=graph,
        missing_links=missing_links,
    )
