from __future__ import annotations

from itertools import product
from typing import Any

import z3


def extract_model_description(
    model: z3.ModelRef,
    namespace: dict[str, Any],
    bound_var_names: set[str] | None = None,
) -> dict:
    """
    Extract a structured domain description from a Z3 model.

    Returns:
        {
            "entities": ["olive_garden", "pine_court"],    # user-defined display names
            "sorts": {"Building": ["olive_garden", "pine_court"]},
            "predicate_signatures": {"is_managed": ["Building"]},
            "predicates": {
                "is_managed": {("olive_garden",): True, ("pine_court",): False},
                ...
            },
            "constants": {"olive_garden": "Building!val!0", ...},
            "raw_model_str": str(model),
        }

    Entity names use user-defined constant names rather than Z3 internal names
    (e.g. "olive_garden" rather than "Building!val!0") so that Phase 3 grounded
    formulas can reference them as truth table keys.

    bound_var_names: names of ForAll/Exists quantifier variables to exclude from
    the constants collection (they are not domain entities).
    """
    if bound_var_names is None:
        bound_var_names = set()

    # Collect declared sorts, predicates, and constants from namespace
    sorts: dict[str, z3.SortRef] = {}
    predicates: dict[str, z3.FuncDeclRef] = {}
    constants: dict[str, z3.ExprRef] = {}

    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if name in bound_var_names:
            continue  # skip quantifier variables
        if isinstance(obj, z3.SortRef):
            sorts[name] = obj
        elif isinstance(obj, z3.FuncDeclRef):
            predicates[name] = obj
        elif isinstance(obj, z3.ExprRef) and obj.sort().kind() != z3.Z3_BOOL_SORT:
            constants[name] = obj

    # ----------------------------------------------------------------
    # Step 1: Extract universe per sort as ExprRef objects
    # ----------------------------------------------------------------
    sort_to_exprs: dict[str, list[z3.ExprRef]] = {}
    sort_to_names: dict[str, list[str]] = {}

    for sort_name, sort_ref in sorts.items():
        universe = model.get_universe(sort_ref)
        if universe:
            exprs = list(universe)
        else:
            exprs = _collect_entity_exprs(model, sort_ref, namespace, bound_var_names)
        sort_to_exprs[sort_name] = exprs
        sort_to_names[sort_name] = [str(e) for e in exprs]

    # Collect all entity ExprRefs and their Z3 string names (deduplicated)
    all_entity_exprs: list[z3.ExprRef] = []
    all_entity_names: list[str] = []  # Z3 internal names
    seen_names: set[str] = set()

    for exprs, names in zip(sort_to_exprs.values(), sort_to_names.values()):
        for expr, name in zip(exprs, names):
            if name not in seen_names:
                seen_names.add(name)
                all_entity_exprs.append(expr)
                all_entity_names.append(name)

    # ----------------------------------------------------------------
    # Step 2: Extract constant interpretations (user_name → z3_internal_name)
    # ----------------------------------------------------------------
    const_interp: dict[str, str] = {}
    for const_name, const_ref in constants.items():
        val = model.evaluate(const_ref, model_completion=True)
        const_interp[const_name] = str(val)

    # ----------------------------------------------------------------
    # Step 3: Build name map: Z3 internal name → preferred user-defined name
    # Prefer longer names (single-char names are almost always bound variables).
    # ----------------------------------------------------------------
    z3_to_user: dict[str, str] = {}
    for user_name, z3_name in const_interp.items():
        if z3_name not in z3_to_user:
            z3_to_user[z3_name] = user_name
        elif len(user_name) > len(z3_to_user[z3_name]):
            z3_to_user[z3_name] = user_name

    # Apply the name map to entity names
    display_entities = [z3_to_user.get(n, n) for n in all_entity_names]

    # ----------------------------------------------------------------
    # Step 4: Build sort → display entity names mapping
    # ----------------------------------------------------------------
    sort_display: dict[str, list[str]] = {}
    for sort_name, z3_names in sort_to_names.items():
        sort_display[sort_name] = [z3_to_user.get(n, n) for n in z3_names]

    # ----------------------------------------------------------------
    # Step 5: Build predicate signatures (pred_name → [sort_name per arg])
    # ----------------------------------------------------------------
    sort_str_to_name: dict[str, str] = {str(sr): sn for sn, sr in sorts.items()}
    pred_signatures: dict[str, list[str]] = {}
    for pred_name, pred_ref in predicates.items():
        sig = []
        for i in range(pred_ref.arity()):
            arg_sort = pred_ref.domain(i)
            sig.append(sort_str_to_name.get(str(arg_sort), str(arg_sort)))
        pred_signatures[pred_name] = sig

    # ----------------------------------------------------------------
    # Step 6: Extract predicate interpretations using ExprRef objects directly
    # ----------------------------------------------------------------
    pred_interp: dict[str, dict] = {}
    for pred_name, pred_ref in predicates.items():
        arity = pred_ref.arity()
        interp: dict = {}

        arg_expr_lists: list[list[z3.ExprRef]] = []
        for i in range(arity):
            arg_sort = pred_ref.domain(i)
            matching: list[z3.ExprRef] = []
            for sort_name, sort_ref in sorts.items():
                if sort_ref.kind() == arg_sort.kind() and str(sort_ref) == str(arg_sort):
                    matching = sort_to_exprs[sort_name]
                    break
            if not matching:
                matching = all_entity_exprs
            arg_expr_lists.append(matching)

        for combo_exprs in product(*arg_expr_lists):
            try:
                val = model.evaluate(pred_ref(*combo_exprs), model_completion=True)
                truth = z3.is_true(val)
            except Exception:
                truth = False
            combo_strs = tuple(
                z3_to_user.get(str(e), str(e)) for e in combo_exprs
            )
            interp[combo_strs] = truth

        pred_interp[pred_name] = interp

    return {
        "entities": display_entities,
        "sorts": sort_display,
        "predicate_signatures": pred_signatures,
        "predicates": pred_interp,
        "constants": const_interp,
        "raw_model_str": str(model),
    }


def _collect_entity_exprs(
    model: z3.ModelRef,
    sort: z3.SortRef,
    namespace: dict[str, Any],
    bound_var_names: set[str] | None = None,
) -> list[z3.ExprRef]:
    """
    Fallback: collect entity ExprRefs for a sort when model.get_universe() is empty.
    """
    if bound_var_names is None:
        bound_var_names = set()
    exprs: list[z3.ExprRef] = []
    seen_strs: set[str] = set()

    for decl in model.decls():
        if decl.arity() == 0 and decl.range() == sort:
            expr = decl()
            s = str(expr)
            if s not in seen_strs:
                seen_strs.add(s)
                exprs.append(expr)

    for name, obj in namespace.items():
        if name in bound_var_names:
            continue
        if (
            isinstance(obj, z3.ExprRef)
            and not isinstance(obj, z3.BoolRef)
            and obj.sort() == sort
        ):
            evaluated = model.evaluate(obj, model_completion=True)
            s = str(evaluated)
            if s not in seen_strs:
                seen_strs.add(s)
                exprs.append(evaluated)

    if not exprs:
        exprs.append(z3.Const("e0", sort))
    return exprs


def format_domain_desc(domain: dict) -> str:
    """
    Format a domain description dict into a human-readable string for LLM prompts.

    Produces a rich format with:
      - Entities grouped by sort (so LLM uses sort-specific iteration lists)
      - Predicate signatures (arg sort annotations)
      - Canonical entity IDs (exact keys to use in truth[...] expressions)
      - Ground atom truth table
    """
    lines = []

    # Sort-grouped entity lists
    sorts: dict[str, list[str]] = domain.get("sorts", {})
    if sorts:
        lines.append("Entities by sort:")
        for sort_name, entities in sorts.items():
            lines.append(f"  {sort_name}: {entities}")
    else:
        lines.append(f"Domain entities: {domain['entities']}")

    lines.append("")

    # Predicate signatures
    pred_sigs: dict[str, list[str]] = domain.get("predicate_signatures", {})
    if pred_sigs:
        lines.append("Predicate signatures (use these names exactly):")
        for pred_name, arg_sorts in pred_sigs.items():
            if arg_sorts:
                sig_str = " × ".join(arg_sorts) + " → bool"
            else:
                sig_str = "→ bool"
            lines.append(f"  {pred_name}({', '.join(arg_sorts)}) → bool")

    lines.append("")

    # Canonical entity IDs
    all_entities = domain.get("entities", [])
    if all_entities:
        lines.append(f"Canonical entity IDs (use these EXACTLY as truth[...] keys):")
        lines.append(f"  {', '.join(all_entities)}")

    lines.append("")

    # Ground atoms
    lines.append("Ground atoms:")
    for pred_name, interp in domain["predicates"].items():
        for args, val in interp.items():
            args_str = ", ".join(args)
            lines.append(f"  {pred_name}({args_str}) = {val}")

    return "\n".join(lines)
