from __future__ import annotations

import logging
from itertools import product
from typing import Any

import z3

logger = logging.getLogger(__name__)


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
    # Step 5: Separate Bool predicates from non-Bool functions;
    #         build predicate signatures.
    # ----------------------------------------------------------------
    sort_str_to_name: dict[str, str] = {str(sr): sn for sn, sr in sorts.items()}

    bool_predicates: dict[str, z3.FuncDeclRef] = {}
    nonbool_functions: dict[str, z3.FuncDeclRef] = {}
    for name, pred_ref in predicates.items():
        if pred_ref.range().kind() == z3.Z3_BOOL_SORT:
            bool_predicates[name] = pred_ref
        else:
            nonbool_functions[name] = pred_ref

    pred_signatures: dict[str, list[str]] = {}
    for pred_name, pred_ref in bool_predicates.items():
        sig = []
        for i in range(pred_ref.arity()):
            arg_sort = pred_ref.domain(i)
            sig.append(sort_str_to_name.get(str(arg_sort), str(arg_sort)))
        pred_signatures[pred_name] = sig

    # ----------------------------------------------------------------
    # Step 6: Extract boolean-predicate interpretations
    # ----------------------------------------------------------------
    pred_interp: dict[str, dict] = {}
    for pred_name, pred_ref in bool_predicates.items():
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

    # ----------------------------------------------------------------
    # Step 7: Booleanize non-Bool functions as {name}_is(..., value) atoms.
    #
    # For a function f: Sort_a × Sort_b → Sort_c (non-Bool), we add:
    #   (a) a boolean predicate f_is(arg_a, arg_b, val_c) = True iff f(...) = val_c
    #   (b) a direct value entry function_values["f(arg_a, arg_b)"] = actual_py_val
    #
    # (a) keeps truth["pred(...)"] lookups uniform.
    # (b) enables value["f(x)"] < value["f(y)"] comparisons in Phase 3,
    #     which is far less error-prone than nested any(truth[f_is(...)]) loops.
    # ----------------------------------------------------------------
    function_values: dict[str, Any] = {}       # "fname(arg, ...)" → Python value
    raw_function_signatures: dict[str, list[str]] = {}  # "fname" → [arg sorts]

    for func_name, func_ref in nonbool_functions.items():
        arity = func_ref.arity()
        range_sort = func_ref.range()
        range_sort_name = sort_str_to_name.get(str(range_sort), str(range_sort))

        # Build argument sort signature for the raw function (without _is suffix)
        raw_sig: list[str] = []
        arg_expr_lists_f: list[list[z3.ExprRef]] = []
        for i in range(arity):
            arg_sort = func_ref.domain(i)
            raw_sig.append(sort_str_to_name.get(str(arg_sort), str(arg_sort)))
            matching_f: list[z3.ExprRef] = []
            for sort_name, sort_ref in sorts.items():
                if sort_ref.kind() == arg_sort.kind() and str(sort_ref) == str(arg_sort):
                    matching_f = sort_to_exprs[sort_name]
                    break
            if not matching_f:
                matching_f = all_entity_exprs
            arg_expr_lists_f.append(matching_f)
        raw_function_signatures[func_name] = raw_sig

        # Get range universe (values the function can return)
        range_universe: list[z3.ExprRef] = list(
            sort_to_exprs.get(range_sort_name, [])
            or model.get_universe(range_sort)
            or []
        )

        bool_name = f"{func_name}_is"
        # Signature: input sorts + range sort
        pred_signatures[bool_name] = raw_sig + [range_sort_name]

        interp_f: dict = {}
        for combo_exprs in product(*arg_expr_lists_f):
            try:
                actual_val = model.evaluate(func_ref(*combo_exprs), model_completion=True)
                actual_str = z3_to_user.get(str(actual_val), str(actual_val))
            except Exception:
                logger.debug(
                    "model_extractor: could not evaluate %s(%s), skipping",
                    func_name, combo_exprs,
                )
                continue

            combo_strs = tuple(z3_to_user.get(str(e), str(e)) for e in combo_exprs)

            # (a) Booleanized f_is atoms
            if range_universe:
                for range_expr in range_universe:
                    range_str = z3_to_user.get(str(range_expr), str(range_expr))
                    key = combo_strs + (range_str,)
                    interp_f[key] = (range_str == actual_str)
            else:
                key = combo_strs + (actual_str,)
                interp_f[key] = True

            # (b) Direct value entry: "fname(arg1, arg2)" → Python value
            val_key = f"{func_name}({', '.join(combo_strs)})"
            function_values[val_key] = _parse_numeric_value(actual_str)

        if interp_f:
            pred_interp[bool_name] = interp_f

    return {
        "entities": display_entities,
        "sorts": sort_display,
        "predicate_signatures": pred_signatures,
        "predicates": pred_interp,
        "constants": const_interp,
        "raw_model_str": str(model),
        # Direct function-value lookup table for value["f(entity)"] expressions.
        "function_values": function_values,
        # Argument sort signatures for raw function names (without _is suffix);
        # used by Phase 3 validator to sort-check value[f"..."] fstring loops.
        "raw_function_signatures": raw_function_signatures,
    }


def _parse_numeric_value(s: str) -> Any:
    """
    Convert a Z3 model value string to a Python int, float, or string.

    Enables value["f(x)"] < value["f(y)"] comparisons in Phase 3 grounded formulas.
    Z3 integer values come back as plain decimal strings (e.g. "42300000000");
    rationals come as "p/q" which we convert to float.
    """
    try:
        return int(s)
    except ValueError:
        pass
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            return int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(s)
    except ValueError:
        return s


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
        bool_sigs = {k: v for k, v in pred_sigs.items() if not k.endswith("_is")}
        func_sigs  = {k: v for k, v in pred_sigs.items() if k.endswith("_is")}

        if bool_sigs:
            lines.append("Predicate signatures (use these names exactly):")
            for pred_name, arg_sorts in bool_sigs.items():
                lines.append(f"  {pred_name}({', '.join(arg_sorts)}) → bool")

        if func_sigs:
            lines.append("")
            lines.append(
                "Function-value relations (truth[\"f_is(entity, v)\"] = True iff f(entity) = v):"
            )
            for pred_name, arg_sorts in func_sigs.items():
                arg_part = ", ".join(arg_sorts[:-1])
                ret_part = arg_sorts[-1]
                lines.append(f"  {pred_name}({arg_part}, {ret_part}) → bool")

    lines.append("")

    # Canonical entity IDs
    all_entities = domain.get("entities", [])
    if all_entities:
        lines.append(f"Canonical entity IDs (use these EXACTLY as truth[...] keys):")
        lines.append(f"  {', '.join(all_entities)}")

    lines.append("")

    # Function values — the preferred lookup table for numeric comparisons.
    # LLM should write  value["fname(entity)"]  to get the actual numeric value
    # and compare directly, e.g.  value["score(alice)"] < value["score(bob)"].
    function_values: dict = domain.get("function_values", {})
    if function_values:
        lines.append(
            "Function values (use value[\"fname(entity)\"] for comparisons, "
            "e.g. value[\"score(alice)\"] < value[\"score(bob)\"]):"
        )
        for fval_key, fval in function_values.items():
            lines.append(f"  value[\"{fval_key}\"] = {fval}")
        lines.append("")

    # Ground atoms
    lines.append("Ground atoms:")
    for pred_name, interp in domain["predicates"].items():
        for args, val in interp.items():
            args_str = ", ".join(args)
            lines.append(f"  {pred_name}({args_str}) = {val}")

    return "\n".join(lines)


def format_domain_schema(domain: dict) -> str:
    """
    Format a domain description WITHOUT truth values — only structural information.

    Outputs: sorts + entity lists, predicate signatures, function signatures,
    canonical entity IDs. Intentionally omits the Ground atoms section.

    Used by Phase 3 Template Generation (Template-Once design) so the LLM
    writes formulas based solely on NL meaning + domain schema, without being
    biased by specific truth values in any particular witness world.
    This preserves structural independence from Phase 1 (Proposition 3).
    """
    lines = []

    # Sort-grouped entity lists
    sorts: dict[str, list[str]] = domain.get("sorts", {})
    if sorts:
        lines.append("Entities by sort:")
        for sort_name, entities in sorts.items():
            lines.append(f"  {sort_name}: {entities}")
    else:
        lines.append(f"Domain entities: {domain.get('entities', [])}")

    lines.append("")

    # Predicate signatures (bool predicates)
    pred_sigs: dict[str, list[str]] = domain.get("predicate_signatures", {})
    if pred_sigs:
        bool_sigs = {k: v for k, v in pred_sigs.items() if not k.endswith("_is")}
        func_sigs  = {k: v for k, v in pred_sigs.items() if k.endswith("_is")}

        if bool_sigs:
            lines.append("Predicate signatures (use these names exactly):")
            for pred_name, arg_sorts in bool_sigs.items():
                lines.append(f"  {pred_name}({', '.join(arg_sorts)}) → bool")

        if func_sigs:
            lines.append("")
            lines.append("Function-value signatures (access via value[\"fname(entity)\"]):")
            for pred_name, arg_sorts in func_sigs.items():
                func_name = pred_name.removesuffix("_is")
                arg_part = ", ".join(arg_sorts[:-1])
                ret_part = arg_sorts[-1]
                lines.append(f"  {func_name}({arg_part}) → {ret_part}")

    lines.append("")

    # Raw function signatures (for value[...] access)
    raw_func_sigs: dict[str, list[str]] = domain.get("raw_function_signatures", {})
    if raw_func_sigs and not pred_sigs:
        lines.append("Function signatures:")
        for fname, arg_sorts in raw_func_sigs.items():
            lines.append(f"  {fname}({', '.join(arg_sorts)}) → value")
        lines.append("")

    # Canonical entity IDs
    all_entities = domain.get("entities", [])
    if all_entities:
        lines.append(f"Canonical entity IDs (use these EXACTLY as truth[...] keys):")
        lines.append(f"  {', '.join(all_entities)}")

    return "\n".join(lines)


def format_filtered_domain_desc(
    domain: dict,
    relevant_predicates: set[str],
) -> str:
    """Format domain description filtered to only show predicates relevant
    to a specific repair target.

    Like format_domain_desc() but omits ground atoms for predicates not in
    relevant_predicates. Keeps entity/sort/canonical ID sections intact
    since they are compact and provide necessary context.
    """
    lines: list[str] = []

    # Sort-grouped entity lists (always shown — compact)
    sorts: dict[str, list[str]] = domain.get("sorts", {})
    if sorts:
        lines.append("Entities by sort:")
        for sort_name, entities in sorts.items():
            lines.append(f"  {sort_name}: {entities}")
    else:
        lines.append(f"Domain entities: {domain['entities']}")
    lines.append("")

    # Predicate signatures (filtered)
    pred_sigs: dict[str, list[str]] = domain.get("predicate_signatures", {})
    if pred_sigs:
        # Include a predicate if it or its base (without _is) is relevant
        filtered_sigs = {
            k: v for k, v in pred_sigs.items()
            if k in relevant_predicates
            or k.removesuffix("_is") in relevant_predicates
        }
        bool_sigs = {k: v for k, v in filtered_sigs.items() if not k.endswith("_is")}
        func_sigs = {k: v for k, v in filtered_sigs.items() if k.endswith("_is")}

        if bool_sigs:
            lines.append("Predicate signatures (relevant to this repair):")
            for pred_name, arg_sorts in bool_sigs.items():
                lines.append(f"  {pred_name}({', '.join(arg_sorts)}) → bool")
        if func_sigs:
            lines.append("")
            lines.append("Function-value relations (relevant):")
            for pred_name, arg_sorts in func_sigs.items():
                func_name = pred_name.removesuffix("_is")
                arg_part = ", ".join(arg_sorts[:-1])
                ret_part = arg_sorts[-1]
                lines.append(f"  value[\"{func_name}({arg_part})\"] : {ret_part}")
    lines.append("")

    # Canonical entity IDs (always shown)
    all_entities = domain.get("entities", [])
    if all_entities:
        lines.append(f"Canonical entity IDs: {', '.join(all_entities)}")
    lines.append("")

    # Function values (filtered)
    function_values: dict = domain.get("function_values", {})
    if function_values:
        filtered_fv = {
            k: v for k, v in function_values.items()
            if any(p in k for p in relevant_predicates)
        }
        if filtered_fv:
            lines.append("Function values (relevant):")
            for fval_key, fval in filtered_fv.items():
                lines.append(f'  value["{fval_key}"] = {fval}')
            lines.append("")

    # Ground atoms — only for relevant predicates
    all_preds = domain.get("predicates", {})
    relevant_count = 0
    omitted_count = 0
    lines.append("Ground atoms (relevant to this repair):")
    for pred_name, interp in all_preds.items():
        if (pred_name in relevant_predicates
                or pred_name.removesuffix("_is") in relevant_predicates):
            for args, val in interp.items():
                args_str = ", ".join(args)
                # For _is predicates (booleanized functions), show in value[]
                # notation consistent with normalized grounded formulas.
                if pred_name.endswith("_is") and len(args) >= 2 and val is True:
                    func_name = pred_name.removesuffix("_is")
                    entity_args = ", ".join(args[:-1])
                    value_arg = args[-1]
                    lines.append(f'  value["{func_name}({entity_args})"] == "{value_arg}"')
                else:
                    lines.append(f"  {pred_name}({args_str}) = {val}")
                relevant_count += 1
        else:
            omitted_count += len(interp)

    if omitted_count > 0:
        lines.append(f"  ... ({omitted_count} ground atoms from other predicates omitted)")

    return "\n".join(lines)
