from __future__ import annotations

import json
import re
from itertools import product
from typing import Any


_ALLOWED_BOOL_OPS = {
    "and",
    "or",
    "not",
    "implies",
    "iff",
    "const_bool",
    "truth",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "all",
    "any",
}

_ALLOWED_TERM_OPS = {"const", "var", "value"}


class IRValidationError(ValueError):
    """Raised when grounded-template IR fails static checks."""


def parse_grounded_template_ir(
    payload: dict[str, Any] | str,
    symbol_context: dict[str, object],
) -> dict[str, Any]:
    """Parse one grounded-template IR payload into a validated JSON object shape."""
    obj = payload
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n", "", text)
            text = re.sub(r"\n```\s*$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise IRValidationError(f"Invalid JSON IR: {e}") from e

    if not isinstance(obj, dict):
        raise IRValidationError("Grounded-template IR must be a JSON object.")

    obj = _normalise_ir_payload(obj, symbol_context)
    expr = obj.get("expr")
    if not isinstance(expr, dict):
        raise IRValidationError("Field 'expr' must be a JSON object.")

    return {"expr": expr}


def _normalise_ir_payload(
    payload: dict[str, Any],
    symbol_context: dict[str, object],
) -> dict[str, Any]:
    """Apply lossless compatibility rewrites for common legacy IR aliases."""
    normalised = dict(payload)
    ctx = _build_symbol_context(symbol_context)
    expr = normalised.get("expr")
    if isinstance(expr, dict):
        normalised["expr"] = _normalise_bool_expr_node(expr, ctx=ctx, scope=set())
        return normalised
    if _looks_like_bool_expr_node(normalised):
        return {"expr": _normalise_bool_expr_node(normalised, ctx=ctx, scope=set())}
    return normalised


def _looks_like_bool_expr_node(node: dict[str, Any]) -> bool:
    op = node.get("op")
    return isinstance(op, str) and bool(op.strip())


def _normalise_bool_expr_node(
    node: dict[str, Any],
    ctx: dict[str, Any],
    scope: set[str],
) -> dict[str, Any]:
    canonical = _coerce_legacy_bool_node(node)
    raw_op = str(canonical.get("op", "")).strip()
    op = {"forall": "all", "exists": "any"}.get(raw_op.lower(), raw_op.lower())
    out = dict(canonical)
    if op != raw_op:
        out["op"] = op

    if op in {"and", "or"}:
        args = out.get("args", node.get("args"))
        if isinstance(args, list):
            out["args"] = [
                _normalise_bool_expr_node(arg, ctx=ctx, scope=scope)
                if isinstance(arg, dict)
                else arg
                for arg in args
            ]
        elif isinstance(args, dict):
            out["args"] = [_normalise_bool_expr_node(args, ctx=ctx, scope=scope)]
        return out

    if op == "not":
        arg = out.get("arg", node.get("arg"))
        if not isinstance(arg, dict):
            legacy_args = out.get("args", node.get("args"))
            if isinstance(legacy_args, list) and len(legacy_args) == 1 and isinstance(legacy_args[0], dict):
                arg = legacy_args[0]
            elif isinstance(legacy_args, dict):
                arg = legacy_args
        if isinstance(arg, dict):
            out["arg"] = _normalise_bool_expr_node(arg, ctx=ctx, scope=scope)
            out.pop("args", None)
        return out

    if op in {"implies", "iff"}:
        left = out.get("left", node.get("left"))
        right = out.get("right", node.get("right"))
        if not isinstance(left, dict) or not isinstance(right, dict):
            legacy_lhs = out.get("lhs", node.get("lhs"))
            legacy_rhs = out.get("rhs", node.get("rhs"))
            legacy_antecedent = out.get("antecedent", node.get("antecedent"))
            legacy_consequent = out.get("consequent", node.get("consequent"))
            legacy_args = out.get("args", node.get("args"))
            if isinstance(legacy_lhs, dict) and isinstance(legacy_rhs, dict):
                left, right = legacy_lhs, legacy_rhs
            elif isinstance(legacy_antecedent, dict) and isinstance(legacy_consequent, dict):
                left, right = legacy_antecedent, legacy_consequent
            elif (
                isinstance(legacy_args, list)
                and len(legacy_args) == 2
                and isinstance(legacy_args[0], dict)
                and isinstance(legacy_args[1], dict)
            ):
                left, right = legacy_args[0], legacy_args[1]
        if isinstance(left, dict) and isinstance(right, dict):
            out["left"] = _normalise_bool_expr_node(left, ctx=ctx, scope=scope)
            out["right"] = _normalise_bool_expr_node(right, ctx=ctx, scope=scope)
            out.pop("lhs", None)
            out.pop("rhs", None)
            out.pop("antecedent", None)
            out.pop("consequent", None)
            out.pop("args", None)
        return out

    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = out.get("left", node.get("left"))
        right = out.get("right", node.get("right"))
        if not isinstance(left, dict) or not isinstance(right, dict):
            legacy_lhs = out.get("lhs", node.get("lhs"))
            legacy_rhs = out.get("rhs", node.get("rhs"))
            legacy_args = out.get("args", node.get("args"))
            if isinstance(legacy_lhs, dict) and isinstance(legacy_rhs, dict):
                left, right = legacy_lhs, legacy_rhs
            elif (
                isinstance(legacy_args, list)
                and len(legacy_args) == 2
                and isinstance(legacy_args[0], dict)
                and isinstance(legacy_args[1], dict)
            ):
                left, right = legacy_args[0], legacy_args[1]
        if isinstance(left, dict) and isinstance(right, dict):
            out["left"] = _normalise_term_expr_node(left, ctx=ctx, scope=scope)
            out["right"] = _normalise_term_expr_node(right, ctx=ctx, scope=scope)
            out.pop("args", None)
            out.pop("lhs", None)
            out.pop("rhs", None)
        return out

    if op == "truth":
        pred = out.get("pred")
        if not isinstance(pred, str) or not pred:
            for legacy_key in ("predicate", "name", "func", "function"):
                legacy_pred = out.get(legacy_key, node.get(legacy_key))
                if isinstance(legacy_pred, str) and legacy_pred:
                    out["pred"] = legacy_pred
                    break
        args = out.get("args", node.get("args"))
        if isinstance(args, list):
            out["args"] = [
                _normalise_term_expr_node(arg, ctx=ctx, scope=scope)
                if isinstance(arg, dict)
                else arg
                for arg in args
            ]
        elif isinstance(args, dict):
            out["args"] = [_normalise_term_expr_node(args, ctx=ctx, scope=scope)]
        elif isinstance(out.get("arg"), dict):
            out["args"] = [_normalise_term_expr_node(out["arg"], ctx=ctx, scope=scope)]
        return out

    if op in {"all", "any"}:
        vars_obj = out.get("vars", node.get("vars"))
        if not isinstance(vars_obj, list):
            legacy_vars = out.get("variables", node.get("variables"))
            if isinstance(legacy_vars, list):
                vars_obj = legacy_vars
            elif isinstance(legacy_vars, dict):
                vars_obj = [legacy_vars]
        if isinstance(vars_obj, dict):
            vars_obj = [vars_obj]
        child_scope = set(scope)
        if isinstance(vars_obj, list):
            normalised_vars: list[dict[str, Any]] = []
            for var_def in vars_obj:
                if isinstance(var_def, dict):
                    var_name = var_def.get("name")
                    if not isinstance(var_name, str) or not var_name:
                        legacy_name = var_def.get("var")
                        if isinstance(legacy_name, str) and legacy_name:
                            var_name = legacy_name
                    var_sort = var_def.get("sort")
                    if not isinstance(var_sort, str) or not var_sort:
                        legacy_sort = var_def.get("type")
                        if isinstance(legacy_sort, str) and legacy_sort:
                            var_sort = legacy_sort
                    item = dict(var_def)
                    if isinstance(var_name, str) and var_name:
                        item["name"] = var_name
                        child_scope.add(var_name)
                    if isinstance(var_sort, str) and var_sort:
                        item["sort"] = var_sort
                    normalised_vars.append(item)
            out["vars"] = normalised_vars
            out.pop("variables", None)
        body = out.get("body", node.get("body"))
        if isinstance(body, dict):
            out["body"] = _normalise_bool_expr_node(body, ctx=ctx, scope=child_scope)
        return out

    return out


def _normalise_term_expr_node(
    node: dict[str, Any],
    ctx: dict[str, Any],
    scope: set[str],
) -> dict[str, Any]:
    canonical = _coerce_legacy_term_node(node)
    op = str(canonical.get("op", "")).strip().lower()
    out = dict(canonical)

    if op == "const":
        name = node.get("name")
        if not isinstance(name, str) or not name:
            value = node.get("value")
            if isinstance(value, str) and value:
                out["name"] = value
        return out

    if op == "value":
        if not isinstance(out.get("func"), str) or not out.get("func"):
            for legacy_key in ("name", "function"):
                legacy_func = out.get(legacy_key, node.get(legacy_key))
                if isinstance(legacy_func, str) and legacy_func:
                    out["func"] = legacy_func
                    break

        raw_args = out.get("args", node.get("args"))
        if raw_args is None:
            raw_args = out.get("arguments", out.get("terms", node.get("arguments", node.get("terms"))))
        normalised_args: list[Any] = []
        if isinstance(raw_args, list):
            normalised_args = [
                _normalise_term_expr_node(arg, ctx=ctx, scope=scope)
                if isinstance(arg, dict)
                else arg
                for arg in raw_args
            ]
            out["args"] = normalised_args
        elif isinstance(raw_args, dict):
            normalised_args = [_normalise_term_expr_node(raw_args, ctx=ctx, scope=scope)]
            out["args"] = normalised_args

        func = node.get("func")
        if (
            isinstance(func, str)
            and func
            and func not in ctx["functions"]
            and func in ctx["constants"]
            and len(normalised_args) == 0
        ):
            return {"op": "const", "name": func}
        return out

    if op == "var":
        name = node.get("name")
        if (
            isinstance(name, str)
            and name
            and name not in scope
            and name in ctx["constants"]
        ):
            return {"op": "const", "name": name}
        return out

    return out


def _coerce_legacy_bool_node(node: dict[str, Any]) -> dict[str, Any]:
    if "op" in node and isinstance(node.get("op"), str) and str(node.get("op")).strip():
        return dict(node)

    # Single-key shorthand nodes: {"and":[...]}, {"truth":{...}}, {"implies":[l,r]}, ...
    if len(node) == 1:
        key, value = next(iter(node.items()))
        op = str(key).strip().lower()
        if op in {"forall", "exists", "all", "any"}:
            if isinstance(value, dict):
                merged = {"op": op}
                merged.update(value)
                return merged
            return {"op": op}
        if op in {"and", "or"}:
            return {"op": op, "args": value}
        if op == "not":
            return {"op": op, "arg": value}
        if op in {"implies", "iff"}:
            if isinstance(value, list) and len(value) == 2:
                return {"op": op, "left": value[0], "right": value[1]}
            if isinstance(value, dict):
                merged = {"op": op}
                merged.update(value)
                return merged
            return {"op": op}
        if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if isinstance(value, list) and len(value) == 2:
                return {"op": op, "left": value[0], "right": value[1]}
            if isinstance(value, dict):
                merged = {"op": op}
                merged.update(value)
                return merged
            return {"op": op}
        if op == "truth":
            if isinstance(value, dict):
                merged = {"op": "truth"}
                merged.update(value)
                return merged
            return {"op": "truth"}
        if op == "const_bool":
            return {"op": "const_bool", "value": value}

    # Legacy tree with top-level boolean keys and no explicit op.
    for key in ("and", "or", "not", "implies", "iff", "truth", "eq", "ne", "lt", "le", "gt", "ge", "all", "any", "forall", "exists"):
        if key in node:
            value = node[key]
            if key in {"and", "or"}:
                return {"op": key, "args": value}
            if key == "not":
                return {"op": key, "arg": value}
            if key in {"implies", "iff"}:
                if isinstance(value, list) and len(value) == 2:
                    return {"op": key, "left": value[0], "right": value[1]}
                if isinstance(value, dict):
                    merged = {"op": key}
                    merged.update(value)
                    return merged
            if key in {"eq", "ne", "lt", "le", "gt", "ge"}:
                if isinstance(value, list) and len(value) == 2:
                    return {"op": key, "left": value[0], "right": value[1]}
                if isinstance(value, dict):
                    merged = {"op": key}
                    merged.update(value)
                    return merged
            if key == "truth":
                if isinstance(value, dict):
                    merged = {"op": "truth"}
                    merged.update(value)
                    return merged
                return {"op": "truth"}
            if key in {"all", "any", "forall", "exists"}:
                if isinstance(value, dict):
                    merged = {"op": key}
                    merged.update(value)
                    return merged
                return {"op": key}

    return dict(node)


def _coerce_legacy_term_node(node: dict[str, Any]) -> dict[str, Any]:
    if "op" in node and isinstance(node.get("op"), str) and str(node.get("op")).strip():
        return dict(node)

    if len(node) == 1:
        key, value = next(iter(node.items()))
        op = str(key).strip().lower()
        if op == "const":
            if isinstance(value, dict):
                merged = {"op": "const"}
                merged.update(value)
                return merged
            return {"op": "const", "name": value}
        if op == "var":
            if isinstance(value, dict):
                merged = {"op": "var"}
                merged.update(value)
                return merged
            return {"op": "var", "name": value}
        if op == "value":
            if isinstance(value, dict):
                merged = {"op": "value"}
                merged.update(value)
                return merged
            return {"op": "value"}

    if "func" in node and ("args" in node or "arguments" in node or "terms" in node):
        out = {"op": "value", "func": node.get("func")}
        if "args" in node:
            out["args"] = node.get("args")
        elif "arguments" in node:
            out["args"] = node.get("arguments")
        else:
            out["args"] = node.get("terms")
        return out

    if "const" in node and isinstance(node.get("const"), str):
        return {"op": "const", "name": node.get("const")}
    if "var" in node and isinstance(node.get("var"), str):
        return {"op": "var", "name": node.get("var")}

    return dict(node)


def validate_grounded_template_ir(
    ir: dict[str, Any],
    symbol_context: dict[str, object],
    required_predicates: set[str] | None = None,
) -> None:
    """Validate a grounded-template IR against the frozen symbol context."""
    ctx = _build_symbol_context(symbol_context)
    expr = ir.get("expr")
    if not isinstance(expr, dict):
        raise IRValidationError("IR field 'expr' must be an object.")

    used_names: set[str] = set()
    _validate_bool_expr(expr, ctx=ctx, scope={}, used_names=used_names)

    if required_predicates:
        missing = sorted(required_predicates - used_names)
        if missing:
            raise IRValidationError(
                "Grounded template drops core predicate/function names from "
                f"sentence DSL logic: {missing[:8]}"
            )


def evaluate_grounded_template_ir(
    ir: dict[str, Any],
    domain: dict[str, Any],
) -> bool | None:
    """Evaluate grounded-template IR on one witness domain."""
    try:
        expr = ir["expr"] if "expr" in ir else ir
        if not isinstance(expr, dict):
            return None

        truth_table: dict[str, dict[tuple[str, ...], bool]] = {}
        for pred_name, interp in domain.get("predicates", {}).items():
            if not isinstance(interp, dict):
                continue
            pred_interp: dict[tuple[str, ...], bool] = {}
            for args, val in interp.items():
                pred_interp[tuple(str(x) for x in args)] = bool(val)
            truth_table[str(pred_name)] = pred_interp

        value_table: dict[str, Any] = dict(domain.get("function_values", {}))
        sorts: dict[str, list[str]] = {
            str(sort): [str(ent) for ent in ents]
            for sort, ents in domain.get("sorts", {}).items()
            if isinstance(ents, list)
        }
        env: dict[str, Any] = {}
        result = _eval_bool_expr(
            expr,
            env=env,
            sorts=sorts,
            truth_table=truth_table,
            value_table=value_table,
        )
        return bool(result)
    except Exception:
        return None


def render_grounded_template_ir(ir: dict[str, Any]) -> str:
    """Render grounded-template IR into a compact debug string."""
    expr = ir.get("expr", ir)
    if not isinstance(expr, dict):
        return json.dumps(ir, ensure_ascii=False)
    try:
        return _render_bool_expr(expr)
    except Exception:
        return json.dumps(ir, ensure_ascii=False)


def _build_symbol_context(symbol_context: dict[str, object]) -> dict[str, Any]:
    sorts: set[str] = set()
    sort_kinds: dict[str, str] = {}
    predicates: dict[str, list[str]] = {}
    functions: dict[str, dict[str, Any]] = {}
    constants: dict[str, str] = {}
    variables: dict[str, str] = {}

    raw_sorts = symbol_context.get("sorts", [])
    if isinstance(raw_sorts, list):
        for item in raw_sorts:
            if isinstance(item, dict) and "name" in item:
                sort_name = str(item["name"])
                sorts.add(sort_name)
                if "type" in item:
                    sort_kinds[sort_name] = str(item.get("type", ""))

    raw_functions = symbol_context.get("functions", [])
    if isinstance(raw_functions, list):
        for item in raw_functions:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if not name:
                continue
            domain = [str(s) for s in item.get("domain", []) if isinstance(s, str)]
            range_sort = str(item.get("range", ""))
            if range_sort:
                sorts.add(range_sort)
            for srt in domain:
                sorts.add(srt)
            if _is_bool_sort(range_sort):
                predicates[name] = domain
            else:
                functions[name] = {"domain": domain, "range": range_sort}

    raw_constants = symbol_context.get("constants", {})
    if isinstance(raw_constants, dict):
        for _, spec in raw_constants.items():
            if not isinstance(spec, dict):
                continue
            sort_name = str(spec.get("sort", ""))
            if sort_name:
                sorts.add(sort_name)
            members = spec.get("members", [])
            if isinstance(members, list):
                for member in members:
                    constants[str(member)] = sort_name
            elif isinstance(members, dict):
                for member in members:
                    constants[str(member)] = sort_name

    raw_variables = symbol_context.get("variables", [])
    if isinstance(raw_variables, list):
        for item in raw_variables:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            sort_name = str(item.get("sort", ""))
            if not name:
                continue
            variables[name] = sort_name
            if sort_name:
                sorts.add(sort_name)

    return {
        "sorts": sorts,
        "sort_kinds": sort_kinds,
        "predicates": predicates,
        "functions": functions,
        "constants": constants,
        "variables": variables,
    }


def _validate_bool_expr(
    node: dict[str, Any],
    ctx: dict[str, Any],
    scope: dict[str, str],
    used_names: set[str],
) -> None:
    op = str(node.get("op", ""))
    if op not in _ALLOWED_BOOL_OPS:
        raise IRValidationError(f"Unsupported boolean op '{op}'.")

    if op in {"and", "or"}:
        args = node.get("args")
        if not isinstance(args, list) or len(args) == 0:
            raise IRValidationError(f"Op '{op}' requires non-empty 'args'.")
        for arg in args:
            if not isinstance(arg, dict):
                raise IRValidationError(f"Op '{op}' args must be objects.")
            _validate_bool_expr(arg, ctx=ctx, scope=scope, used_names=used_names)
        return

    if op == "not":
        arg = node.get("arg")
        if not isinstance(arg, dict):
            raise IRValidationError("Op 'not' requires object field 'arg'.")
        _validate_bool_expr(arg, ctx=ctx, scope=scope, used_names=used_names)
        return

    if op in {"implies", "iff"}:
        left = node.get("left")
        right = node.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise IRValidationError(f"Op '{op}' requires object fields 'left' and 'right'.")
        _validate_bool_expr(left, ctx=ctx, scope=scope, used_names=used_names)
        _validate_bool_expr(right, ctx=ctx, scope=scope, used_names=used_names)
        return

    if op == "const_bool":
        if not isinstance(node.get("value"), bool):
            raise IRValidationError("Op 'const_bool' requires boolean field 'value'.")
        return

    if op == "truth":
        pred = node.get("pred")
        args = node.get("args")
        if not isinstance(pred, str) or not pred:
            raise IRValidationError("Op 'truth' requires non-empty string field 'pred'.")
        if pred not in ctx["predicates"]:
            raise IRValidationError(f"Undeclared predicate '{pred}'.")
        if not isinstance(args, list):
            raise IRValidationError("Op 'truth' requires list field 'args'.")
        expected_sorts: list[str] = ctx["predicates"][pred]
        if len(args) != len(expected_sorts):
            raise IRValidationError(
                f"Predicate '{pred}' arity mismatch: expected {len(expected_sorts)}, got {len(args)}."
            )
        used_names.add(pred)
        for arg, expected_sort in zip(args, expected_sorts, strict=False):
            if not isinstance(arg, dict):
                raise IRValidationError(f"Predicate '{pred}' args must be objects.")
            actual_sort = _validate_term_expr(arg, ctx=ctx, scope=scope, used_names=used_names)
            _assert_sort_compatible(actual_sort, expected_sort, f"predicate '{pred}'")
        return

    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = node.get("left")
        right = node.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise IRValidationError(f"Op '{op}' requires object fields 'left' and 'right'.")
        left_sort = _validate_term_expr(left, ctx=ctx, scope=scope, used_names=used_names)
        right_sort = _validate_term_expr(right, ctx=ctx, scope=scope, used_names=used_names)
        if op in {"eq", "ne"}:
            _assert_term_pair_compatible(left_sort, right_sort, op)
        else:
            _assert_order_comparable(left_sort, right_sort, ctx=ctx, op=op)
        return

    if op in {"all", "any"}:
        vars_obj = node.get("vars")
        body = node.get("body")
        if not isinstance(vars_obj, list) or len(vars_obj) == 0:
            raise IRValidationError(f"Op '{op}' requires non-empty list field 'vars'.")
        if not isinstance(body, dict):
            raise IRValidationError(f"Op '{op}' requires object field 'body'.")
        child_scope = dict(scope)
        for var_def in vars_obj:
            if not isinstance(var_def, dict):
                raise IRValidationError(f"Op '{op}' vars entries must be objects.")
            name = str(var_def.get("name", ""))
            sort_name = str(var_def.get("sort", ""))
            if not name or not sort_name:
                raise IRValidationError(f"Op '{op}' vars require both 'name' and 'sort'.")
            if sort_name not in ctx["sorts"]:
                raise IRValidationError(f"Variable '{name}' uses undeclared sort '{sort_name}'.")
            child_scope[name] = sort_name
        _validate_bool_expr(body, ctx=ctx, scope=child_scope, used_names=used_names)
        return

    raise IRValidationError(f"Unsupported boolean op '{op}'.")


def _validate_term_expr(
    node: dict[str, Any],
    ctx: dict[str, Any],
    scope: dict[str, str],
    used_names: set[str],
) -> str | None:
    op = str(node.get("op", ""))
    if op not in _ALLOWED_TERM_OPS:
        raise IRValidationError(
            f"Expected term op in {_ALLOWED_TERM_OPS}, got '{op}'."
        )

    if op == "const":
        name = node.get("name")
        if not isinstance(name, str) or not name:
            raise IRValidationError("Op 'const' requires non-empty string field 'name'.")
        if name not in ctx["constants"]:
            raise IRValidationError(f"Undeclared constant '{name}'.")
        return ctx["constants"][name] or None

    if op == "var":
        name = node.get("name")
        if not isinstance(name, str) or not name:
            raise IRValidationError("Op 'var' requires non-empty string field 'name'.")
        if name not in scope:
            raise IRValidationError(f"Undeclared variable '{name}' in current scope.")
        return scope[name] or None

    if op == "value":
        func = node.get("func")
        args = node.get("args")
        if not isinstance(func, str) or not func:
            raise IRValidationError("Op 'value' requires non-empty string field 'func'.")
        if func not in ctx["functions"]:
            raise IRValidationError(f"Undeclared function '{func}'.")
        if not isinstance(args, list):
            raise IRValidationError("Op 'value' requires list field 'args'.")
        sig = ctx["functions"][func]
        expected_sorts: list[str] = sig["domain"]
        if len(args) != len(expected_sorts):
            raise IRValidationError(
                f"Function '{func}' arity mismatch: expected {len(expected_sorts)}, got {len(args)}."
            )
        used_names.add(func)
        for arg, expected_sort in zip(args, expected_sorts, strict=False):
            if not isinstance(arg, dict):
                raise IRValidationError(f"Function '{func}' args must be objects.")
            actual_sort = _validate_term_expr(arg, ctx=ctx, scope=scope, used_names=used_names)
            _assert_sort_compatible(actual_sort, expected_sort, f"function '{func}'")
        return sig.get("range") or None

    raise IRValidationError(f"Unsupported term op '{op}'.")


def _assert_sort_compatible(actual: str | None, expected: str | None, owner: str) -> None:
    if not actual or not expected:
        return
    if actual != expected:
        raise IRValidationError(
            f"Sort mismatch in {owner}: expected '{expected}', got '{actual}'."
        )


def _assert_term_pair_compatible(left_sort: str | None, right_sort: str | None, op: str) -> None:
    if not left_sort or not right_sort:
        return
    if left_sort != right_sort:
        raise IRValidationError(
            f"Sort mismatch for '{op}': left is '{left_sort}', right is '{right_sort}'."
        )


def _assert_order_comparable(
    left_sort: str | None,
    right_sort: str | None,
    ctx: dict[str, Any],
    op: str,
) -> None:
    if not left_sort or not right_sort:
        raise IRValidationError(
            f"Ordering operator '{op}' requires both operand sorts to be known."
        )
    if left_sort != right_sort:
        raise IRValidationError(
            f"Sort mismatch for '{op}': left is '{left_sort}', right is '{right_sort}'."
        )
    if not _is_numeric_sort(left_sort, ctx):
        raise IRValidationError(
            f"Ordering operator '{op}' is only allowed on numeric sorts; got '{left_sort}'."
        )


def _eval_bool_expr(
    node: dict[str, Any],
    env: dict[str, Any],
    sorts: dict[str, list[str]],
    truth_table: dict[str, dict[tuple[str, ...], bool]],
    value_table: dict[str, Any],
) -> bool:
    op = str(node.get("op", ""))

    if op == "and":
        args = node["args"]
        return all(
            _eval_bool_expr(arg, env=env, sorts=sorts, truth_table=truth_table, value_table=value_table)
            for arg in args
        )
    if op == "or":
        args = node["args"]
        return any(
            _eval_bool_expr(arg, env=env, sorts=sorts, truth_table=truth_table, value_table=value_table)
            for arg in args
        )
    if op == "not":
        return not _eval_bool_expr(
            node["arg"], env=env, sorts=sorts, truth_table=truth_table, value_table=value_table
        )
    if op == "implies":
        left = _eval_bool_expr(
            node["left"], env=env, sorts=sorts, truth_table=truth_table, value_table=value_table
        )
        right = _eval_bool_expr(
            node["right"], env=env, sorts=sorts, truth_table=truth_table, value_table=value_table
        )
        return (not left) or right
    if op == "iff":
        left = _eval_bool_expr(
            node["left"], env=env, sorts=sorts, truth_table=truth_table, value_table=value_table
        )
        right = _eval_bool_expr(
            node["right"], env=env, sorts=sorts, truth_table=truth_table, value_table=value_table
        )
        return left == right
    if op == "const_bool":
        return bool(node["value"])
    if op == "truth":
        pred = str(node["pred"])
        args = tuple(
            str(_eval_term_expr(arg, env=env, truth_table=truth_table, value_table=value_table))
            for arg in node["args"]
        )
        pred_interp = truth_table.get(pred)
        if pred_interp is None or args not in pred_interp:
            raise KeyError(f"Missing truth entry for {pred}{args}")
        return bool(pred_interp[args])
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = _eval_term_expr(node["left"], env=env, truth_table=truth_table, value_table=value_table)
        right = _eval_term_expr(node["right"], env=env, truth_table=truth_table, value_table=value_table)
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "gt":
            return left > right
        return left >= right
    if op in {"all", "any"}:
        vars_obj = node["vars"]
        body = node["body"]
        value_lists = []
        names = []
        for var_def in vars_obj:
            name = str(var_def["name"])
            sort_name = str(var_def["sort"])
            if sort_name not in sorts:
                raise KeyError(f"Missing sort '{sort_name}' in witness domain.")
            names.append(name)
            value_lists.append(list(sorts[sort_name]))
        if not value_lists:
            return True if op == "all" else False
        tuples_iter = product(*value_lists)
        if op == "all":
            for values in tuples_iter:
                child_env = dict(env)
                child_env.update({k: v for k, v in zip(names, values, strict=False)})
                if not _eval_bool_expr(
                    body,
                    env=child_env,
                    sorts=sorts,
                    truth_table=truth_table,
                    value_table=value_table,
                ):
                    return False
            return True
        for values in tuples_iter:
            child_env = dict(env)
            child_env.update({k: v for k, v in zip(names, values, strict=False)})
            if _eval_bool_expr(
                body,
                env=child_env,
                sorts=sorts,
                truth_table=truth_table,
                value_table=value_table,
            ):
                return True
        return False
    raise KeyError(f"Unsupported op '{op}'.")


def _eval_term_expr(
    node: dict[str, Any],
    env: dict[str, Any],
    truth_table: dict[str, dict[tuple[str, ...], bool]],
    value_table: dict[str, Any],
) -> Any:
    op = str(node.get("op", ""))
    if op == "const":
        return str(node["name"])
    if op == "var":
        name = str(node["name"])
        if name not in env:
            raise KeyError(f"Variable '{name}' is not bound.")
        return env[name]
    if op == "value":
        func = str(node["func"])
        args = [
            str(_eval_term_expr(arg, env=env, truth_table=truth_table, value_table=value_table))
            for arg in node["args"]
        ]
        key = f"{func}({', '.join(args)})"
        if key not in value_table:
            raise KeyError(f"Missing value entry '{key}'.")
        return value_table[key]
    raise KeyError(f"Unsupported term op '{op}'.")


def _render_bool_expr(node: dict[str, Any]) -> str:
    op = str(node.get("op", ""))
    if op == "and":
        return "(" + " and ".join(_render_bool_expr(arg) for arg in node["args"]) + ")"
    if op == "or":
        return "(" + " or ".join(_render_bool_expr(arg) for arg in node["args"]) + ")"
    if op == "not":
        return f"(not {_render_bool_expr(node['arg'])})"
    if op == "implies":
        left = _render_bool_expr(node["left"])
        right = _render_bool_expr(node["right"])
        return f"(({left}) -> ({right}))"
    if op == "iff":
        left = _render_bool_expr(node["left"])
        right = _render_bool_expr(node["right"])
        return f"(({left}) <-> ({right}))"
    if op == "const_bool":
        return "True" if node.get("value") else "False"
    if op == "truth":
        args = ", ".join(_render_term_expr(arg) for arg in node.get("args", []))
        return f"truth({node.get('pred')}({args}))"
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = _render_term_expr(node["left"])
        right = _render_term_expr(node["right"])
        symbol = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }[op]
        return f"({left} {symbol} {right})"
    if op in {"all", "any"}:
        quant = "forall" if op == "all" else "exists"
        vars_text = ", ".join(f"{v['name']}:{v['sort']}" for v in node.get("vars", []))
        body = _render_bool_expr(node["body"])
        return f"({quant} [{vars_text}] . {body})"
    return json.dumps(node, ensure_ascii=False)


def _render_term_expr(node: dict[str, Any]) -> str:
    op = str(node.get("op", ""))
    if op == "const":
        return str(node.get("name", ""))
    if op == "var":
        return str(node.get("name", ""))
    if op == "value":
        args = ", ".join(_render_term_expr(arg) for arg in node.get("args", []))
        return f"value({node.get('func')}({args}))"
    return json.dumps(node, ensure_ascii=False)


def _is_bool_sort(range_sort: str) -> bool:
    normalized = range_sort.strip().lower()
    return normalized in {"bool", "boolsort", "bool_sort"}


def _is_numeric_sort(sort_name: str, ctx: dict[str, Any]) -> bool:
    normalized = sort_name.strip().lower()
    if (
        normalized in {"intsort", "real_sort", "realsort", "int"}
        or normalized.startswith("bitvecsort(")
    ):
        return True
    sort_kind = str(ctx.get("sort_kinds", {}).get(sort_name, "")).strip().lower()
    if (
        sort_kind in {"intsort", "real_sort", "realsort", "int"}
        or sort_kind.startswith("bitvecsort(")
    ):
        return True
    return False
