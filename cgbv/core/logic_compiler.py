from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import z3
from z3 import (
    ArraySort,
    BitVecSort,
    BoolSort,
    Const,
    DeclareSort,
    EnumSort,
    IntSort,
    RealSort,
    SortRef,
)

logger = logging.getLogger(__name__)

_CANONICAL_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "sorts",
    "functions",
    "constants",
    "variables",
    "background_constraints",
    "sentences",
    "query",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SORT_MISMATCH_RE = re.compile(
    r"Sort mismatch in function call '([^']+)' arg (\d+): expected '([^']+)', got '([^']+)'"
)


class SortManager:
    """Adapted from ProofOfThought's sort manager with minimal CGBV changes."""

    MAX_BITVEC_SIZE = 65536

    def __init__(self) -> None:
        self.sorts: dict[str, SortRef] = {}
        self.constants: dict[str, Any] = {}
        self._initialize_builtin_sorts()

    def _initialize_builtin_sorts(self) -> None:
        self.sorts.update({
            "BoolSort": BoolSort(),
            "IntSort": IntSort(),
            "RealSort": RealSort(),
        })

    @staticmethod
    def _topological_sort_sorts(sort_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dependencies: dict[str, list[str]] = {}
        for sort_def in sort_defs:
            if "name" not in sort_def:
                raise ValueError(f"Sort definition missing 'name': {sort_def}")
            name = str(sort_def["name"])
            sort_type = str(sort_def.get("type", ""))
            deps: list[str] = []
            if sort_type.startswith("ArraySort(") and sort_type.endswith(")"):
                inner = sort_type[len("ArraySort("):-1]
                deps.extend([part.strip() for part in inner.split(",") if part.strip()])
            dependencies[name] = deps

        in_degree = {
            name: len([dep for dep in deps if dep in dependencies])
            for name, deps in dependencies.items()
        }
        queue = [name for name, degree in in_degree.items() if degree == 0]
        ordered: list[str] = []

        while queue:
            current = queue.pop(0)
            ordered.append(current)
            for name, deps in dependencies.items():
                if current in deps and name not in ordered:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(ordered) != len(dependencies):
            remaining = set(dependencies) - set(ordered)
            raise ValueError(f"Circular sort dependency detected: {sorted(remaining)}")

        name_to_def = {str(item["name"]): item for item in sort_defs}
        return [name_to_def[name] for name in ordered]

    def create_sorts(self, sort_defs: list[dict[str, Any]]) -> None:
        for sort_def in self._topological_sort_sorts(sort_defs):
            name = str(sort_def["name"])
            sort_type = str(sort_def["type"])
            if sort_type == "EnumSort":
                values = list(sort_def.get("values", []))
                enum_sort, enum_consts = EnumSort(name, values)
                self.sorts[name] = enum_sort
                for value_name, const in zip(values, enum_consts, strict=False):
                    self.constants[value_name] = const
            elif sort_type.startswith("BitVecSort(") and sort_type.endswith(")"):
                size = int(sort_type[len("BitVecSort("):-1].strip())
                if size <= 0 or size > self.MAX_BITVEC_SIZE:
                    raise ValueError(f"Invalid BitVecSort size for {name}: {size}")
                self.sorts[name] = BitVecSort(size)
            elif sort_type.startswith("ArraySort(") and sort_type.endswith(")"):
                inner = sort_type[len("ArraySort("):-1]
                domain_name, range_name = [part.strip() for part in inner.split(",")]
                if domain_name not in self.sorts or range_name not in self.sorts:
                    raise ValueError(f"ArraySort {name} references undefined sorts")
                self.sorts[name] = ArraySort(self.sorts[domain_name], self.sorts[range_name])
            elif sort_type == "IntSort":
                self.sorts[name] = IntSort()
            elif sort_type == "RealSort":
                self.sorts[name] = RealSort()
            elif sort_type == "BoolSort":
                self.sorts[name] = BoolSort()
            elif sort_type == "DeclareSort":
                self.sorts[name] = DeclareSort(name)
            else:
                raise ValueError(f"Unknown sort type for {name}: {sort_type}")

    def create_functions(self, func_defs: list[dict[str, Any]]) -> dict[str, Any]:
        functions: dict[str, Any] = {}
        for func_def in func_defs:
            name = str(func_def["name"])
            domain_names = list(func_def.get("domain", []))
            range_name = str(func_def["range"])
            if range_name not in self.sorts:
                raise ValueError(f"Function '{name}' references undefined range sort '{range_name}'")
            domain = []
            for sort_name in domain_names:
                if sort_name not in self.sorts:
                    raise ValueError(f"Function '{name}' references undefined sort '{sort_name}'")
                domain.append(self.sorts[sort_name])
            functions[name] = z3.Function(name, *domain, self.sorts[range_name])
        return functions

    def create_constants(self, constants_defs: dict[str, Any]) -> None:
        for _, constants in constants_defs.items():
            sort_name = str(constants["sort"])
            if sort_name not in self.sorts:
                raise ValueError(f"Undefined sort '{sort_name}' in constants")
            members = constants.get("members", [])
            if isinstance(members, list):
                for member in members:
                    self.constants[str(member)] = Const(str(member), self.sorts[sort_name])
            elif isinstance(members, dict):
                for member in members:
                    self.constants[str(member)] = Const(str(member), self.sorts[sort_name])
            else:
                raise ValueError(f"Invalid constant members for sort '{sort_name}'")

    def create_variables(self, var_defs: list[dict[str, Any]]) -> dict[str, Any]:
        variables: dict[str, Any] = {}
        for var_def in var_defs:
            name = str(var_def["name"])
            sort_name = str(var_def["sort"])
            if sort_name not in self.sorts:
                raise ValueError(f"Variable '{name}' references undefined sort '{sort_name}'")
            variables[name] = Const(name, self.sorts[sort_name])
        return variables


@dataclass(slots=True)
class SymbolTable:
    sorts: dict[str, z3.SortRef]
    functions: dict[str, z3.FuncDeclRef]
    constants: dict[str, z3.ExprRef]
    variables: dict[str, z3.ExprRef]

    def namespace(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        data.update(self.sorts)
        data.update(self.constants)
        data.update(self.functions)
        data.update(self.variables)
        return data


@dataclass(slots=True)
class CompiledTheory:
    payload: dict[str, Any]
    symbol_table: SymbolTable
    premises: list[z3.ExprRef]
    q: z3.ExprRef
    background_constraints: list[z3.ExprRef]
    namespace: dict[str, Any]
    bound_var_names: set[str]
    raw_code: str


def compile_theory_dsl(
    payload: dict,
    premises_nl: list[str],
    conclusion_nl: str,
) -> CompiledTheory:
    if not isinstance(payload, dict):
        raise ValueError("DSL payload must be a JSON object")

    working_payload = _normalise_theory_payload(payload)
    last_error: ValueError | None = None

    for _ in range(3):
        try:
            return _compile_theory_dsl_once(working_payload, premises_nl, conclusion_nl)
        except ValueError as exc:
            last_error = exc
            harmonized = _try_harmonize_entity_sort_mismatch(working_payload, str(exc))
            if harmonized is None:
                break
            working_payload = harmonized

    if last_error is not None:
        raise last_error
    raise ValueError("DSL compile failed with unknown error")


def _compile_theory_dsl_once(
    payload: dict[str, Any],
    premises_nl: list[str],
    conclusion_nl: str,
) -> CompiledTheory:
    required = [
        "sorts",
        "functions",
        "constants",
        "variables",
        "background_constraints",
        "sentences",
        "query",
    ]
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"DSL payload missing top-level fields: {missing}")

    sentences = payload.get("sentences", [])
    if not isinstance(sentences, list):
        raise ValueError("'sentences' must be a list")
    if len(sentences) != len(premises_nl):
        raise ValueError(
            f"DSL has {len(sentences)} premise sentences, but the task has {len(premises_nl)} premises"
        )
    for idx, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            raise ValueError(f"Sentence {idx} must be an object")
        if _normalise_nl_text(str(sentence.get("nl", ""))) != _normalise_nl_text(premises_nl[idx]):
            raise ValueError(f"Sentence {idx} NL text does not match the source premise")
        if "logic" not in sentence:
            raise ValueError(f"Sentence {idx} is missing 'logic'")

    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError("'query' must be an object")
    if _normalise_nl_text(str(query.get("nl", ""))) != _normalise_nl_text(conclusion_nl):
        raise ValueError("Query NL text does not match the source conclusion")
    if "logic" not in query:
        raise ValueError("Query is missing 'logic'")

    sort_manager = SortManager()
    sort_manager.create_sorts(_ensure_list_of_dicts(payload.get("sorts"), "sorts"))
    functions = sort_manager.create_functions(_ensure_list_of_dicts(payload.get("functions"), "functions"))
    sort_manager.create_constants(_ensure_dict(payload.get("constants"), "constants"))
    variables = sort_manager.create_variables(_ensure_list_of_dicts(payload.get("variables"), "variables"))

    symbol_table = SymbolTable(
        sorts=dict(sort_manager.sorts),
        functions=functions,
        constants=dict(sort_manager.constants),
        variables=variables,
    )
    namespace = symbol_table.namespace()

    bound_var_names = set(symbol_table.variables.keys())
    premise_formulas = [
        compile_sentence_logic(sentence["logic"], symbol_table, dict(symbol_table.variables))
        for sentence in sentences
    ]
    query_formula = compile_sentence_logic(query["logic"], symbol_table, dict(symbol_table.variables))
    background_constraints = [
        compile_sentence_logic(item, symbol_table, dict(symbol_table.variables))
        for item in _ensure_list(payload.get("background_constraints"), "background_constraints")
    ]
    for formula in premise_formulas + [query_formula] + background_constraints:
        _ensure_bool_formula(formula)

    raw_code = render_compiled_theory(
        CompiledTheory(
            payload=_deep_copy_jsonable(payload),
            symbol_table=symbol_table,
            premises=premise_formulas,
            q=query_formula,
            background_constraints=background_constraints,
            namespace=namespace,
            bound_var_names=bound_var_names,
            raw_code="",
        )
    )
    return CompiledTheory(
        payload=_deep_copy_jsonable(payload),
        symbol_table=symbol_table,
        premises=premise_formulas,
        q=query_formula,
        background_constraints=background_constraints,
        namespace=namespace,
        bound_var_names=bound_var_names,
        raw_code=raw_code,
    )


def compile_sentence_logic(
    logic_obj: Any,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
) -> z3.ExprRef:
    canonical = _normalise_logic_obj(logic_obj)
    formula = _compile_bool_expr(canonical, symbol_table, dict(variable_scope))
    _ensure_bool_formula(formula)
    return formula


def parse_logic_string(source: str) -> dict[str, Any]:
    text = _strip_outer_parentheses(str(source).strip())
    if not text:
        raise ValueError("Logic string must not be empty")
    if _is_forbidden_bool_literal(text):
        raise ValueError("Boolean literals 'true/false' are not allowed in logic strings")
    quant = _parse_quantifier_head(text)
    if quant is not None:
        op, vars_list, body_text = quant
        return {"op": op, "vars": vars_list, "body": parse_logic_string(body_text)}

    compare = _split_top_level_comparison(text)
    if compare is not None:
        op, left_text, right_text = compare
        return {
            "op": op,
            "left": _parse_term_string(left_text),
            "right": _parse_term_string(right_text),
        }

    call = _parse_call_like(text)
    if call is not None:
        func_name, args = call
        op = func_name.lower()
        if op in {"and", "or"}:
            if not args:
                raise ValueError(f"Operator '{op}' requires at least one argument")
            return {"op": op, "args": [parse_logic_string(arg) for arg in args]}
        if op == "not":
            if len(args) != 1:
                raise ValueError("Operator 'not' requires exactly one argument")
            return {"op": "not", "arg": parse_logic_string(args[0])}
        if op in {"implies", "iff"}:
            if len(args) != 2:
                raise ValueError(f"Operator '{op}' requires exactly two arguments")
            return {"op": op, "left": parse_logic_string(args[0]), "right": parse_logic_string(args[1])}
        if op in {"eq", "ne"}:
            if len(args) != 2:
                raise ValueError(f"Operator '{op}' requires exactly two arguments")
            return {
                "op": op,
                "left": _parse_term_string(args[0]),
                "right": _parse_term_string(args[1]),
            }
        _validate_atom_call_string(text)
        return {"atom": _normalise_atom_string(text)}

    infix_impl = _split_top_level_keyword(text, "implies")
    if infix_impl is not None:
        left, right = infix_impl
        return {"op": "implies", "left": parse_logic_string(left), "right": parse_logic_string(right)}

    infix_iff = _split_top_level_keyword(text, "iff")
    if infix_iff is not None:
        left, right = infix_iff
        return {"op": "iff", "left": parse_logic_string(left), "right": parse_logic_string(right)}

    infix_or = _split_top_level_multi_keyword(text, "or")
    if infix_or is not None:
        if not infix_or:
            raise ValueError("Operator 'or' requires at least one argument")
        return {"op": "or", "args": [parse_logic_string(part) for part in infix_or]}

    infix_and = _split_top_level_multi_keyword(text, "and")
    if infix_and is not None:
        if not infix_and:
            raise ValueError("Operator 'and' requires at least one argument")
        return {"op": "and", "args": [parse_logic_string(part) for part in infix_and]}

    infix_not = _split_prefix_keyword(text, "not")
    if infix_not is not None:
        return {"op": "not", "arg": parse_logic_string(infix_not)}

    raise ValueError(f"Unsupported logic string syntax: {text}")


def canonicalize_logic_obj(logic_obj: Any) -> dict[str, Any]:
    return _normalise_logic_obj(logic_obj)


def canonicalize_theory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("DSL payload must be a JSON object")
    return _normalise_theory_payload(payload)


def render_logic_string(logic_obj: Any) -> str:
    canonical = _normalise_logic_obj(logic_obj)
    return _render_compact_logic_expr(canonical)


def to_compact_dsl_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = canonicalize_theory_payload(payload)

    predicates: dict[str, str] = {}
    functions: dict[str, str] = {}
    for func_def in _ensure_list_of_dicts(canonical.get("functions", []), "functions"):
        name = str(func_def["name"])
        domain = [_compact_sort_name(str(sort_name)) for sort_name in func_def.get("domain", [])]
        range_sort = _compact_sort_name(str(func_def["range"]))
        signature = _render_compact_signature(domain, range_sort)
        if range_sort == "Bool":
            predicates[name] = signature
        else:
            functions[name] = signature

    constants: dict[str, str] = {}
    for bucket in _ensure_dict(canonical.get("constants", {}), "constants").values():
        if not isinstance(bucket, dict):
            continue
        sort_name = _compact_sort_name(str(bucket.get("sort", "")))
        for member in bucket.get("members", []) or []:
            constants[str(member)] = sort_name

    return {
        "symbols": {
            "sorts": [str(sort_def["name"]) for sort_def in _ensure_list_of_dicts(canonical.get("sorts", []), "sorts")],
            "predicates": predicates,
            "functions": functions,
            "constants": constants,
        },
        "sentences": [
            {
                "nl": str(sentence.get("nl", "")),
                "logic": render_logic_string(sentence["logic"]),
            }
            for sentence in _ensure_list_of_dicts(canonical.get("sentences", []), "sentences")
        ],
        "query": {
            "nl": str(_ensure_dict(canonical.get("query"), "query").get("nl", "")),
            "logic": render_logic_string(_ensure_dict(canonical.get("query"), "query")["logic"]),
        },
        "background": [
            render_logic_string(item)
            for item in _ensure_list(canonical.get("background_constraints", []), "background_constraints")
        ],
    }


def to_compact_dsl_payload_safe(payload: Any) -> dict[str, Any]:
    """Best-effort compact DSL view for prompts.

    This helper never returns canonical heavy payload fields such as
    `background_constraints`, even when canonicalization fails.
    """
    try:
        return to_compact_dsl_payload(_ensure_dict(payload, "dsl_payload"))
    except Exception:
        return _best_effort_compact_payload(payload)


def _best_effort_compact_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "symbols": {"sorts": [], "predicates": {}, "functions": {}, "constants": {}},
            "sentences": [],
            "query": {"nl": "", "logic": ""},
            "background": [],
        }

    symbols_out = _best_effort_symbols(payload)
    sentences_out = _best_effort_sentences(payload.get("sentences"))
    query_out = _best_effort_query(payload.get("query"))
    background_out = _best_effort_background(payload)
    return {
        "symbols": symbols_out,
        "sentences": sentences_out,
        "query": query_out,
        "background": background_out,
    }


def _best_effort_symbols(payload: dict[str, Any]) -> dict[str, Any]:
    symbols = payload.get("symbols")
    if isinstance(symbols, dict):
        raw_sorts = symbols.get("sorts", [])
        sorts: list[str] = []
        if isinstance(raw_sorts, list):
            for item in raw_sorts:
                if isinstance(item, str):
                    name = item.strip()
                    if name:
                        sorts.append(name)
                elif isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if name:
                        sorts.append(name)
        predicates = symbols.get("predicates", {})
        functions = symbols.get("functions", {})
        constants = symbols.get("constants", {})
        return {
            "sorts": sorts,
            "predicates": _string_map(predicates),
            "functions": _string_map(functions),
            "constants": _string_map(constants),
        }

    predicates: dict[str, str] = {}
    functions: dict[str, str] = {}
    for func_def in payload.get("functions", []) if isinstance(payload.get("functions"), list) else []:
        if not isinstance(func_def, dict):
            continue
        name = str(func_def.get("name", "")).strip()
        if not name:
            continue
        domain_raw = func_def.get("domain", [])
        domain: list[str] = []
        if isinstance(domain_raw, list):
            domain = [_compact_sort_name(str(part).strip()) for part in domain_raw if str(part).strip()]
        range_sort = _compact_sort_name(str(func_def.get("range", "")).strip())
        if not range_sort:
            continue
        sig = _render_compact_signature(domain, range_sort)
        if range_sort == "Bool":
            predicates[name] = sig
        else:
            functions[name] = sig

    constants: dict[str, str] = {}
    constants_obj = payload.get("constants", {})
    if isinstance(constants_obj, dict):
        for bucket in constants_obj.values():
            if not isinstance(bucket, dict):
                continue
            sort_name = _compact_sort_name(str(bucket.get("sort", "")).strip())
            if not sort_name:
                continue
            members = bucket.get("members", [])
            if isinstance(members, list):
                for member in members:
                    const_name = str(member).strip()
                    if const_name:
                        constants[const_name] = sort_name

    sorts: list[str] = []
    raw_sorts = payload.get("sorts", [])
    if isinstance(raw_sorts, list):
        for item in raw_sorts:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    sorts.append(name)
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    sorts.append(name)

    return {
        "sorts": sorts,
        "predicates": predicates,
        "functions": functions,
        "constants": constants,
    }


def _best_effort_sentences(raw_sentences: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sentences, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_sentences:
        if not isinstance(item, dict):
            continue
        out.append({
            "nl": str(item.get("nl", "")),
            "logic": _best_effort_logic_string(item.get("logic")),
        })
    return out


def _best_effort_query(raw_query: Any) -> dict[str, Any]:
    if not isinstance(raw_query, dict):
        return {"nl": "", "logic": ""}
    return {
        "nl": str(raw_query.get("nl", "")),
        "logic": _best_effort_logic_string(raw_query.get("logic")),
    }


def _best_effort_background(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("background")
    if raw is None:
        raw = payload.get("background_constraints", [])
    if not isinstance(raw, list):
        return []
    return [_best_effort_logic_string(item) for item in raw]


def _best_effort_logic_string(logic_obj: Any) -> str:
    if isinstance(logic_obj, str):
        return logic_obj.strip()
    try:
        return render_logic_string(logic_obj)
    except Exception:
        try:
            return json.dumps(logic_obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(logic_obj)


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = str(item).strip()
    return out


def render_compiled_theory(compiled: CompiledTheory) -> str:
    payload = compiled.payload
    lines = ["from z3 import *", "", "ctx = Context()", ""]

    sort_defs = _ensure_list_of_dicts(payload.get("sorts", []), "sorts")
    for sort_def in sort_defs:
        name = str(sort_def["name"])
        sort_type = str(sort_def["type"])
        if sort_type == "DeclareSort":
            lines.append(f"{name} = DeclareSort({name!r}, ctx)")
        elif sort_type == "EnumSort":
            values = [repr(str(v)) for v in sort_def.get("values", [])]
            lines.append(
                f"{name}, ({', '.join(str(v) for v in sort_def.get('values', []))}) = "
                f"EnumSort({name!r}, [{', '.join(values)}], ctx)"
            )
        elif sort_type in {"BoolSort", "IntSort", "RealSort"}:
            lines.append(f"{name} = {_render_sort_expr(sort_type, ctx_name='ctx')}")
        elif sort_type.startswith("ArraySort(") or sort_type.startswith("BitVecSort("):
            lines.append(f"{name} = {_render_sort_expr(sort_type, ctx_name='ctx')}")
        else:
            lines.append(f"# unsupported sort declaration preserved for reference: {name} ({sort_type})")
    if sort_defs:
        lines.append("")

    for constants in _ensure_dict(payload.get("constants", {}), "constants").values():
        if not isinstance(constants, dict):
            continue
        sort_expr = _render_sort_expr(str(constants["sort"]), ctx_name="ctx")
        members = constants.get("members", [])
        if isinstance(members, list):
            for member in members:
                lines.append(f"{member} = Const({member!r}, {sort_expr})")
        elif isinstance(members, dict):
            for member in members:
                lines.append(f"{member} = Const({member!r}, {sort_expr})")
    if payload.get("constants"):
        lines.append("")

    for func_def in _ensure_list_of_dicts(payload.get("functions", []), "functions"):
        name = str(func_def["name"])
        domain_exprs = [_render_sort_expr(str(sort_name), ctx_name="ctx") for sort_name in func_def.get("domain", [])]
        range_expr = _render_sort_expr(str(func_def["range"]), ctx_name="ctx")
        args = ", ".join([repr(name), *domain_exprs, range_expr])
        lines.append(f"{name} = Function({args})")
    if payload.get("functions"):
        lines.append("")

    declared_vars = _collect_variable_declarations(payload)
    for name, sort_name in declared_vars.items():
        sort_expr = _render_sort_expr(str(sort_name), ctx_name="ctx")
        lines.append(f"{name} = Const({name!r}, {sort_expr})")
    if declared_vars:
        lines.append("")

    lines.append("background_constraints = [")
    for logic_obj in _ensure_list(payload.get("background_constraints", []), "background_constraints"):
        lines.append(f"    {_render_logic_obj(logic_obj)},")
    lines.append("]")
    lines.append("")
    lines.append("premises = [")
    for sentence in _ensure_list_of_dicts(payload.get("sentences", []), "sentences"):
        lines.append(f"    {_render_logic_obj(_ensure_dict(sentence.get('logic'), 'sentence.logic'))},")
    lines.append("]")
    query = _ensure_dict(payload.get("query"), "query")
    lines.append(f"q = {_render_logic_obj(_ensure_dict(query.get('logic'), 'query.logic'))}")
    return "\n".join(lines)


def _render_sort_expr(sort_name: str, ctx_name: str | None = None) -> str:
    if sort_name in {"BoolSort", "IntSort", "RealSort"}:
        if ctx_name is None:
            return f"{sort_name}()"
        return f"{sort_name}({ctx_name})"
    if sort_name.startswith("BitVecSort(") and sort_name.endswith(")"):
        inner = sort_name[len("BitVecSort("):-1].strip()
        if ctx_name is None:
            return f"BitVecSort({inner})"
        return f"BitVecSort({inner}, {ctx_name})"
    if sort_name.startswith("ArraySort(") and sort_name.endswith(")"):
        inner = sort_name[len("ArraySort("):-1]
        domain_name, range_name = [part.strip() for part in inner.split(",", maxsplit=1)]
        return (
            f"ArraySort("
            f"{_render_sort_expr(domain_name, ctx_name=ctx_name)}, "
            f"{_render_sort_expr(range_name, ctx_name=ctx_name)})"
        )
    return sort_name


def _collect_variable_declarations(payload: dict[str, Any]) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for var_def in _ensure_list_of_dicts(payload.get("variables", []), "variables"):
        name = str(var_def["name"])
        declarations[name] = str(var_def["sort"])

    logic_objects: list[dict[str, Any]] = []
    logic_objects.extend(_ensure_list(payload.get("background_constraints", []), "background_constraints"))
    for sentence in _ensure_list_of_dicts(payload.get("sentences", []), "sentences"):
        logic_objects.append(_ensure_dict(sentence.get("logic"), "sentence.logic"))
    query = _ensure_dict(payload.get("query"), "query")
    logic_objects.append(_ensure_dict(query.get("logic"), "query.logic"))

    for logic_obj in logic_objects:
        for var_name, sort_name in _collect_quantifier_variables(logic_obj).items():
            existing_sort = declarations.get(var_name)
            if existing_sort is not None and existing_sort != sort_name:
                raise ValueError(f"Variable '{var_name}' has inconsistent sort usage")
            declarations.setdefault(var_name, sort_name)
    return declarations


def _collect_quantifier_variables(logic_obj: dict[str, Any]) -> dict[str, str]:
    declarations: dict[str, str] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("all", "any", "forall", "exists"):
                if key in node:
                    for var_def in _ensure_list(node[key], key):
                        if not isinstance(var_def, dict):
                            raise ValueError("Quantifier variables must be objects with name/sort")
                        name = str(var_def.get("name", ""))
                        sort_name = str(var_def.get("sort", ""))
                        if not name or not sort_name:
                            raise ValueError("Quantifier variables must declare both name and sort")
                        declarations[name] = sort_name
            op = str(node.get("op", ""))
            if op in {"all", "any", "forall", "exists"}:
                raw_vars = node.get("vars", node.get("args"))
                if isinstance(raw_vars, list):
                    for var_def in raw_vars:
                        if not isinstance(var_def, dict) or "name" not in var_def or "sort" not in var_def:
                            continue
                        name = str(var_def.get("name", ""))
                        sort_name = str(var_def.get("sort", ""))
                        if not name or not sort_name:
                            raise ValueError("Quantifier variables must declare both name and sort")
                        declarations[name] = sort_name
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(logic_obj)
    return declarations


def _render_logic_obj(logic_obj: Any) -> str:
    return _render_bool_expr(_normalise_logic_obj(logic_obj))


def _render_compact_logic_expr(expr_obj: dict[str, Any]) -> str:
    expr_obj = _normalise_bool_expr(expr_obj)
    atom = expr_obj.get("atom")
    if isinstance(atom, str):
        return atom

    op = str(expr_obj.get("op", ""))
    if op in {"and", "or"}:
        args = _ensure_list(expr_obj.get("args"), f"{op}.args")
        return f"{op}({', '.join(_render_compact_logic_expr(_ensure_dict(arg, f'{op}.args')) for arg in args)})"
    if op == "not":
        arg = _ensure_dict(expr_obj.get("arg"), "not.arg")
        return f"not({_render_compact_logic_expr(arg)})"
    if op in {"implies", "iff"}:
        left = _render_compact_logic_expr(_ensure_dict(expr_obj.get("left"), f"{op}.left"))
        right = _render_compact_logic_expr(_ensure_dict(expr_obj.get("right"), f"{op}.right"))
        return f"{op}({left}, {right})"
    if op in {"all", "any"}:
        vars_list = _ensure_list_of_dicts(expr_obj.get("vars"), f"{op}.vars")
        if not vars_list:
            raise ValueError(f"Operator '{op}' requires at least one variable")
        prefix = "forall" if op == "all" else "exists"
        vars_blob = ", ".join(f"{var['name']}:{var['sort']}" for var in vars_list)
        body = _render_compact_logic_expr(_ensure_dict(expr_obj.get("body"), f"{op}.body"))
        return f"{prefix} {vars_blob}. {body}"
    if op in {"eq", "ne"}:
        left = _render_compact_term(expr_obj.get("left"))
        right = _render_compact_term(expr_obj.get("right"))
        symbol = "==" if op == "eq" else "!="
        return f"{left}{symbol}{right}"
    if op == "const":
        raise ValueError("Boolean literal constants are not allowed in compact logic DSL")

    raise ValueError(f"Unsupported boolean expression object: {expr_obj}")


def _render_compact_term(term_obj: Any) -> str:
    if isinstance(term_obj, str):
        return term_obj
    raise ValueError(f"Unsupported term expression object: {term_obj}")


def _render_compact_signature(domain: list[str], range_sort: str) -> str:
    if not domain:
        return f"->{range_sort}"
    return f"{','.join(domain)}->{range_sort}"


def _compact_sort_name(sort_name: str) -> str:
    alias_map = {
        "BoolSort": "Bool",
        "IntSort": "Int",
        "RealSort": "Real",
    }
    return alias_map.get(sort_name, sort_name)


def _render_bool_expr(expr_obj: dict[str, Any]) -> str:
    expr_obj = _normalise_bool_expr(expr_obj)
    atom = expr_obj.get("atom")
    if isinstance(atom, str):
        return _render_atom(atom)

    op = str(expr_obj.get("op", ""))
    if op in {"and", "or"}:
        args = _ensure_list(expr_obj.get("args"), f"{op}.args")
        if not args:
            raise ValueError(f"Operator '{op}' requires at least one argument")
        rendered_args = ", ".join(_render_bool_expr(_ensure_dict(arg, f"{op}.args")) for arg in args)
        fn_name = "And" if op == "and" else "Or"
        return f"{fn_name}({rendered_args})"
    if op == "not":
        arg = _ensure_dict(expr_obj.get("arg"), "not.arg")
        return f"Not({_render_bool_expr(arg)})"
    if op in {"implies", "iff"}:
        left = _render_bool_expr(_ensure_dict(expr_obj.get("left"), f"{op}.left"))
        right = _render_bool_expr(_ensure_dict(expr_obj.get("right"), f"{op}.right"))
        if op == "implies":
            return f"Implies({left}, {right})"
        return f"({left} == {right})"
    if op == "const":
        return "BoolVal(True)" if bool(expr_obj.get("value")) else "BoolVal(False)"
    if op in {"all", "any"}:
        quantifier_vars = _ensure_list_of_dicts(expr_obj.get("vars"), f"{op}.vars")
        if not quantifier_vars:
            raise ValueError(f"Operator '{op}' requires at least one variable")
        body = _render_bool_expr(_ensure_dict(expr_obj.get("body"), f"{op}.body"))
        names = ", ".join(str(var_def["name"]) for var_def in quantifier_vars)
        fn_name = "ForAll" if op == "all" else "Exists"
        return f"{fn_name}([{names}], {body})"
    if op in {"eq", "ne"}:
        left = _render_term_expr(expr_obj.get("left"))
        right = _render_term_expr(expr_obj.get("right"))
        symbol = "==" if op == "eq" else "!="
        return f"({left} {symbol} {right})"

    raise ValueError(f"Unsupported boolean expression object: {expr_obj}")


def extract_logic_predicates(logic_obj: Any) -> set[str]:
    names: set[str] = set()

    def _collect_call_heads(expr_node: ast.AST) -> None:
        for sub in ast.walk(expr_node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                names.add(sub.func.id)

    def _collect_term_heads(term_obj: Any) -> None:
        if not isinstance(term_obj, str):
            return
        try:
            parsed = ast.parse(term_obj, mode="eval").body
        except SyntaxError:
            return
        _collect_call_heads(parsed)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            atom = node.get("atom")
            if isinstance(atom, str):
                try:
                    parsed = ast.parse(atom, mode="eval").body
                except SyntaxError:
                    parsed = None
                if parsed is not None:
                    _collect_call_heads(parsed)
            op = str(node.get("op", ""))
            if op in {"eq", "ne"}:
                _collect_term_heads(node.get("left"))
                _collect_term_heads(node.get("right"))
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    try:
        canonical = _normalise_logic_obj(logic_obj)
    except ValueError:
        return names
    _walk(canonical)
    return names


def _compile_bool_expr(
    expr_obj: Any,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
) -> z3.ExprRef:
    expr_obj = _normalise_bool_expr(expr_obj)
    if not isinstance(expr_obj, dict):
        raise ValueError("Boolean expression objects must be JSON objects")

    atom = expr_obj.get("atom")
    if isinstance(atom, str):
        return _compile_atom(atom, symbol_table, variable_scope)

    op = str(expr_obj.get("op", ""))
    if op in {"and", "or"}:
        args = _ensure_list(expr_obj.get("args"), f"{op}.args")
        if not args:
            raise ValueError(f"Operator '{op}' requires at least one argument")
        compiled_args = [_compile_bool_expr(arg, symbol_table, variable_scope) for arg in args]
        return z3.And(*compiled_args) if op == "and" else z3.Or(*compiled_args)
    if op == "not":
        arg = _ensure_dict(expr_obj.get("arg"), "not.arg")
        return z3.Not(_compile_bool_expr(arg, symbol_table, variable_scope))
    if op in {"implies", "iff"}:
        left = _compile_bool_expr(_ensure_dict(expr_obj.get("left"), f"{op}.left"), symbol_table, variable_scope)
        right = _compile_bool_expr(_ensure_dict(expr_obj.get("right"), f"{op}.right"), symbol_table, variable_scope)
        if op == "implies":
            return z3.Implies(left, right)
        return left == right
    if op == "const":
        return z3.BoolVal(bool(expr_obj.get("value")))
    if op in {"all", "any"}:
        quantifier_vars = _ensure_list_of_dicts(expr_obj.get("vars"), f"{op}.vars")
        if not quantifier_vars:
            raise ValueError(f"Operator '{op}' requires at least one variable")
        local_scope = dict(variable_scope)
        bound_vars: list[z3.ExprRef] = []
        seen_names: set[str] = set()
        for var_def in quantifier_vars:
            name = str(var_def.get("name", ""))
            sort_name = str(var_def.get("sort", ""))
            if not name or not sort_name:
                raise ValueError("Quantifier variables must declare both name and sort")
            if name in seen_names:
                raise ValueError(f"Quantifier '{op}' has duplicate variable '{name}'")
            seen_names.add(name)
            if sort_name not in symbol_table.sorts:
                raise ValueError(f"Unknown quantifier sort '{sort_name}'")
            var_ref = z3.Const(name, symbol_table.sorts[sort_name])
            local_scope[name] = var_ref
            bound_vars.append(var_ref)
        body = _compile_bool_expr(_ensure_dict(expr_obj.get("body"), f"{op}.body"), symbol_table, local_scope)
        return z3.ForAll(bound_vars, body) if op == "all" else z3.Exists(bound_vars, body)
    if op in {"eq", "ne"}:
        left = _compile_term_expr(expr_obj.get("left"), symbol_table, variable_scope)
        right = _compile_term_expr(expr_obj.get("right"), symbol_table, variable_scope)
        if not left.sort().eq(right.sort()):
            raise ValueError(
                f"Sort mismatch in operator '{op}': left is '{left.sort()}', right is '{right.sort()}'"
            )
        return left == right if op == "eq" else left != right

    raise ValueError(f"Unsupported boolean expression object: {expr_obj}")


def _compile_atom(
    atom: str,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
) -> z3.ExprRef:
    atom = _normalise_atom_string(atom)
    try:
        parsed = ast.parse(atom, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid atom syntax: {atom}") from exc

    if isinstance(parsed, ast.Compare):
        return _compile_compare(parsed, symbol_table, variable_scope, atom)
    if not isinstance(parsed, ast.Call):
        raise ValueError(f"Atom must be a function call, got: {atom}")
    result = _compile_function_call(parsed, symbol_table, variable_scope, atom)
    _ensure_bool_formula(result)
    return result


def _normalise_theory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload_copy = _deep_copy_jsonable(payload)
    payload_copy = _expand_compact_payload(payload_copy)
    raw_aliases = payload_copy.pop("__constant_aliases__", {})
    constant_aliases = raw_aliases if isinstance(raw_aliases, dict) else {}
    if isinstance(payload_copy.get("constants"), dict):
        canonical_constants, canonical_aliases = _sanitize_canonical_constants(
            _ensure_dict(payload_copy.get("constants"), "constants")
        )
        payload_copy["constants"] = canonical_constants
        constant_aliases = {**canonical_aliases, **constant_aliases}

    background_raw = payload_copy.get("background_constraints", payload_copy.get("background", []))
    background_constraints = [
        _normalise_logic_obj(_rewrite_logic_with_aliases(item, constant_aliases))
        for item in _ensure_list(background_raw, "background_constraints")
    ]

    normalised_sentences: list[dict[str, Any]] = []
    for idx, sentence in enumerate(_ensure_list_of_dicts(payload_copy.get("sentences", []), "sentences")):
        if "logic" not in sentence:
            raise ValueError(f"Sentence {idx} is missing 'logic'")
        normalised_sentences.append({
            "nl": str(sentence.get("nl", "")),
            "logic": _normalise_logic_obj(_rewrite_logic_with_aliases(sentence["logic"], constant_aliases)),
        })

    query = _ensure_dict(payload_copy.get("query"), "query")
    if "logic" not in query:
        raise ValueError("Query is missing 'logic'")
    normalised_query = {
        "nl": str(query.get("nl", "")),
        "logic": _normalise_logic_obj(_rewrite_logic_with_aliases(query["logic"], constant_aliases)),
    }

    canonical_payload = {
        "sorts": _ensure_list_of_dicts(payload_copy.get("sorts", []), "sorts"),
        "functions": _ensure_list_of_dicts(payload_copy.get("functions", []), "functions"),
        "constants": _ensure_dict(payload_copy.get("constants", {}), "constants"),
        "variables": _normalise_variables(payload_copy.get("variables", [])),
        "background_constraints": background_constraints,
        "sentences": normalised_sentences,
        "query": normalised_query,
    }
    return canonical_payload


def _expand_compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "symbols" not in payload:
        return payload

    symbols = _ensure_dict(payload.get("symbols"), "symbols")
    expanded = dict(payload)
    expanded["sorts"] = _normalise_compact_sorts(symbols.get("sorts", []))
    expanded["functions"] = _normalise_compact_functions(
        symbols.get("predicates", {}),
        symbols.get("functions", {}),
    )
    expanded_constants, const_aliases = _normalise_compact_constants(symbols.get("constants", {}))
    expanded["constants"] = expanded_constants
    expanded["variables"] = _normalise_variables(payload.get("variables", []))
    expanded["sorts"] = _inject_missing_compact_sorts(
        expanded["sorts"],
        expanded["functions"],
        expanded["constants"],
        expanded["variables"],
    )
    if "background_constraints" not in expanded:
        expanded["background_constraints"] = payload.get("background", [])
    if const_aliases:
        expanded["__constant_aliases__"] = const_aliases
    return expanded


def _normalise_compact_sorts(raw_sorts: Any) -> list[dict[str, Any]]:
    sorts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(_ensure_list(raw_sorts, "symbols.sorts")):
        if isinstance(item, str):
            name = item.strip()
            if not name:
                raise ValueError(f"symbols.sorts[{idx}] must not be empty")
            if name in {"Bool", "BoolSort", "Int", "IntSort", "Real", "RealSort"}:
                continue
            if name not in seen:
                sorts.append({"name": name, "type": "DeclareSort"})
                seen.add(name)
            continue
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"symbols.sorts[{idx}] missing sort name")
            if name in seen:
                continue
            sort_type = str(item.get("type", "DeclareSort")).strip() or "DeclareSort"
            sort_decl: dict[str, Any] = {"name": name, "type": sort_type}
            if "values" in item:
                sort_decl["values"] = list(item.get("values", []))
            sorts.append(sort_decl)
            seen.add(name)
            continue
        raise ValueError(f"symbols.sorts[{idx}] must be string or object")
    return sorts


def _normalise_compact_functions(
    predicates: Any,
    functions: Any,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for name, signature in _ensure_dict(predicates, "symbols.predicates").items():
        func_name = str(name).strip()
        if not _IDENTIFIER_RE.fullmatch(func_name):
            raise ValueError(f"Invalid predicate name '{func_name}'")
        decl = _signature_to_function_decl(func_name, signature, force_bool_range=True)
        if func_name in seen:
            raise ValueError(f"Duplicate symbol '{func_name}' in symbols")
        merged.append(decl)
        seen.add(func_name)

    for name, signature in _ensure_dict(functions, "symbols.functions").items():
        func_name = str(name).strip()
        if not _IDENTIFIER_RE.fullmatch(func_name):
            raise ValueError(f"Invalid function name '{func_name}'")
        decl = _signature_to_function_decl(func_name, signature, force_bool_range=False)
        if func_name in seen:
            raise ValueError(f"Duplicate symbol '{func_name}' in symbols")
        merged.append(decl)
        seen.add(func_name)

    return merged


def _signature_to_function_decl(
    func_name: str,
    signature: Any,
    *,
    force_bool_range: bool,
) -> dict[str, Any]:
    sig = str(signature).strip()
    if "->" not in sig:
        raise ValueError(f"Signature for '{func_name}' must use 'domain->range' syntax")
    domain_raw, range_raw = sig.rsplit("->", maxsplit=1)
    domain = (
        [_canonical_sort_name(part.strip()) for part in domain_raw.split(",") if part.strip()]
        if domain_raw.strip()
        else []
    )
    range_sort = _canonical_sort_name(range_raw.strip())
    if force_bool_range and range_sort != "BoolSort":
        raise ValueError(f"Predicate '{func_name}' must return Bool")
    return {"name": func_name, "domain": domain, "range": range_sort}


def _normalise_compact_constants(raw_constants: Any) -> tuple[dict[str, Any], dict[str, str]]:
    constants_obj = _ensure_dict(raw_constants, "symbols.constants")
    if not constants_obj:
        return {}, {}

    grouped: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    used_constant_names: set[str] = set()
    for name, sort_info in constants_obj.items():
        raw_const_name = str(name).strip()
        if not raw_const_name:
            raise ValueError("Constant name must not be empty")
        const_name = _allocate_identifier(raw_const_name, prefix="c", used=used_constant_names)
        if const_name != raw_const_name:
            aliases[raw_const_name] = const_name

        if isinstance(sort_info, str):
            sort_name = _canonical_sort_name(sort_info.strip())
            bucket = grouped.setdefault(sort_name, {"sort": sort_name, "members": []})
            bucket["members"].append(const_name)
            continue

        if isinstance(sort_info, dict):
            sort_name = str(sort_info.get("sort", "")).strip()
            if not sort_name:
                raise ValueError(f"Constant '{const_name}' missing sort")
            canonical_sort = _canonical_sort_name(sort_name)
            bucket = grouped.setdefault(canonical_sort, {"sort": canonical_sort, "members": []})
            bucket["members"].append(const_name)
            continue

        raise ValueError(f"Constant '{const_name}' must map to a sort string")

    for bucket in grouped.values():
        seen: set[str] = set()
        deduped: list[str] = []
        for member in bucket.get("members", []):
            if member in seen:
                continue
            seen.add(member)
            deduped.append(member)
        bucket["members"] = deduped

    return grouped, aliases


def _sanitize_canonical_constants(
    raw_constants: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    grouped: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    used_constant_names: set[str] = set()

    for bucket_name, bucket in raw_constants.items():
        if not isinstance(bucket, dict):
            continue
        sort_name = _canonical_sort_name(str(bucket.get("sort", "")).strip())
        if not sort_name:
            raise ValueError(f"Constant bucket '{bucket_name}' missing sort")
        members = bucket.get("members", [])
        if isinstance(members, dict):
            member_iterable = list(members.keys())
        elif isinstance(members, list):
            member_iterable = members
        else:
            continue
        out_bucket = grouped.setdefault(sort_name, {"sort": sort_name, "members": []})
        for member in member_iterable:
            raw_name = str(member).strip()
            if not raw_name:
                continue
            norm_name = _allocate_identifier(raw_name, prefix="c", used=used_constant_names)
            if norm_name != raw_name:
                aliases[raw_name] = norm_name
            out_bucket["members"].append(norm_name)

    for bucket in grouped.values():
        seen: set[str] = set()
        deduped: list[str] = []
        for member in bucket["members"]:
            if member in seen:
                continue
            seen.add(member)
            deduped.append(member)
        bucket["members"] = deduped

    return grouped, aliases


def _normalise_variables(raw_variables: Any) -> list[dict[str, Any]]:
    if isinstance(raw_variables, dict):
        out: list[dict[str, Any]] = []
        for name, sort in raw_variables.items():
            var_name = str(name).strip()
            sort_name = _canonical_sort_name(str(sort).strip())
            if not var_name or not sort_name:
                raise ValueError("Variable map entries must provide name and sort")
            out.append({"name": var_name, "sort": sort_name})
        return out

    variables = _ensure_list(raw_variables, "variables")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(variables):
        if not isinstance(item, dict):
            raise ValueError(f"'variables[{idx}]' must be an object")
        name = str(item.get("name", "")).strip()
        sort_name = _canonical_sort_name(str(item.get("sort", "")).strip())
        if not name or not sort_name:
            raise ValueError("Variable declarations must include name and sort")
        out.append({"name": name, "sort": sort_name})
    return out


def _canonical_sort_name(sort_name: str) -> str:
    alias_map = {
        "Bool": "BoolSort",
        "Int": "IntSort",
        "Real": "RealSort",
    }
    return alias_map.get(sort_name, sort_name)


def _inject_missing_compact_sorts(
    sorts: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    constants: dict[str, Any],
    variables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    builtin = {"BoolSort", "IntSort", "RealSort", "Bool", "Int", "Real"}
    seen = {str(item.get("name", "")).strip() for item in sorts}
    out = list(sorts)

    def _add(sort_name: str) -> None:
        candidate = _canonical_sort_name(sort_name.strip())
        if not candidate or candidate in builtin or candidate in seen:
            return
        seen.add(candidate)
        out.append({"name": candidate, "type": "DeclareSort"})

    for func in functions:
        for sort_name in func.get("domain", []) or []:
            _add(str(sort_name))
        _add(str(func.get("range", "")))
    for bucket in constants.values():
        if isinstance(bucket, dict):
            _add(str(bucket.get("sort", "")))
    for var in variables:
        _add(str(var.get("sort", "")))
    return out


def _normalise_identifier(name: str, *, prefix: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        candidate = prefix
    if candidate[0].isdigit():
        candidate = f"{prefix}_{candidate}"
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(f"Invalid identifier '{name}'")
    return candidate


def _allocate_identifier(name: str, *, prefix: str, used: set[str]) -> str:
    base = _normalise_identifier(name, prefix=prefix)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _rewrite_logic_with_aliases(logic_obj: Any, aliases: dict[str, str]) -> Any:
    if not aliases:
        return logic_obj
    if isinstance(logic_obj, str):
        return _replace_symbol_aliases_in_logic_string(logic_obj, aliases)
    if isinstance(logic_obj, list):
        return [_rewrite_logic_with_aliases(item, aliases) for item in logic_obj]
    if isinstance(logic_obj, dict):
        return {key: _rewrite_logic_with_aliases(value, aliases) for key, value in logic_obj.items()}
    return logic_obj


def _replace_symbol_aliases_in_logic_string(text: str, aliases: dict[str, str]) -> str:
    out = str(text)
    for source, target in sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
        if source == target:
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
        out = re.sub(pattern, target, out)
    return out


def _try_harmonize_entity_sort_mismatch(
    payload: dict[str, Any],
    error_message: str,
) -> dict[str, Any] | None:
    match = _SORT_MISMATCH_RE.search(error_message)
    if match is None:
        return None

    expected_sort = match.group(3).strip()
    got_sort = match.group(4).strip()
    if expected_sort != "Entity" or got_sort == expected_sort:
        return None

    sort_defs = _ensure_list_of_dicts(payload.get("sorts", []), "sorts")
    sort_names = {str(item.get("name", "")).strip() for item in sort_defs}
    if expected_sort not in sort_names or got_sort not in sort_names:
        return None
    function_defs = _ensure_list_of_dicts(payload.get("functions", []), "functions")
    for func_def in function_defs:
        domain = [str(part).strip() for part in func_def.get("domain", []) or []]
        range_sort = str(func_def.get("range", "")).strip()
        if got_sort in domain or got_sort == range_sort:
            # got_sort is part of declared function signatures; collapsing it to
            # Entity would damage intended many-sort typing.
            return None

    remapped = _deep_copy_jsonable(payload)

    for bucket in _ensure_dict(remapped.get("constants", {}), "constants").values():
        if not isinstance(bucket, dict):
            continue
        if str(bucket.get("sort", "")).strip() == got_sort:
            bucket["sort"] = expected_sort

    for var_def in _ensure_list_of_dicts(remapped.get("variables", []), "variables"):
        if str(var_def.get("sort", "")).strip() == got_sort:
            var_def["sort"] = expected_sort

    for sentence in _ensure_list_of_dicts(remapped.get("sentences", []), "sentences"):
        if "logic" in sentence:
            sentence["logic"] = _remap_quantifier_sorts(sentence["logic"], got_sort, expected_sort)
    query = _ensure_dict(remapped.get("query", {}), "query")
    if "logic" in query:
        query["logic"] = _remap_quantifier_sorts(query["logic"], got_sort, expected_sort)
    remapped["background_constraints"] = [
        _remap_quantifier_sorts(item, got_sort, expected_sort)
        for item in _ensure_list(remapped.get("background_constraints", []), "background_constraints")
    ]

    logger.warning(
        "Auto-harmonized sort mismatch by remapping quantifier/constant/variable sort '%s' -> '%s'",
        got_sort,
        expected_sort,
    )
    return remapped


def _remap_quantifier_sorts(logic_obj: Any, source_sort: str, target_sort: str) -> Any:
    if isinstance(logic_obj, str):
        return logic_obj
    if isinstance(logic_obj, list):
        return [_remap_quantifier_sorts(item, source_sort, target_sort) for item in logic_obj]
    if isinstance(logic_obj, dict):
        out: dict[str, Any] = {}
        for key, value in logic_obj.items():
            if key in {"all", "any", "forall", "exists"} and isinstance(value, list):
                remapped_vars: list[Any] = []
                for item in value:
                    if isinstance(item, dict):
                        var_obj = dict(item)
                        if str(var_obj.get("sort", "")).strip() == source_sort:
                            var_obj["sort"] = target_sort
                        remapped_vars.append(var_obj)
                    else:
                        remapped_vars.append(item)
                out[key] = remapped_vars
                continue
            out[key] = _remap_quantifier_sorts(value, source_sort, target_sort)

        op = str(out.get("op", "")).strip().lower()
        if op in {"all", "any", "forall", "exists"}:
            vars_blob = out.get("vars")
            if isinstance(vars_blob, list):
                remapped_vars = []
                for item in vars_blob:
                    if isinstance(item, dict):
                        var_obj = dict(item)
                        if str(var_obj.get("sort", "")).strip() == source_sort:
                            var_obj["sort"] = target_sort
                        remapped_vars.append(var_obj)
                    else:
                        remapped_vars.append(item)
                out["vars"] = remapped_vars
        return out
    return logic_obj


def _normalise_logic_obj(logic_obj: Any) -> dict[str, Any]:
    if isinstance(logic_obj, str):
        return _normalise_bool_expr(parse_logic_string(logic_obj))
    if not isinstance(logic_obj, dict):
        raise ValueError("Logic object must be a JSON object")
    if set(logic_obj.keys()) == {"expr"}:
        return _normalise_bool_expr(logic_obj["expr"])

    if "kind" in logic_obj:
        kind = str(logic_obj.get("kind", "")).strip().lower()
        if kind == "constraint":
            if "constraint" in logic_obj:
                body_expr = _normalise_bool_expr(logic_obj["constraint"])
            elif "expr" in logic_obj:
                body_expr = _normalise_bool_expr(logic_obj["expr"])
            else:
                raise ValueError("Constraint logic must include 'constraint' or 'expr'")
            return _wrap_legacy_quantifier(body_expr, logic_obj)
        if kind == "rule":
            if "implies" not in logic_obj:
                raise ValueError("Rule logic must include an 'implies' object")
            return _wrap_legacy_quantifier(_normalise_legacy_implies_obj(logic_obj["implies"]), logic_obj)
        raise ValueError(f"Unsupported logic kind '{kind}'")

    if "implies" in logic_obj and "op" not in logic_obj:
        return _wrap_legacy_quantifier(_normalise_legacy_implies_obj(logic_obj["implies"]), logic_obj)

    if "constraint" in logic_obj and "op" not in logic_obj and "atom" not in logic_obj:
        return _wrap_legacy_quantifier(_normalise_bool_expr(logic_obj["constraint"]), logic_obj)

    return _normalise_bool_expr(logic_obj)


def _extract_legacy_quantifier(logic_obj: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    has_universal = "all" in logic_obj or "forall" in logic_obj
    has_existential = "any" in logic_obj or "exists" in logic_obj
    if has_universal and has_existential:
        raise ValueError("A single logic object cannot contain both universal and existential quantifiers")
    if not has_universal and not has_existential:
        return None, []

    if has_universal:
        quant_op = "all"
        raw_vars = logic_obj.get("all", logic_obj.get("forall"))
        field_name = "all.vars"
    else:
        quant_op = "any"
        raw_vars = logic_obj.get("any", logic_obj.get("exists"))
        field_name = "any.vars"

    vars_list = _ensure_list_of_dicts(raw_vars, field_name)
    normalised_vars: list[dict[str, Any]] = []
    for var_def in vars_list:
        name = str(var_def.get("name", "")).strip()
        sort_name = _canonical_sort_name(str(var_def.get("sort", "")).strip())
        if not name or not sort_name:
            raise ValueError("Quantifier variables must declare both name and sort")
        normalised_vars.append({"name": name, "sort": sort_name})
    return quant_op, normalised_vars


def _wrap_legacy_quantifier(
    body_expr: dict[str, Any],
    logic_obj: dict[str, Any],
) -> dict[str, Any]:
    quant_op, vars_list = _extract_legacy_quantifier(logic_obj)
    if quant_op is None:
        return body_expr
    return {"op": quant_op, "vars": vars_list, "body": body_expr}


def _normalise_legacy_implies_obj(implies_obj: Any) -> dict[str, Any]:
    if not isinstance(implies_obj, dict):
        raise ValueError("Rule logic must include an 'implies' object")

    args_obj = implies_obj.get("args")
    if isinstance(args_obj, list) and len(args_obj) == 2:
        left_raw, right_raw = args_obj
    else:
        left_raw = implies_obj.get("left", implies_obj.get("antecedent", implies_obj.get("lhs")))
        right_raw = implies_obj.get("right", implies_obj.get("consequent", implies_obj.get("rhs")))

    if left_raw is None or right_raw is None:
        raise ValueError("Implication must provide both antecedent/left and consequent/right")

    left = _normalise_bool_expr(left_raw)
    right = _normalise_bool_expr(right_raw)
    return {"op": "implies", "left": left, "right": right}


def _normalise_bool_expr(expr_obj: Any) -> dict[str, Any]:
    if isinstance(expr_obj, list):
        return {"op": "and", "args": [_normalise_bool_expr(item) for item in expr_obj]}
    if not isinstance(expr_obj, dict):
        raise ValueError("Boolean expression objects must be JSON objects")

    if set(expr_obj.keys()) == {"expr"}:
        return _normalise_bool_expr(expr_obj["expr"])

    atom = expr_obj.get("atom")
    if isinstance(atom, str):
        atom = _normalise_atom_string(atom)
        if atom == "true":
            raise ValueError("Boolean literal 'true' is not allowed")
        if atom == "false":
            raise ValueError("Boolean literal 'false' is not allowed")
        compare_expr = _compare_string_to_expr(atom)
        if compare_expr is not None:
            return compare_expr
        return {"atom": atom}

    op_raw = str(expr_obj.get("op", "")).strip().lower()
    if not op_raw:
        if "implies" in expr_obj:
            return _wrap_legacy_quantifier(_normalise_legacy_implies_obj(expr_obj["implies"]), expr_obj)
        if "constraint" in expr_obj:
            return _wrap_legacy_quantifier(_normalise_bool_expr(expr_obj["constraint"]), expr_obj)
        quant_op, vars_list = _extract_legacy_quantifier(expr_obj)
        if quant_op is not None:
            if "body" not in expr_obj:
                raise ValueError(f"Quantifier '{quant_op}' must provide a 'body'")
            return {"op": quant_op, "vars": vars_list, "body": _normalise_bool_expr(expr_obj["body"])}
        raise ValueError(f"Unsupported boolean expression object: {expr_obj}")

    op = {
        "forall": "all",
        "exists": "any",
        "neq": "ne",
    }.get(op_raw, op_raw)

    if op in {"and", "or"}:
        args = _ensure_list(expr_obj.get("args"), f"{op}.args")
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        normalised_args = [_normalise_bool_expr(arg) for arg in args]
        flattened: list[dict[str, Any]] = []
        for arg in normalised_args:
            if arg.get("op") == op:
                flattened.extend(_ensure_list(arg.get("args"), f"{op}.args"))
            else:
                flattened.append(arg)
        if op == "and":
            if any(_is_false_expr(arg) for arg in flattened):
                return {"op": "const", "value": False}
            flattened = [arg for arg in flattened if not _is_true_expr(arg)]
            if not flattened:
                return {"op": "const", "value": True}
        else:
            if any(_is_true_expr(arg) for arg in flattened):
                return {"op": "const", "value": True}
            flattened = [arg for arg in flattened if not _is_false_expr(arg)]
            if not flattened:
                return {"op": "const", "value": False}
        if len(flattened) == 1:
            return flattened[0]
        return {"op": op, "args": flattened}
    if op == "not":
        raw_arg = expr_obj.get("arg")
        if raw_arg is None:
            raw_args = _ensure_list(expr_obj.get("args"), "not.args")
            if len(raw_args) != 1:
                raise ValueError("Operator 'not' requires exactly one argument")
            raw_arg = raw_args[0]
        child = _normalise_bool_expr(raw_arg)
        if _is_true_expr(child):
            return {"op": "const", "value": False}
        if _is_false_expr(child):
            return {"op": "const", "value": True}
        return {"op": "not", "arg": child}
    if op in {"implies", "iff"}:
        args_obj = expr_obj.get("args")
        left_raw = expr_obj.get("left", expr_obj.get("antecedent", expr_obj.get("lhs")))
        right_raw = expr_obj.get("right", expr_obj.get("consequent", expr_obj.get("rhs")))
        if (left_raw is None or right_raw is None) and isinstance(args_obj, list) and len(args_obj) == 2:
            left_raw, right_raw = args_obj
        if left_raw is None or right_raw is None:
            raise ValueError(f"Operator '{op}' requires both left/right operands")
        left = _normalise_bool_expr(left_raw)
        right = _normalise_bool_expr(right_raw)
        if op == "implies":
            if _is_true_expr(left):
                return right
            if _is_false_expr(left):
                return {"op": "const", "value": True}
            if _is_true_expr(right):
                return {"op": "const", "value": True}
            if _is_false_expr(right):
                return {"op": "not", "arg": left}
        return {"op": op, "left": left, "right": right}
    if op in {"all", "any"}:
        vars_raw = expr_obj.get("vars", expr_obj.get("args"))
        vars_list: list[dict[str, Any]] = []
        body_raw = expr_obj.get("body")
        if isinstance(vars_raw, list):
            consumed = 0
            for item in vars_raw:
                if isinstance(item, dict) and {"name", "sort"} <= set(item.keys()):
                    vars_list.append({
                        "name": str(item.get("name", "")),
                        "sort": _canonical_sort_name(str(item.get("sort", ""))),
                    })
                    consumed += 1
                else:
                    break
            if consumed < len(vars_raw) and body_raw is None:
                remaining = vars_raw[consumed:]
                body_raw = remaining[0] if len(remaining) == 1 else {"op": "and", "args": remaining}
        elif vars_raw is not None:
            raise ValueError(f"Operator '{op}' requires vars as a list")
        if body_raw is None and "constraint" in expr_obj:
            body_raw = expr_obj.get("constraint")
        if body_raw is None:
            raise ValueError(f"Operator '{op}' requires a body")
        for var_def in vars_list:
            if not var_def["name"] or not var_def["sort"]:
                raise ValueError("Quantifier variables must declare both name and sort")
        return {"op": op, "vars": vars_list, "body": _normalise_bool_expr(body_raw)}
    if op == "const":
        return {"op": "const", "value": bool(expr_obj.get("value"))}
    if op in {"eq", "ne"}:
        left_raw = expr_obj.get("left")
        right_raw = expr_obj.get("right")
        if left_raw is None or right_raw is None:
            args_obj = _ensure_list(expr_obj.get("args"), f"{op}.args")
            if len(args_obj) != 2:
                raise ValueError(f"Operator '{op}' requires exactly two arguments")
            left_raw, right_raw = args_obj
        return {"op": op, "left": _normalise_term_expr(left_raw), "right": _normalise_term_expr(right_raw)}

    raise ValueError(f"Unsupported boolean expression object: {expr_obj}")


def _normalise_term_expr(term_obj: Any) -> Any:
    if isinstance(term_obj, dict):
        atom = term_obj.get("atom")
        if isinstance(atom, str):
            return _normalise_atom_string(atom)
    if isinstance(term_obj, str):
        return _normalise_atom_string(term_obj)
    return term_obj


def _normalise_atom_string(atom: str) -> str:
    text = str(atom).strip()
    if re.fullmatch(r"(?i)true(?:\(\))?", text):
        return "true"
    if re.fullmatch(r"(?i)false(?:\(\))?", text):
        return "false"
    open_count = text.count("(")
    close_count = text.count(")")
    if open_count > close_count:
        text = text + (")" * (open_count - close_count))
    eq_match = re.fullmatch(r"(.+?)\s*=\s*(.+)", text)
    if eq_match and "==" not in text and "!=" not in text and ">=" not in text and "<=" not in text:
        raise ValueError("Assignment-style '=' is not supported, use '==' for equality")
    return text


def _compare_string_to_expr(atom: str) -> dict[str, Any] | None:
    try:
        parsed = ast.parse(atom, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(parsed, ast.Compare) or len(parsed.ops) != 1 or len(parsed.comparators) != 1:
        return None
    op = parsed.ops[0]
    if isinstance(op, ast.Eq):
        name = "eq"
    elif isinstance(op, ast.NotEq):
        name = "ne"
    else:
        return None
    return {
        "op": name,
        "left": _ast_term_to_json(parsed.left),
        "right": _ast_term_to_json(parsed.comparators[0]),
    }


def _is_forbidden_bool_literal(text: str) -> bool:
    return bool(re.fullmatch(r"(?i)true(?:\(\))?|false(?:\(\))?", text.strip()))


def _parse_quantifier_head(text: str) -> tuple[str, list[dict[str, str]], str] | None:
    match = re.match(r"^(forall|exists)\b", text, flags=re.IGNORECASE)
    if match is None:
        return None
    quant_keyword = match.group(1).lower()
    rest = text[match.end():].strip()
    if not rest:
        raise ValueError(f"Quantifier '{quant_keyword}' is missing variable declarations")

    dot_idx = _find_top_level_char(rest, ".")
    if dot_idx < 0:
        raise ValueError(f"Quantifier '{quant_keyword}' must use 'vars. body' syntax")
    vars_blob = rest[:dot_idx].strip()
    body = rest[dot_idx + 1:].strip()
    if not vars_blob:
        raise ValueError(f"Quantifier '{quant_keyword}' is missing variables")
    if not body:
        raise ValueError(f"Quantifier '{quant_keyword}' is missing body")

    vars_list: list[dict[str, str]] = []
    for token in _split_top_level(vars_blob, ","):
        part = token.strip()
        if not part:
            continue
        var_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", part)
        if var_match is None:
            raise ValueError(
                f"Invalid quantifier variable declaration '{part}', expected 'name:Sort'"
            )
        vars_list.append({
            "name": var_match.group(1),
            "sort": _canonical_sort_name(var_match.group(2)),
        })
    if not vars_list:
        raise ValueError(f"Quantifier '{quant_keyword}' requires at least one variable")
    return ("all" if quant_keyword == "forall" else "any", vars_list, body)


def _strip_outer_parentheses(text: str) -> str:
    current = text.strip()
    while current.startswith("(") and current.endswith(")"):
        depth = 0
        balanced = True
        encloses_all = True
        for idx, ch in enumerate(current):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and idx != len(current) - 1:
                    encloses_all = False
                    break
        if not balanced or depth != 0 or not encloses_all:
            break
        current = current[1:-1].strip()
    return current


def _split_prefix_keyword(text: str, keyword: str) -> str | None:
    prefix = f"{keyword} "
    if not text.startswith(prefix):
        return None
    tail = text[len(prefix):].strip()
    if not tail:
        raise ValueError(f"Operator '{keyword}' requires one argument")
    return tail


def _split_top_level_keyword(text: str, keyword: str) -> tuple[str, str] | None:
    depth = 0
    klen = len(keyword)
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in logic string: {text}")
            continue
        if depth == 0 and idx <= len(text) - klen and text[idx:idx + klen].lower() == keyword:
            prev = text[idx - 1] if idx > 0 else " "
            nxt = text[idx + klen] if idx + klen < len(text) else " "
            if (not prev.isalnum() and prev != "_") and (not nxt.isalnum() and nxt != "_"):
                left = _strip_outer_parentheses(text[:idx].strip())
                right = _strip_outer_parentheses(text[idx + klen:].strip())
                if not left or not right:
                    raise ValueError(f"Operator '{keyword}' requires two arguments")
                return left, right
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in logic string: {text}")
    return None


def _split_top_level_multi_keyword(text: str, keyword: str) -> list[str] | None:
    depth = 0
    klen = len(keyword)
    parts: list[str] = []
    start = 0
    found = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == "(":
            depth += 1
            idx += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in logic string: {text}")
            idx += 1
            continue
        if depth == 0 and idx <= len(text) - klen and text[idx:idx + klen].lower() == keyword:
            prev = text[idx - 1] if idx > 0 else " "
            nxt = text[idx + klen] if idx + klen < len(text) else " "
            if (not prev.isalnum() and prev != "_") and (not nxt.isalnum() and nxt != "_"):
                part = _strip_outer_parentheses(text[start:idx].strip())
                if not part:
                    raise ValueError(f"Operator '{keyword}' has an empty argument")
                parts.append(part)
                start = idx + klen
                found = True
                idx += klen
                continue
        idx += 1
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in logic string: {text}")
    if not found:
        return None
    tail = _strip_outer_parentheses(text[start:].strip())
    if not tail:
        raise ValueError(f"Operator '{keyword}' has an empty argument")
    parts.append(tail)
    return parts


def _split_top_level_comparison(text: str) -> tuple[str, str, str] | None:
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in logic string: {text}")
            continue
        if depth != 0:
            continue
        if text.startswith("==", idx):
            left = text[:idx].strip()
            right = text[idx + 2:].strip()
            if not left or not right:
                raise ValueError("Equality comparison requires both left and right terms")
            return "eq", left, right
        if text.startswith("!=", idx):
            left = text[:idx].strip()
            right = text[idx + 2:].strip()
            if not left or not right:
                raise ValueError("Inequality comparison requires both left and right terms")
            return "ne", left, right
        if ch == "=":
            prev = text[idx - 1] if idx > 0 else ""
            nxt = text[idx + 1] if idx + 1 < len(text) else ""
            if prev != "=" and nxt != "=":
                raise ValueError("Assignment-style '=' is not supported, use '==' for equality")
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in logic string: {text}")
    return None


def _parse_call_like(text: str) -> tuple[str, list[str]] | None:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if match is None:
        return None
    func_name = match.group(1)
    open_idx = text.find("(", match.start(1) + len(func_name))
    if open_idx < 0:
        return None

    depth = 0
    close_idx = -1
    for idx in range(open_idx, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                close_idx = idx
                break
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in logic string: {text}")
    if close_idx < 0 or close_idx != len(text) - 1:
        return None

    args_blob = text[open_idx + 1:close_idx].strip()
    args = [] if not args_blob else [part.strip() for part in _split_top_level(args_blob, ",")]
    if any(not arg for arg in args):
        raise ValueError(f"Function call contains empty argument: {text}")
    return func_name, args


def _parse_term_string(text: str) -> str:
    term = str(text).strip()
    if not term:
        raise ValueError("Term expression must not be empty")
    if _is_forbidden_bool_literal(term):
        raise ValueError("Boolean literals cannot be used as terms")
    try:
        parsed = ast.parse(term, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid term syntax: {term}") from exc
    return _ast_term_to_json(parsed)


def _validate_atom_call_string(text: str) -> None:
    if _is_forbidden_bool_literal(text):
        raise ValueError("Boolean literals 'true/false' are not allowed")
    try:
        parsed = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid atom syntax: {text}") from exc
    if not isinstance(parsed, ast.Call):
        raise ValueError(f"Atom must be a predicate/function call: {text}")
    if not isinstance(parsed.func, ast.Name):
        raise ValueError(f"Atom cannot use attribute access: {text}")
    if parsed.keywords:
        raise ValueError(f"Atom cannot use keyword arguments: {text}")
    for arg in parsed.args:
        _ast_term_to_json(arg)


def _split_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in expression: {text}")
        elif ch == delimiter and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in expression: {text}")
    parts.append(text[start:])
    return parts


def _find_top_level_char(text: str, target: str) -> int:
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in expression: {text}")
        elif ch == target and depth == 0:
            return idx
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in expression: {text}")
    return -1


def _ast_term_to_json(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("Boolean literals cannot be used as terms")
        if isinstance(node.value, (int, float)):
            return ast.unparse(node)
        raise ValueError(f"Unsupported term literal: {ast.unparse(node)}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return ast.unparse(node)
        raise ValueError(f"Unsupported term syntax: {ast.unparse(node)}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords:
            raise ValueError("Function terms cannot use keyword arguments")
        args = ", ".join(_render_ast_term_arg(arg) for arg in node.args)
        return f"{node.func.id}({args})"
    raise ValueError(f"Unsupported term syntax: {ast.unparse(node)}")


def _render_ast_term_arg(node: ast.AST) -> str:
    value = _ast_term_to_json(node)
    if not isinstance(value, str):
        raise ValueError("Term arguments must render to strings")
    return value


def _compile_function_call(
    parsed: ast.Call,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
    atom: str,
) -> z3.ExprRef:
    if not isinstance(parsed.func, ast.Name):
        raise ValueError(f"Atom cannot use attribute access: {atom}")
    func_name = parsed.func.id
    if func_name not in symbol_table.functions:
        raise ValueError(f"Atom references undeclared function '{func_name}'")
    if parsed.keywords:
        raise ValueError(f"Atom cannot use keyword arguments: {atom}")

    func_ref = symbol_table.functions[func_name]
    if len(parsed.args) != func_ref.arity():
        raise ValueError(
            f"Function '{func_name}' expects {func_ref.arity()} args, got {len(parsed.args)}"
        )

    args = [_compile_term_node(node, symbol_table, variable_scope) for node in parsed.args]
    for arg_index, arg in enumerate(args):
        expected_sort = func_ref.domain(arg_index)
        if not expected_sort.eq(arg.sort()):
            raise ValueError(
                "Sort mismatch in function call "
                f"'{func_name}' arg {arg_index + 1}: expected '{expected_sort}', "
                f"got '{arg.sort()}' (atom: {atom})"
            )
    try:
        return func_ref(*args)
    except z3.Z3Exception as exc:
        raise ValueError(
            f"Sort mismatch in function call '{func_name}' (atom: {atom}): {exc}"
        ) from exc


def _compile_term_expr(
    term_obj: Any,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
) -> z3.ExprRef:
    if isinstance(term_obj, str):
        try:
            parsed = ast.parse(term_obj, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"Invalid term syntax: {term_obj}") from exc
        return _compile_term_node(parsed, symbol_table, variable_scope)
    raise ValueError(f"Unsupported term expression object: {term_obj}")


def _compile_term_node(
    node: ast.AST,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
) -> z3.ExprRef:
    if isinstance(node, ast.Name):
        ref = variable_scope.get(node.id)
        if ref is None:
            ref = symbol_table.constants.get(node.id)
        if ref is None:
            raise ValueError(f"Term references undeclared name '{node.id}'")
        return ref
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise ValueError("Boolean literals cannot be used as terms")
        if isinstance(value, int):
            return z3.IntVal(value)
        if isinstance(value, float):
            return z3.RealVal(str(value))
        if isinstance(value, str):
            ref = variable_scope.get(value)
            if ref is None:
                ref = symbol_table.constants.get(value)
            if ref is None:
                raise ValueError(f"Term references undeclared name '{value}'")
            return ref
        raise ValueError(f"Unsupported term literal: {ast.unparse(node)}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _compile_term_node(node.operand, symbol_table, variable_scope)
        if operand.sort().kind() not in {z3.Z3_INT_SORT, z3.Z3_REAL_SORT}:
            raise ValueError(f"Unsupported term syntax: {ast.unparse(node)}")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call):
        term = _compile_function_call(node, symbol_table, variable_scope, ast.unparse(node))
        if term.sort().kind() == z3.Z3_BOOL_SORT:
            raise ValueError(f"Function term must not be Bool-valued: {ast.unparse(node)}")
        return term
    raise ValueError(f"Unsupported term syntax: {ast.unparse(node)}")


def _compile_compare(
    parsed: ast.Compare,
    symbol_table: SymbolTable,
    variable_scope: dict[str, z3.ExprRef],
    atom: str,
) -> z3.ExprRef:
    if len(parsed.ops) != 1 or len(parsed.comparators) != 1:
        raise ValueError(f"Only simple binary comparisons are supported: {atom}")
    left = _compile_term_node(parsed.left, symbol_table, variable_scope)
    right = _compile_term_node(parsed.comparators[0], symbol_table, variable_scope)
    if not left.sort().eq(right.sort()):
        raise ValueError(
            f"Sort mismatch in comparison '{atom}': left is '{left.sort()}', right is '{right.sort()}'"
        )
    op = parsed.ops[0]
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    raise ValueError(f"Unsupported comparison operator in atom: {atom}")


def _render_atom(atom: str) -> str:
    atom = _normalise_atom_string(atom)
    parsed = ast.parse(atom, mode="eval").body
    if isinstance(parsed, ast.Compare):
        if len(parsed.ops) != 1 or len(parsed.comparators) != 1:
            raise ValueError(f"Only simple binary comparisons are supported: {atom}")
        left = _render_term_node(parsed.left)
        right = _render_term_node(parsed.comparators[0])
        if isinstance(parsed.ops[0], ast.Eq):
            return f"({left} == {right})"
        if isinstance(parsed.ops[0], ast.NotEq):
            return f"({left} != {right})"
        raise ValueError(f"Unsupported comparison operator in atom: {atom}")
    if not isinstance(parsed, ast.Call) or not isinstance(parsed.func, ast.Name):
        raise ValueError(f"Atom must be a function call, got: {atom}")
    if parsed.keywords:
        raise ValueError(f"Atom cannot use keyword arguments: {atom}")
    rendered_args = ", ".join(_render_term_node(arg) for arg in parsed.args)
    return f"{parsed.func.id}({rendered_args})"


def _render_term_expr(term_obj: Any) -> str:
    if isinstance(term_obj, str):
        parsed = ast.parse(term_obj, mode="eval").body
        return _render_term_node(parsed)
    raise ValueError(f"Unsupported term expression object: {term_obj}")


def _render_term_node(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("Boolean literals cannot be used as terms")
        if isinstance(node.value, (int, float)):
            return ast.unparse(node)
        if isinstance(node.value, str):
            return node.value
        raise ValueError(f"Unsupported term syntax: {ast.unparse(node)}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return ast.unparse(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords:
            raise ValueError(f"Function terms cannot use keyword arguments: {ast.unparse(node)}")
        args = ", ".join(_render_term_node(arg) for arg in node.args)
        return f"{node.func.id}({args})"
    raise ValueError(f"Unsupported term syntax: {ast.unparse(node)}")


def _is_true_expr(expr_obj: dict[str, Any]) -> bool:
    return expr_obj == {"atom": "true"} or expr_obj == {"op": "const", "value": True}


def _is_false_expr(expr_obj: dict[str, Any]) -> bool:
    return expr_obj == {"atom": "false"} or expr_obj == {"op": "const", "value": False}


def _normalise_nl_text(text: str) -> str:
    normalized = str(text).strip()
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return " ".join(normalized.split())


def _ensure_bool_formula(formula: z3.ExprRef) -> None:
    if not isinstance(formula, z3.ExprRef):
        raise ValueError(f"Expected a z3.ExprRef, got {type(formula).__name__}")
    if formula.sort().kind() != z3.Z3_BOOL_SORT:
        raise ValueError(f"Expected a Bool formula, got sort {formula.sort()}")


def _ensure_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a list")
    return value


def _ensure_list_of_dicts(value: Any, name: str) -> list[dict[str, Any]]:
    items = _ensure_list(value, name)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"'{name}[{idx}]' must be an object")
    return items


def _ensure_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be an object")
    return value


def _deep_copy_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))
