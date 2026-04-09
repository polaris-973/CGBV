from cgbv.core.phase3_grounded import _parse_batch_output, _validate_formula, _validate_formula_runtime
from cgbv.solver.z3_solver import Z3Solver


def test_runtime_validation_rejects_non_executable_grounded_formula() -> None:
    solver = Z3Solver(timeout_ms=100)
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}, {"name": "IntSort", "type": "IntSort"}],
        "functions": [
            {"name": "p", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "score", "domain": ["Entity"], "range": "IntSort"},
        ],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
    }
    domain = {
        "sorts": {"Entity": ["alice"]},
        "predicates": {
            "p": {("alice",): True},
        },
        "function_values": {},
    }
    ir = {
        "expr": {
            "op": "eq",
            "left": {"op": "value", "func": "score", "args": [{"op": "const", "name": "alice"}]},
            "right": {"op": "value", "func": "score", "args": [{"op": "const", "name": "alice"}]},
        },
    }

    static_err = _validate_formula(ir, domain, symbol_context=symbol_context)
    assert static_err is None

    err = _validate_formula_runtime(ir, domain, solver)
    assert err is not None
    assert "could not be executed" in err


def test_runtime_validation_accepts_executable_grounded_formula() -> None:
    solver = Z3Solver(timeout_ms=100)
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "p", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
    }
    domain = {
        "sorts": {"Entity": ["alice"]},
        "predicates": {
            "p": {("alice",): True},
        },
        "function_values": {},
    }
    ir = {
        "expr": {"op": "truth", "pred": "p", "args": [{"op": "const", "name": "alice"}]},
    }

    static_err = _validate_formula(ir, domain, symbol_context=symbol_context)
    assert static_err is None

    err = _validate_formula_runtime(ir, domain, solver)
    assert err is None


def test_static_validation_rejects_all_with_bare_boolean_argument() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "p", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
    }
    domain = {
        "predicates": {
            "p": {("alice",): True},
        },
        "function_values": {},
    }
    ir = {
        "expr": {"op": "all", "body": {"op": "truth", "pred": "p", "args": [{"op": "const", "name": "alice"}]}},
    }

    err = _validate_formula(ir, domain, symbol_context=symbol_context)

    assert err is not None
    assert "requires non-empty list field 'vars'" in err


def test_static_validation_rejects_all_with_multiple_arguments() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "p", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
    }
    domain = {
        "predicates": {
            "p": {("alice",): True},
        },
        "function_values": {},
    }
    ir = {
        "expr": {
            "op": "all",
            "vars": [{"name": "x", "sort": "Entity"}],
            "body": {"op": "truth", "pred": "p", "args": [{"op": "var", "name": "y"}]},
        },
    }

    err = _validate_formula(ir, domain, symbol_context=symbol_context)

    assert err is not None
    assert "Undeclared variable 'y'" in err


def test_parse_batch_output_accepts_indexed_object_map() -> None:
    raw = """{
      "1": {"expr": {"op": "truth", "pred": "p", "args": [{"op": "const", "name": "alice"}]}},
      "2": {"expr": {"op": "truth", "pred": "p", "args": [{"op": "const", "name": "alice"}]}}
    }"""

    parsed = _parse_batch_output(raw, 2)

    assert 0 in parsed
    assert 1 in parsed
    assert parsed[0]["expr"]["op"] == "truth"


def test_parse_batch_output_accepts_results_list_payload() -> None:
    raw = """{
      "results": [
        {"index": 1, "expr": {"op": "truth", "pred": "p", "args": [{"op": "const", "name": "alice"}]}},
        {"index": 2, "payload": {"expr": {"op": "truth", "pred": "p", "args": [{"op": "const", "name": "alice"}]}}}
      ]
    }"""

    parsed = _parse_batch_output(raw, 2)

    assert 0 in parsed
    assert 1 in parsed
    assert parsed[1]["expr"]["op"] == "truth"
