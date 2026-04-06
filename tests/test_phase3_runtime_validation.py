from cgbv.core.phase3_grounded import _validate_formula_runtime
from cgbv.solver.z3_solver import Z3Solver


def test_runtime_validation_rejects_non_executable_grounded_formula() -> None:
    solver = Z3Solver(timeout_ms=100)
    domain = {
        "predicates": {
            "p": {("alice",): True},
        },
        "function_values": {},
    }

    err = _validate_formula_runtime('truth[f"p({x})"]', domain, solver)

    assert err is not None
    assert "could not be executed" in err


def test_runtime_validation_accepts_executable_grounded_formula() -> None:
    solver = Z3Solver(timeout_ms=100)
    domain = {
        "predicates": {
            "p": {("alice",): True},
        },
        "function_values": {},
    }

    err = _validate_formula_runtime('truth["p(alice)"]', domain, solver)

    assert err is None
