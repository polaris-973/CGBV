from cgbv.core.grounded_template_ir import (
    evaluate_grounded_template_ir,
    parse_grounded_template_ir,
    render_grounded_template_ir,
    validate_grounded_template_ir,
)


def _symbol_context() -> dict:
    return {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "p", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice", "bob"]}},
        "variables": [{"name": "x", "sort": "Entity"}],
    }


def test_parse_validate_evaluate_quantifier_ir() -> None:
    payload = {
        "expr": {
            "op": "all",
            "vars": [{"name": "x", "sort": "Entity"}],
            "body": {"op": "truth", "pred": "p", "args": [{"op": "var", "name": "x"}]},
        },
    }
    domain = {
        "sorts": {"Entity": ["alice", "bob"]},
        "predicates": {"p": {("alice",): True, ("bob",): True}},
        "function_values": {},
    }

    ir = parse_grounded_template_ir(payload, _symbol_context())
    validate_grounded_template_ir(ir, _symbol_context(), required_predicates={"p"})

    assert evaluate_grounded_template_ir(ir, domain) is True
    assert "forall" in render_grounded_template_ir(ir)


def test_validate_rejects_undeclared_predicate() -> None:
    payload = {
        "expr": {"op": "truth", "pred": "q", "args": [{"op": "const", "name": "alice"}]},
    }
    ir = parse_grounded_template_ir(payload, _symbol_context())

    try:
        validate_grounded_template_ir(ir, _symbol_context())
    except ValueError as e:
        assert "Undeclared predicate 'q'" in str(e)
    else:
        raise AssertionError("Expected validate_grounded_template_ir to reject undeclared predicate.")


def test_parse_supports_top_level_expr_node_without_expr_wrapper() -> None:
    payload = {
        "op": "forall",
        "variables": [{"var": "x", "type": "Entity"}],
        "body": {"op": "truth", "pred": "p", "args": [{"op": "var", "name": "x"}]},
    }

    ir = parse_grounded_template_ir(payload, _symbol_context())
    validate_grounded_template_ir(ir, _symbol_context(), required_predicates={"p"})

    assert ir["expr"]["op"] == "all"
    assert ir["expr"]["vars"] == [{"var": "x", "type": "Entity", "name": "x", "sort": "Entity"}]


def test_parse_supports_legacy_shorthand_truth_and_term_aliases() -> None:
    payload = {
        "expr": {
            "implies": [
                {"truth": {"predicate": "p", "args": [{"const": "alice"}]}},
                {"truth": {"name": "p", "args": [{"const": "alice"}]}},
            ]
        }
    }
    symbol_context = _symbol_context()

    ir = parse_grounded_template_ir(payload, symbol_context)
    validate_grounded_template_ir(ir, symbol_context, required_predicates={"p"})

    assert ir["expr"]["op"] == "implies"
    assert ir["expr"]["left"]["op"] == "truth"
    assert ir["expr"]["left"]["pred"] == "p"
    assert ir["expr"]["left"]["args"][0]["op"] == "const"
    assert ir["expr"]["right"]["args"][0]["op"] == "const"
