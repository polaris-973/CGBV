from pathlib import Path

import z3

from cgbv.core.grounded_template_ir import parse_grounded_template_ir, validate_grounded_template_ir
from cgbv.core.logic_compiler import SymbolTable, extract_logic_predicates
from cgbv.core.phase3_grounded import (
    _build_template_batch_messages,
    _build_template_messages,
    _materialise_template,
    _parse_batch_output,
    _symbol_context,
)
from cgbv.core.phase5_repair import _extract_relevant_predicates, _normalize_fis_references
from cgbv.prompts.prompt_engine import PromptEngine


def test_extract_logic_predicates_includes_term_level_functions() -> None:
    logic_obj = {
        "kind": "constraint",
        "constraint": {
            "atom": "is_text_sequence(output_of(machine_translation))",
        },
    }

    names = extract_logic_predicates(logic_obj)

    assert "is_text_sequence" in names
    assert "output_of" in names


def test_ir_ordering_rejects_entity_sort_comparison() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice", "bob"]}},
        "variables": [],
    }
    payload = {
        "expr": {
            "op": "gt",
            "left": {"op": "const", "name": "bob"},
            "right": {"op": "const", "name": "alice"},
        },
    }

    ir = parse_grounded_template_ir(payload, symbol_context)
    try:
        validate_grounded_template_ir(ir, symbol_context)
    except ValueError as e:
        assert "only allowed on numeric sorts" in str(e)
    else:
        raise AssertionError("Expected ordering over Entity sort to be rejected.")


def test_ir_ordering_accepts_numeric_alias_sort() -> None:
    symbol_context = {
        "sorts": [
            {"name": "Entity", "type": "DeclareSort"},
            {"name": "Score", "type": "IntSort"},
        ],
        "functions": [
            {"name": "score", "domain": ["Entity"], "range": "Score"},
        ],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice", "bob"]}},
        "variables": [],
    }
    payload = {
        "expr": {
            "op": "gt",
            "left": {"op": "value", "func": "score", "args": [{"op": "const", "name": "alice"}]},
            "right": {"op": "value", "func": "score", "args": [{"op": "const", "name": "bob"}]},
        },
    }

    ir = parse_grounded_template_ir(payload, symbol_context)
    validate_grounded_template_ir(ir, symbol_context)


def test_ir_parser_normalizes_legacy_operator_field_aliases() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "p", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice", "bob"]}},
        "variables": [],
    }
    payload = {
        "expr": {
            "op": "implies",
            "antecedent": {
                "op": "not",
                "args": [
                    {"op": "truth", "pred": "p", "args": [{"op": "const", "value": "alice"}]},
                ],
            },
            "consequent": {
                "op": "eq",
                "args": [
                    {"op": "const", "value": "alice"},
                    {"op": "const", "value": "alice"},
                ],
            },
        },
    }

    ir = parse_grounded_template_ir(payload, symbol_context)
    validate_grounded_template_ir(ir, symbol_context)

    expr = ir["expr"]
    assert "left" in expr and "right" in expr
    assert expr["left"]["op"] == "not"
    assert "arg" in expr["left"]
    assert expr["right"]["op"] == "eq"
    assert expr["right"]["left"]["op"] == "const"
    assert expr["right"]["left"]["name"] == "alice"


def test_ir_parser_normalizes_zero_arity_value_call_to_constant() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "is_pet", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["tom"]}},
        "variables": [],
    }
    payload = {
        "expr": {
            "op": "truth",
            "pred": "is_pet",
            "args": [{"op": "value", "func": "tom", "args": []}],
        },
    }

    ir = parse_grounded_template_ir(payload, symbol_context)
    validate_grounded_template_ir(ir, symbol_context)

    arg0 = ir["expr"]["args"][0]
    assert arg0 == {"op": "const", "name": "tom"}


def test_ir_parser_normalizes_eq_lhs_rhs_alias() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice", "bob"]}},
        "variables": [],
    }
    payload = {
        "expr": {
            "op": "eq",
            "lhs": {"op": "const", "value": "alice"},
            "rhs": {"op": "const", "value": "bob"},
        },
    }

    ir = parse_grounded_template_ir(payload, symbol_context)
    validate_grounded_template_ir(ir, symbol_context)

    assert ir["expr"]["left"]["name"] == "alice"
    assert ir["expr"]["right"]["name"] == "bob"


def test_phase5_extract_relevant_predicates_supports_ir_debug_render() -> None:
    names = _extract_relevant_predicates(
        "Implies(p(a), q(a))",
        "truth(is_pet(alice)) and (value(score(alice)) > 3)",
    )

    assert "is_pet" in names
    assert "score" in names


def test_phase5_normalize_fis_supports_ir_debug_render_form() -> None:
    normalized = _normalize_fis_references(
        "truth(monthly_rent_is(apt1, low)) and truth(is_apartment(apt1))"
    )

    assert 'value["monthly_rent(apt1)"] == "low"' in normalized
    assert "truth(is_apartment(apt1))" in normalized


def test_phase3_prompts_use_minimal_expr_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt_engine = PromptEngine(
        templates_dir=str(root / "cgbv" / "prompts"),
        few_shot_dir=str(root / "cgbv" / "prompts" / "few_shot"),
    )

    single = _build_template_messages(
        sentence="Alice is a pet.",
        domain_schema_str="(schema)",
        prompt_engine=prompt_engine,
        sentence_logic={},
        symbol_context={},
        sentence_index=3,
    )[0]["content"]
    assert "Sentence Index (Debug Context Only)" in single
    assert "`sentence_index`: `3`" in single
    assert "- `expr` (object)" in single
    assert "- `role`: " not in single
    assert "- `kind`: " not in single

    batch = _build_template_batch_messages(
        sentences=["S1", "S2"],
        domain_schema_str="(schema)",
        prompt_engine=prompt_engine,
        sentence_logic_pairs=[
            {"sentence_index": 0, "logic_json": "{}", "core_predicates": []},
            {"sentence_index": 1, "logic_json": "{}", "core_predicates": []},
        ],
        symbol_context={},
    )[0]["content"]
    assert "Debug sentence_index: `0`" in batch
    assert "Debug sentence_index: `1`" in batch
    assert "`expr`" in batch
    assert "Required metadata:" not in batch
    assert "role=" not in batch
    assert "kind=grounded_template" not in batch


def test_parse_batch_output_uses_slot_index_mapping() -> None:
    raw = """[1] {"expr":{"op":"const_bool","value":true}}
[2] {"expr":{"op":"const_bool","value":false}}"""

    parsed = _parse_batch_output(raw, 2)

    assert set(parsed.keys()) == {0, 1}
    assert parsed[0]["expr"]["value"] is True
    assert parsed[1]["expr"]["value"] is False


def test_materialise_template_ignores_payload_metadata_alignment_fields() -> None:
    symbol_context = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "is_pet", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"Entity": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
    }
    domain = {
        "sorts": {"Entity": ["alice"]},
        "predicates": {"is_pet": {("alice",): True}},
        "function_values": {},
    }
    payload = {
        "sentence_index": 1,
        "role": "conclusion",
        "kind": "grounded_template",
        "expr": {"op": "truth", "pred": "is_pet", "args": [{"op": "const", "name": "alice"}]},
    }

    tmpl = _materialise_template(
        idx=0,
        sentence="Alice is a pet.",
        payload=payload,
        domain=domain,
        symbol_context=symbol_context,
        sentence_logic=None,
        solver=None,
    )

    assert tmpl.sentence_index == 0
    assert tmpl.template_ir == {"expr": payload["expr"]}


def test_symbol_context_falls_back_to_symbol_table() -> None:
    entity = z3.DeclareSort("Entity")
    score = z3.IntSort()
    symbol_table = SymbolTable(
        sorts={"Entity": entity, "IntSort": score},
        functions={"score": z3.Function("score", entity, score)},
        constants={"alice": z3.Const("alice", entity)},
        variables={"x": z3.Const("x", entity)},
    )

    ctx = _symbol_context({}, symbol_table)

    assert ctx["sorts"]
    assert ctx["functions"]
    assert ctx["constants"]
    assert ctx["variables"]
