import z3
import pytest

import cgbv.core.phase1_formalize as phase1_formalize
from cgbv.core.logic_compiler import compile_sentence_logic, compile_theory_dsl

from cgbv.core.phase1_formalize import (
    Phase1Result,
    _run_bridge_check,
    _check_model_vacuousness,
    _validate_output,
    check_z3_sort_consistency,
)


def test_validate_output_rejects_raw_python_bool_premise() -> None:
    thing = z3.DeclareSort("Thing")
    item = z3.Const("item", thing)
    is_valid = z3.Function("is_valid", thing, z3.BoolSort())

    err = _validate_output(
        ["Item is valid."],
        {
            "premises": [True],
            "q": is_valid(item),
        },
    )

    assert err is not None
    assert "raw Python boolean" in err[0]


def test_validate_output_rejects_boolean_literal_in_q() -> None:
    thing = z3.DeclareSort("Thing")
    item = z3.Const("item", thing)
    is_valid = z3.Function("is_valid", thing, z3.BoolSort())

    err = _validate_output(
        ["Item is valid."],
        {
            "premises": [is_valid(item)],
            "q": z3.BoolVal(True),
        },
    )

    assert err is not None
    assert "literal boolean constant" in err[0]


def test_validate_output_rejects_strengthened_disjunction() -> None:
    thing = z3.DeclareSort("Thing")
    item = z3.Const("item", thing)
    is_spicy = z3.Function("is_spicy", thing, z3.BoolSort())
    is_hot = z3.Function("is_hot", thing, z3.BoolSort())

    err = _validate_output(
        ["Item is spicy or hot."],
        {
            "premises": [z3.And(z3.Or(is_spicy(item), is_hot(item)), is_spicy(item))],
            "q": is_hot(item),
        },
    )

    assert err is not None
    assert "strengthens a disjunctive NL premise" in err[0]


def test_validate_output_allows_plain_disjunction() -> None:
    thing = z3.DeclareSort("Thing")
    item = z3.Const("item", thing)
    is_spicy = z3.Function("is_spicy", thing, z3.BoolSort())
    is_hot = z3.Function("is_hot", thing, z3.BoolSort())

    err = _validate_output(
        ["Item is spicy or hot."],
        {
            "premises": [z3.Or(is_spicy(item), is_hot(item))],
            "q": is_hot(item),
        },
    )

    assert err is None


def test_sort_check_detects_bool_predicate_used_as_entity_even_with_fences() -> None:
    code = """```python
from z3 import *

Entity = DeclareSort('Entity')
x = Const('x', Entity)
y = Const('y', Entity)

security_deposit_at = Function('security_deposit_at', Entity, Entity, BoolSort())
equal_to = Function('equal_to', Entity, Entity, BoolSort())

premises = [
    ForAll([x, y], equal_to(security_deposit_at(x, y), y))
]
q = equal_to(y, y)
```"""

    err = check_z3_sort_consistency(code)

    assert err is not None
    assert "security_deposit_at" in err
    assert "equal_to" in err


def test_vacuous_check_skips_when_reachable_ground_fact_exists() -> None:
    item = z3.DeclareSort("Item")
    dried_thai_chilies = z3.Const("dried_thai_chilies", item)
    baked_by_melissa_product = z3.Const("baked_by_melissa_product", item)
    x = z3.Const("x", item)

    is_baked_sweet = z3.Function("is_baked_sweet", item, z3.BoolSort())
    is_spicy = z3.Function("is_spicy", item, z3.BoolSort())
    is_cupcake = z3.Function("is_cupcake", item, z3.BoolSort())
    is_mala_hotpot = z3.Function("is_mala_hotpot", item, z3.BoolSort())
    is_from_baked_by_melissa = z3.Function(
        "is_from_baked_by_melissa",
        item,
        z3.BoolSort(),
    )

    premises = [
        z3.ForAll([x], z3.Implies(is_baked_sweet(x), z3.Not(is_spicy(x)))),
        z3.ForAll([x], z3.Implies(is_cupcake(x), is_baked_sweet(x))),
        z3.ForAll([x], z3.Implies(is_mala_hotpot(x), is_spicy(x))),
        z3.ForAll(
            [x],
            z3.Implies(is_from_baked_by_melissa(x), is_cupcake(x)),
        ),
        z3.Or(
            is_spicy(dried_thai_chilies),
            is_mala_hotpot(dried_thai_chilies),
            z3.Not(is_baked_sweet(dried_thai_chilies)),
        ),
    ]
    q = is_mala_hotpot(dried_thai_chilies)

    solver = z3.Solver()
    solver.add(*premises)
    solver.add(q)
    assert solver.check() == z3.sat

    vacuous, always_false = _check_model_vacuousness(
        solver.model(),
        premises,
        q,
        {
            "dried_thai_chilies": dried_thai_chilies,
            "baked_by_melissa_product": baked_by_melissa_product,
            "x": x,
            "is_baked_sweet": is_baked_sweet,
            "is_spicy": is_spicy,
            "is_cupcake": is_cupcake,
            "is_mala_hotpot": is_mala_hotpot,
            "is_from_baked_by_melissa": is_from_baked_by_melissa,
        },
        {"x"},
    )

    assert vacuous is False
    assert always_false == []


def test_vacuous_check_detects_all_rules_no_grounding() -> None:
    person = z3.DeclareSort("Person")
    mary = z3.Const("mary", person)
    x = z3.Const("x", person)

    is_student = z3.Function("is_student", person, z3.BoolSort())
    studies_hard = z3.Function("studies_hard", person, z3.BoolSort())
    passes_exams = z3.Function("passes_exams", person, z3.BoolSort())

    premises = [
        z3.ForAll([x], z3.Implies(is_student(x), studies_hard(x))),
        z3.ForAll([x], z3.Implies(studies_hard(x), passes_exams(x))),
    ]
    q = passes_exams(mary)

    solver = z3.Solver()
    solver.add(*premises)
    solver.add(z3.Not(q))
    assert solver.check() == z3.sat

    vacuous, always_false = _check_model_vacuousness(
        solver.model(),
        premises,
        q,
        {
            "mary": mary,
            "x": x,
            "is_student": is_student,
            "studies_hard": studies_hard,
            "passes_exams": passes_exams,
        },
        {"x"},
    )

    assert vacuous is True
    assert set(always_false) == {"is_student", "studies_hard", "passes_exams"}


@pytest.mark.anyio
async def test_phase15_bridge_writeback_is_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [
            {"name": "is_student", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "studies", "domain": ["Entity"], "range": "BoolSort"},
        ],
        "constants": {"entities": {"sort": "Entity", "members": ["alice"]}},
        "variables": [{"name": "x", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [{"nl": "All students study.", "logic": "forall x:Entity. implies(is_student(x), studies(x))"}],
        "query": {"nl": "Alice studies.", "logic": "studies(alice)"},
    }
    compiled = compile_theory_dsl(payload, ["All students study."], "Alice studies.")

    result = Phase1Result(
        verdict="Not Entailed",
        verdict_pre_bridge="Not Entailed",
        premises=list(compiled.premises),
        q=compiled.q,
        background_constraints=[],
        bound_var_names=set(compiled.bound_var_names),
        model_info=None,
        model_info_q=None,
        namespace=compiled.namespace,
        raw_code=compiled.raw_code,
        dsl_payload=compiled.payload,
        symbol_table=compiled.symbol_table,
    )

    legacy_bridge_logic = {
        "kind": "rule",
        "forall": [{"name": "x", "sort": "Entity"}],
        "implies": {
            "antecedent": {"atom": "studies(x)"},
            "consequent": {"atom": "is_student(x)"},
        },
    }
    bridge_formula = compile_sentence_logic(
        legacy_bridge_logic,
        compiled.symbol_table,
        dict(compiled.symbol_table.variables),
    )

    monkeypatch.setattr(phase1_formalize, "_find_disconnected_premises", lambda premises, q: [0])
    monkeypatch.setattr(phase1_formalize, "_get_connected_predicates", lambda idx, premises, q: set())

    async def _fake_repair_bridge_premise(**kwargs):
        return bridge_formula, legacy_bridge_logic

    monkeypatch.setattr(phase1_formalize, "_repair_bridge_premise", _fake_repair_bridge_premise)

    class _DummySolver:
        def check_entailment(self, premises, q):
            return "Not Entailed", None

    updated = await _run_bridge_check(
        result=result,
        premises_nl=["All students study."],
        conclusion_nl="Alice studies.",
        llm=None,  # patched out by fake repair function
        solver=_DummySolver(),
        prompt_engine=None,  # patched out by fake repair function
        task_type="entailment",
        bridge_retries=1,
    )

    assert updated.dsl_payload["background_constraints"]
    appended = updated.dsl_payload["background_constraints"][-1]
    assert isinstance(appended, dict)
    assert appended.get("op") == "all"
    assert "kind" not in appended
