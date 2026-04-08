import z3

from cgbv.core.phase1_formalize import (
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
