from __future__ import annotations

from cgbv.core.logic_compiler import compile_theory_dsl


def test_compile_theory_accepts_numeric_term_literals() -> None:
    payload = {
        "symbols": {
            "sorts": ["Entity", "Int"],
            "predicates": {},
            "functions": {"population_of": "Entity->Int"},
            "constants": {"metropolis": "Entity"},
        },
        "sentences": [
            {
                "nl": "The population of metropolis is 42300000000.",
                "logic": "population_of(metropolis)==42300000000",
            }
        ],
        "query": {
            "nl": "The population of metropolis is 42300000000.",
            "logic": "population_of(metropolis)==42300000000",
        },
        "background": [],
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["The population of metropolis is 42300000000."],
        conclusion_nl="The population of metropolis is 42300000000.",
    )

    assert "42300000000" in str(compiled.premises[0])
    assert "42300000000" in str(compiled.q)


def test_compile_theory_sanitizes_invalid_constant_names_from_compact_symbols() -> None:
    payload = {
        "symbols": {
            "sorts": ["Entity"],
            "predicates": {"is_budget_anchor": "Entity->Bool"},
            "functions": {},
            "constants": {"billion_42.3": "Entity"},
        },
        "sentences": [
            {
                "nl": "billion_42.3 is a budget anchor.",
                "logic": "is_budget_anchor(billion_42.3)",
            }
        ],
        "query": {
            "nl": "billion_42.3 is a budget anchor.",
            "logic": "is_budget_anchor(billion_42.3)",
        },
        "background": [],
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["billion_42.3 is a budget anchor."],
        conclusion_nl="billion_42.3 is a budget anchor.",
    )

    symbol_names = set(compiled.symbol_table.constants.keys())
    assert "billion_42.3" not in symbol_names
    assert any(name.startswith("billion_42_3") for name in symbol_names)
    assert "billion_42_3" in str(compiled.premises[0])


def test_compile_theory_auto_harmonizes_entity_sort_mismatch() -> None:
    payload = {
        "sorts": [
            {"name": "Entity", "type": "DeclareSort"},
            {"name": "Pet", "type": "DeclareSort"},
            {"name": "Building", "type": "DeclareSort"},
        ],
        "functions": [
            {"name": "is_pet", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "is_managed_building", "domain": ["Entity"], "range": "BoolSort"},
            {
                "name": "is_allowed_in_managed_building",
                "domain": ["Entity", "Entity"],
                "range": "BoolSort",
            },
        ],
        "constants": {
            "pets": {"sort": "Pet", "members": ["fluffy"]},
            "buildings": {"sort": "Building", "members": ["apt_a"]},
        },
        "variables": [
            {"name": "p", "sort": "Pet"},
            {"name": "b", "sort": "Building"},
        ],
        "background_constraints": [],
        "sentences": [
            {
                "nl": "Any pet is allowed in any managed building.",
                "logic": "forall p:Pet, b:Building. implies(and(is_pet(p), is_managed_building(b)), is_allowed_in_managed_building(p,b))",
            }
        ],
        "query": {
            "nl": "Fluffy is allowed in apt_a.",
            "logic": "is_allowed_in_managed_building(fluffy, apt_a)",
        },
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["Any pet is allowed in any managed building."],
        conclusion_nl="Fluffy is allowed in apt_a.",
    )

    variable_sorts = {item["name"]: item["sort"] for item in compiled.payload["variables"]}
    assert variable_sorts["p"] == "Entity"
    assert variable_sorts["b"] == "Entity"
    constant_sorts = {
        bucket["sort"]
        for bucket in compiled.payload["constants"].values()
        if isinstance(bucket, dict)
    }
    assert "Entity" in constant_sorts


def test_entity_harmonization_does_not_collapse_active_many_sort_domains() -> None:
    payload = {
        "sorts": [
            {"name": "Entity", "type": "DeclareSort"},
            {"name": "Book", "type": "DeclareSort"},
        ],
        "functions": [
            {"name": "reads", "domain": ["Entity", "Book"], "range": "BoolSort"},
        ],
        "constants": {
            "entities": {"sort": "Entity", "members": ["alice"]},
            "books": {"sort": "Book", "members": ["novel_a"]},
        },
        "variables": [],
        "background_constraints": [],
        "sentences": [
            {
                "nl": "Novel_a reads novel_a.",
                "logic": "reads(novel_a, novel_a)",
            }
        ],
        "query": {
            "nl": "Alice reads novel_a.",
            "logic": "reads(alice, novel_a)",
        },
    }

    try:
        compile_theory_dsl(
            payload=payload,
            premises_nl=["Novel_a reads novel_a."],
            conclusion_nl="Alice reads novel_a.",
        )
        raise AssertionError("Expected sort mismatch, but compile unexpectedly succeeded.")
    except ValueError as exc:
        assert "Sort mismatch in function call 'reads'" in str(exc)
