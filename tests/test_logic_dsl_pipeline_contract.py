import json
import z3
import pytest
from types import SimpleNamespace

from cgbv.core.logic_compiler import compile_theory_dsl, parse_logic_string
from cgbv.core.pipeline import _apply_phase5_repairs_to_dsl, _append_bridge_logic_to_dsl
from cgbv.core.phase1_formalize import _dsl_payload_for_prompt as _phase1_prompt_dsl
from cgbv.core.phase5_repair import _dsl_payload_for_prompt as _phase5_prompt_dsl
from cgbv.core.phase1_formalize import _extract_dsl_json
from cgbv.core.phase5_repair import _parse_unified_output
from cgbv.prompts.prompt_engine import PromptEngine


def _sample_payload() -> dict:
    return {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [
            {"name": "is_student", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "studies", "domain": ["Entity"], "range": "BoolSort"},
        ],
        "constants": {
            "entities": {"sort": "Entity", "members": ["alice"]}
        },
        "variables": [{"name": "x", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [
            {
                "nl": "All students study.",
                "logic": {
                    "op": "all",
                    "vars": [{"name": "x", "sort": "Entity"}],
                    "body": {
                        "op": "implies",
                        "left": {"atom": "is_student(x)"},
                        "right": {"atom": "studies(x)"},
                    },
                },
            }
        ],
        "query": {
            "nl": "Alice studies.",
            "logic": {"atom": "studies(alice)"},
        },
    }


def test_compile_theory_dsl_smoke() -> None:
    compiled = compile_theory_dsl(
        payload=_sample_payload(),
        premises_nl=["All students study."],
        conclusion_nl="Alice studies.",
    )

    assert len(compiled.premises) == 1
    assert str(compiled.q) == "studies(alice)"
    assert "premises = [" in compiled.raw_code
    assert "studies = Function" in compiled.raw_code


def test_compile_theory_dsl_accepts_compact_symbols_and_string_logic() -> None:
    payload = {
        "symbols": {
            "sorts": ["Entity"],
            "predicates": {
                "is_student": "Entity->Bool",
                "studies": "Entity->Bool",
            },
            "functions": {},
            "constants": {"alice": "Entity"},
        },
        "sentences": [
            {"nl": "All students study.", "logic": "forall x:Entity. implies(is_student(x), studies(x))"}
        ],
        "query": {"nl": "Alice studies.", "logic": "studies(alice)"},
        "background": [],
        "unknown_top_level": {"should": "be pruned"},
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["All students study."],
        conclusion_nl="Alice studies.",
    )

    assert str(compiled.premises[0]).startswith("ForAll")
    assert str(compiled.q) == "studies(alice)"
    assert sorted(compiled.payload.keys()) == [
        "background_constraints",
        "constants",
        "functions",
        "query",
        "sentences",
        "sorts",
        "variables",
    ]
    assert "unknown_top_level" not in compiled.payload


def test_phase1_phase5_prompt_dsl_uses_compact_view() -> None:
    payload = _sample_payload()
    phase1_prompt_dsl = _phase1_prompt_dsl(payload)
    phase5_prompt_dsl = _phase5_prompt_dsl(payload)

    for text in (phase1_prompt_dsl, phase5_prompt_dsl):
        assert '"symbols"' in text
        assert '"sentences"' in text
        assert '"query"' in text
        assert '"background"' in text
        assert '"background_constraints"' not in text


def test_phase1_phase5_prompt_dsl_fallback_stays_compact_for_malformed_payload() -> None:
    malformed_payload = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "p", "domain": ["Entity"], "range": "BoolSort"}],
        "constants": {"entities": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
        "background_constraints": "BROKEN_BACKGROUND_LIST",
        "sentences": "BROKEN_SENTENCE_LIST",
        "query": "BROKEN_QUERY",
    }

    for renderer in (_phase1_prompt_dsl, _phase5_prompt_dsl):
        text = renderer(malformed_payload)
        parsed = json.loads(text)
        assert sorted(parsed.keys()) == ["background", "query", "sentences", "symbols"]
        assert "background_constraints" not in text
        assert isinstance(parsed["background"], list)
        assert isinstance(parsed["sentences"], list)
        assert isinstance(parsed["query"], dict)
        assert isinstance(parsed["symbols"], dict)


def test_compile_theory_dsl_compact_signature_normalises_int_real_bool_domain_aliases() -> None:
    payload = {
        "symbols": {
            "sorts": [],
            "predicates": {"is_positive": "Int->Bool"},
            "functions": {"succ": "Int->Int", "is_equal_real": "Real,Real->Bool"},
            "constants": {},
        },
        "sentences": [
            {
                "nl": "Positive integers stay positive after succ.",
                "logic": "forall x:Int. implies(is_positive(x), is_positive(succ(x)))",
            }
        ],
        "query": {
            "nl": "There exists a positive integer.",
            "logic": "exists x:Int. is_positive(x)",
        },
        "background": [],
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["Positive integers stay positive after succ."],
        conclusion_nl="There exists a positive integer.",
    )
    assert len(compiled.premises) == 1
    assert str(compiled.q).startswith("Exists")


def test_parse_logic_string_supports_core_syntax() -> None:
    parsed = parse_logic_string("forall x:Entity, y:Entity. implies(p(x), and(q(y), ne(x,y)))")
    assert parsed["op"] == "all"
    assert parsed["vars"][0] == {"name": "x", "sort": "Entity"}
    assert parsed["vars"][1] == {"name": "y", "sort": "Entity"}
    assert parsed["body"]["op"] == "implies"


def test_parse_logic_string_rejects_true_false_literals() -> None:
    with pytest.raises(ValueError, match="Boolean literals"):
        parse_logic_string("true")
    with pytest.raises(ValueError, match="Boolean literals"):
        parse_logic_string("false")


def test_parse_logic_string_accepts_infix_and_implies_forms() -> None:
    parsed = parse_logic_string("(and(is_man(x), is_man(y), is_taller_than(x, y)) implies can_block_shooting_of(x, y))")
    assert parsed["op"] == "implies"
    assert parsed["left"]["op"] == "and"
    assert parsed["right"]["atom"] == "can_block_shooting_of(x, y)"

    parsed2 = parse_logic_string("(is_managed_building(b) and allows_pets(b))")
    assert parsed2["op"] == "and"
    assert parsed2["args"][0]["atom"] == "is_managed_building(b)"
    assert parsed2["args"][1]["atom"] == "allows_pets(b)"


def test_parse_logic_string_strips_redundant_outer_parentheses() -> None:
    parsed = parse_logic_string("((implies(p(x), q(x))))")
    assert parsed["op"] == "implies"
    assert parsed["left"]["atom"] == "p(x)"
    assert parsed["right"]["atom"] == "q(x)"


def test_parse_logic_string_infix_precedence_or_over_and_split() -> None:
    parsed = parse_logic_string("p(x) and q(x) or r(x)")
    assert parsed["op"] == "or"
    assert parsed["args"][0]["op"] == "and"
    assert parsed["args"][1]["atom"] == "r(x)"


def test_rendered_raw_code_is_executable_z3_compatibility_code() -> None:
    compiled = compile_theory_dsl(
        payload=_sample_payload(),
        premises_nl=["All students study."],
        conclusion_nl="Alice studies.",
    )

    namespace: dict[str, object] = {}
    exec(compiled.raw_code, namespace)  # noqa: S102

    assert isinstance(namespace["q"], z3.BoolRef)
    assert isinstance(namespace["premises"], list)
    assert str(namespace["q"]) == "studies(alice)"


def test_rendered_raw_code_executes_with_enum_sorts() -> None:
    payload = {
        "sorts": [{"name": "Color", "type": "EnumSort", "values": ["red", "blue"]}],
        "functions": [{"name": "likes", "domain": ["Color"], "range": "BoolSort"}],
        "constants": {},
        "variables": [{"name": "c", "sort": "Color"}],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "Red is liked.",
                "logic": {
                    "kind": "constraint",
                    "constraint": {"atom": "likes(red)"},
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "Blue is liked.",
            "logic": {
                "kind": "constraint",
                "constraint": {"atom": "likes(blue)"},
            },
        },
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["Red is liked."],
        conclusion_nl="Blue is liked.",
    )

    namespace: dict[str, object] = {}
    exec(compiled.raw_code, namespace)  # noqa: S102

    assert isinstance(namespace["q"], z3.BoolRef)
    assert str(namespace["q"]) == "likes(blue)"


def test_compile_theory_dsl_rejects_non_atomic_atom_string() -> None:
    payload = _sample_payload()
    payload["query"]["logic"] = {"atom": "studies(alice) and is_student(alice)"}

    with pytest.raises(ValueError, match="function call"):
        compile_theory_dsl(
            payload=payload,
            premises_nl=["All students study."],
            conclusion_nl="Alice studies.",
        )


def test_extract_dsl_json_from_fenced_output() -> None:
    raw = """```json
    {"sorts":[],"functions":[],"constants":{},"variables":[],"background_constraints":[],"sentences":[],"query":{}}
    ```"""

    payload = _extract_dsl_json(raw)

    assert payload is not None
    assert payload["sorts"] == []


def test_extract_dsl_json_returns_none_for_none_literal() -> None:
    assert _extract_dsl_json("NONE") is None


def test_parse_unified_output_json_payload() -> None:
    raw = """
    {
      "repairs": [
        {"sentence_index": 3, "logic": {"atom": "p(a)"}}
      ],
      "bridges": [
        {
          "op": "all",
          "vars": [{"name": "x", "sort": "Entity"}],
          "body": {
            "op": "implies",
            "left": {"atom": "p(x)"},
            "right": {"atom": "q(x)"}
          }
        }
      ]
    }
    """

    repairs, bridges = _parse_unified_output(raw, {3})

    assert 3 in repairs
    assert repairs[3]["atom"] == "p(a)"
    assert len(bridges) == 1
    assert bridges[0]["op"] == "all"


def test_pipeline_writeback_canonicalizes_repair_and_bridge_logic() -> None:
    payload = {
        "symbols": {
            "sorts": ["Entity"],
            "predicates": {"p": "Entity->Bool", "q": "Entity->Bool"},
            "functions": {},
            "constants": {"alice": "Entity"},
        },
        "sentences": [{"nl": "P implies Q.", "logic": "forall x:Entity. implies(p(x), q(x))"}],
        "query": {"nl": "Q(alice).", "logic": "q(alice)"},
        "background": [],
    }

    repairs = [
        SimpleNamespace(
            success=True,
            repaired_logic="forall x:Entity. implies(q(x), p(x))",
            sentence_index=0,
        )
    ]
    updated = _apply_phase5_repairs_to_dsl(payload, repairs)
    assert updated["sentences"][0]["logic"]["op"] == "all"

    with_bridge = _append_bridge_logic_to_dsl(updated, ["forall x:Entity. implies(p(x), q(x))"])
    assert with_bridge["background_constraints"][0]["op"] == "all"


def test_phase1_prompt_uses_dsl_few_shot_examples() -> None:
    engine = PromptEngine(
        templates_dir="./cgbv/prompts",
        few_shot_dir="./cgbv/prompts/few_shot",
    )

    rendered = engine.render(
        "phase1_formalize.j2",
        dataset="folio",
        premises=["All students study."],
        conclusion="Alice studies.",
        world_assumption="owa",
    )

    assert '"sorts"' in rendered
    assert "from z3 import *" not in rendered
    assert "- `sentence_index`" not in rendered
    assert "- `role`" not in rendered
    assert "- `kind`" not in rendered
    assert "forall x:S. body" in rendered
    assert "exists x:S. body" in rendered


def test_compile_theory_dsl_normalises_inline_exists_constraint_shape() -> None:
    payload = _sample_payload()
    payload["sentences"][0]["logic"] = {
        "constraint": {
            "op": "exists",
            "args": [
                {"name": "x", "sort": "Entity"},
                {"atom": "is_student(x)"},
            ],
        },
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["All students study."],
        conclusion_nl="Alice studies.",
    )

    assert isinstance(compiled.premises[0], z3.QuantifierRef)
    assert compiled.payload["sentences"][0]["logic"]["op"] == "any"
    assert compiled.payload["sentences"][0]["logic"]["vars"][0]["name"] == "x"


def test_compile_theory_dsl_supports_function_terms_inside_atoms() -> None:
    payload = {
        "sorts": [
            {"name": "Entity", "type": "DeclareSort"},
            {"name": "Money", "type": "DeclareSort"},
        ],
        "functions": [
            {"name": "monthly_rent_at", "domain": ["Entity"], "range": "Money"},
            {"name": "is_equal_to", "domain": ["Money", "Money"], "range": "BoolSort"},
        ],
        "constants": {
            "entities": {"sort": "Entity", "members": ["olive_garden"]},
            "money": {"sort": "Money", "members": ["m2000"]},
        },
        "variables": [{"name": "b", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "The monthly rent at the Olive Garden is $2000.",
                "logic": {
                    "kind": "constraint",
                    "constraint": {"atom": "is_equal_to(monthly_rent_at(olive_garden), m2000)"},
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "The monthly rent at the Olive Garden is $2000.",
            "logic": {
                "kind": "constraint",
                "constraint": {"atom": "monthly_rent_at(olive_garden) == m2000"},
            },
        },
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["The monthly rent at the Olive Garden is $2000."],
        conclusion_nl="The monthly rent at the Olive Garden is $2000.",
    )

    assert str(compiled.premises[0]) == "is_equal_to(monthly_rent_at(olive_garden), m2000)"
    assert str(compiled.q) == "monthly_rent_at(olive_garden) == m2000"


def test_compile_theory_dsl_raises_value_error_for_function_sort_mismatch() -> None:
    payload = {
        "sorts": [
            {"name": "Person", "type": "DeclareSort"},
            {"name": "City", "type": "DeclareSort"},
        ],
        "functions": [
            {"name": "is_resident", "domain": ["Person"], "range": "BoolSort"},
        ],
        "constants": {
            "people": {"sort": "Person", "members": ["alice"]},
            "cities": {"sort": "City", "members": ["paris"]},
        },
        "variables": [],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "Paris is a resident.",
                "logic": {
                    "kind": "constraint",
                    "constraint": {"atom": "is_resident(paris)"},
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "Alice is a resident.",
            "logic": {
                "kind": "constraint",
                "constraint": {"atom": "is_resident(alice)"},
            },
        },
    }

    with pytest.raises(ValueError, match="Sort mismatch in function call"):
        compile_theory_dsl(
            payload=payload,
            premises_nl=["Paris is a resident."],
            conclusion_nl="Alice is a resident.",
        )


def test_compile_theory_dsl_normalizes_legacy_quantified_implies_and_iff_shapes() -> None:
    payload = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [
            {"name": "is_student", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "is_learner", "domain": ["Entity"], "range": "BoolSort"},
        ],
        "constants": {"entities": {"sort": "Entity", "members": ["alice"]}},
        "variables": [],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "All students are learners.",
                "logic": {
                    "kind": "constraint",
                    "constraint": {
                        "forall": [{"name": "x", "sort": "Entity"}],
                        "implies": {
                            "antecedent": {"atom": "is_student(x)"},
                            "consequent": {"atom": "is_learner(x)"},
                        },
                    },
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "Alice is a learner iff Alice is a learner.",
            "logic": {
                "kind": "constraint",
                "constraint": {
                    "op": "iff",
                    "args": [
                        {"atom": "is_learner(alice)"},
                        {"atom": "is_learner(alice)"},
                    ],
                },
            },
        },
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["All students are learners."],
        conclusion_nl="Alice is a learner iff Alice is a learner.",
    )

    assert isinstance(compiled.premises[0], z3.QuantifierRef)
    assert str(compiled.q) == "is_learner(alice) == is_learner(alice)"


def test_compile_theory_dsl_rejects_assignment_style_equals_and_normalises_smart_quotes() -> None:
    payload = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [{"name": "reads", "domain": ["Entity", "Entity"], "range": "BoolSort"}],
        "constants": {
            "entities": {"sort": "Entity", "members": ["harry", "walden"]},
        },
        "variables": [{"name": "x", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "Harry read the book \"Walden\" by Henry Thoreau.",
                "logic": {
                    "kind": "constraint",
                    "constraint": {"atom": "reads(harry, walden)"},
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "Harry is Harry.",
            "logic": {
                "kind": "constraint",
                "constraint": {"atom": "x = harry"},
            },
        },
    }

    with pytest.raises(ValueError, match="Assignment-style"):
        compile_theory_dsl(
            payload=payload,
            premises_nl=["Harry read the book “Walden” by Henry Thoreau."],
            conclusion_nl="Harry is Harry.",
        )


def test_compile_theory_dsl_rejects_true_false_boolean_literals() -> None:
    payload = {
        "sorts": [{"name": "Book", "type": "DeclareSort"}],
        "functions": [{"name": "contains_knowledge", "domain": ["Book"], "range": "BoolSort"}],
        "constants": {"books": {"sort": "Book", "members": ["walden"]}},
        "variables": [{"name": "b", "sort": "Book"}],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "Books contain tons of knowledge.",
                "logic": {
                    "kind": "rule",
                    "forall": [{"name": "b", "sort": "Book"}],
                    "implies": {
                        "antecedent": {"atom": "true()"},
                        "consequent": {"atom": "contains_knowledge(b)"},
                    },
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "Walden contains knowledge.",
            "logic": {
                "kind": "constraint",
                "constraint": {"atom": "contains_knowledge(walden)"},
            },
        },
    }

    with pytest.raises(ValueError, match="Boolean literal 'true'"):
        compile_theory_dsl(
            payload=payload,
            premises_nl=["Books contain tons of knowledge."],
            conclusion_nl="Walden contains knowledge.",
        )


def test_compile_theory_dsl_repairs_missing_closing_parenthesis_in_atom() -> None:
    payload = {
        "sorts": [
            {"name": "Entity", "type": "DeclareSort"},
            {"name": "Money", "type": "DeclareSort"},
        ],
        "functions": [
            {"name": "security_deposit_at", "domain": ["Entity"], "range": "Money"},
            {"name": "is_more_than", "domain": ["Money", "Money"], "range": "BoolSort"},
        ],
        "constants": {
            "entities": {"sort": "Entity", "members": ["olive_garden"]},
            "money": {"sort": "Money", "members": ["m1500"]},
        },
        "variables": [{"name": "b", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": "$1500 is less than the security deposit at a managed building.",
                "logic": {
                    "kind": "constraint",
                    "exists": [{"name": "b", "sort": "Entity"}],
                    "constraint": {"atom": "is_more_than(m1500, security_deposit_at(b)"},
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": "$1500 is less than the security deposit at a managed building.",
            "logic": {
                "kind": "constraint",
                "exists": [{"name": "b", "sort": "Entity"}],
                "constraint": {"atom": "is_more_than(m1500, security_deposit_at(b))"},
            },
        },
    }

    compiled = compile_theory_dsl(
        payload=payload,
        premises_nl=["$1500 is less than the security deposit at a managed building."],
        conclusion_nl="$1500 is less than the security deposit at a managed building.",
    )

    assert "security_deposit_at(b)" in str(compiled.premises[0])


def test_compile_theory_dsl_old_and_new_shapes_are_semantically_equivalent() -> None:
    premises = ["All students are learners."]
    conclusion = "Alice is a learner."

    old_payload = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [
            {"name": "is_student", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "is_learner", "domain": ["Entity"], "range": "BoolSort"},
        ],
        "constants": {"entities": {"sort": "Entity", "members": ["alice"]}},
        "variables": [{"name": "x", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [
            {
                "sentence_index": 0,
                "role": "premise",
                "nl": premises[0],
                "logic": {
                    "kind": "rule",
                    "forall": [{"name": "x", "sort": "Entity"}],
                    "implies": {
                        "antecedent": {"atom": "is_student(x)"},
                        "consequent": {"atom": "is_learner(x)"},
                    },
                },
            }
        ],
        "query": {
            "sentence_index": 1,
            "role": "conclusion",
            "nl": conclusion,
            "logic": {
                "kind": "constraint",
                "constraint": {"atom": "is_learner(alice)"},
            },
        },
    }
    new_payload = {
        "sorts": [{"name": "Entity", "type": "DeclareSort"}],
        "functions": [
            {"name": "is_student", "domain": ["Entity"], "range": "BoolSort"},
            {"name": "is_learner", "domain": ["Entity"], "range": "BoolSort"},
        ],
        "constants": {"entities": {"sort": "Entity", "members": ["alice"]}},
        "variables": [{"name": "x", "sort": "Entity"}],
        "background_constraints": [],
        "sentences": [
            {
                "nl": premises[0],
                "logic": {
                    "op": "all",
                    "vars": [{"name": "x", "sort": "Entity"}],
                    "body": {
                        "op": "implies",
                        "left": {"atom": "is_student(x)"},
                        "right": {"atom": "is_learner(x)"},
                    },
                },
            }
        ],
        "query": {"nl": conclusion, "logic": {"atom": "is_learner(alice)"}},
    }

    compiled_old = compile_theory_dsl(old_payload, premises, conclusion)
    compiled_new = compile_theory_dsl(new_payload, premises, conclusion)

    assert [str(p) for p in compiled_old.premises] == [str(p) for p in compiled_new.premises]
    assert str(compiled_old.q) == str(compiled_new.q)
