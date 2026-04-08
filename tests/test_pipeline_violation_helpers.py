import z3

from cgbv.core.multi_witness import WitnessCheckResult
from cgbv.core.gap_analysis import compute_gap_analysis
from cgbv.core.phase2_witness import Phase2Result
from cgbv.core.phase4_check import Mismatch, Phase4Result, SentenceEval
from cgbv.core.phase5_repair import Phase5Result, RepairEntry
from cgbv.core.phase3_grounded import Phase3Result
from cgbv.core.pipeline import (
    _merge_witness_bank,
    _open_issues_from_history_witness_results,
    _phase4_violation_keys,
    _phase4_violation_keys_for_sentences,
    _phase5_to_dict,
)


def test_phase4_violation_keys_skip_eliminated_witnesses() -> None:
    phase4_results = [
        Phase4Result(
            mismatches=[
                Mismatch(
                    sentence_index=1,
                    nl_sentence="s1",
                    mismatch_type="weakening",
                    fol_truth=True,
                    grounded_truth=False,
                    fol_formula_str="p(a)",
                    grounded_formula='truth["p(a)"]',
                )
            ],
            all_passed=False,
            evaluations=[
                SentenceEval(
                    sentence_index=1,
                    nl_sentence="s1",
                    fol_truth=True,
                    grounded_truth=False,
                    mismatch=True,
                    mismatch_type="weakening",
                    grounding_failed=False,
                    witness_index=0,
                    witness_side="not_q",
                )
            ],
            witness_index=0,
            witness_side="not_q",
        ),
        Phase4Result(
            mismatches=[],
            all_passed=False,
            evaluations=[
                SentenceEval(
                    sentence_index=2,
                    nl_sentence="s2",
                    fol_truth=None,
                    grounded_truth=None,
                    mismatch=False,
                    mismatch_type=None,
                    grounding_failed=False,
                    witness_index=1,
                    witness_side="q",
                    error="eval failed",
                )
            ],
            witness_index=1,
            witness_side="q",
            num_unverifiable=1,
        ),
    ]

    keys = _phase4_violation_keys(phase4_results, eliminated_witness_indices={0})

    assert ("mismatch", 1, 0, "not_q") not in keys
    assert ("unverifiable", 2, 1, "q") in keys

def test_phase4_violation_keys_for_sentences_filters_slice() -> None:
    phase4_results = [
        Phase4Result(
            mismatches=[
                Mismatch(
                    sentence_index=1,
                    nl_sentence="s1",
                    mismatch_type="weakening",
                    fol_truth=True,
                    grounded_truth=False,
                    fol_formula_str="p(a)",
                    grounded_formula='truth["p(a)"]',
                ),
                Mismatch(
                    sentence_index=3,
                    nl_sentence="s3",
                    mismatch_type="strengthening",
                    fol_truth=False,
                    grounded_truth=True,
                    fol_formula_str="q(a)",
                    grounded_formula='truth["q(a)"]',
                ),
            ],
            all_passed=False,
            evaluations=[
                SentenceEval(
                    sentence_index=2,
                    nl_sentence="s2",
                    fol_truth=None,
                    grounded_truth=None,
                    mismatch=False,
                    mismatch_type=None,
                    grounding_failed=False,
                    witness_index=0,
                    witness_side="not_q",
                    error="eval failed",
                )
            ],
            witness_index=0,
            witness_side="not_q",
            num_unverifiable=1,
        ),
    ]

    keys = _phase4_violation_keys_for_sentences(phase4_results, {2, 3})

    assert ("mismatch", 1, 0, "not_q") not in keys
    assert ("unverifiable", 2, 0, "not_q") in keys
    assert ("mismatch", 3, 0, "not_q") in keys


def test_merge_witness_bank_deduplicates_identical_domains() -> None:
    phase2_a = Phase2Result(
        domain={"entities": ["alice"], "sorts": {"Entity": ["alice"]}, "predicates": {}},
        domain_desc_str="",
        model=None,
        witness_side="not_q",
    )
    phase2_b = Phase2Result(
        domain={"entities": ["alice"], "sorts": {"Entity": ["alice"]}, "predicates": {}},
        domain_desc_str="",
        model=None,
        witness_side="not_q",
    )
    phase2_c = Phase2Result(
        domain={"entities": ["bob"], "sorts": {"Entity": ["bob"]}, "predicates": {}},
        domain_desc_str="",
        model=None,
        witness_side="q",
    )
    dummy_phase4 = Phase4Result(
        mismatches=[],
        all_passed=True,
        evaluations=[],
        witness_index=0,
        witness_side="not_q",
    )
    witness_results = [
        WitnessCheckResult(
            witness_index=0,
            phase2=phase2_b,
            phase3=Phase3Result(grounded=[]),
            phase4=dummy_phase4,
        ),
        WitnessCheckResult(
            witness_index=1,
            phase2=phase2_c,
            phase3=Phase3Result(grounded=[]),
            phase4=dummy_phase4,
        ),
    ]

    merged = _merge_witness_bank([phase2_a], witness_results)

    assert len(merged) == 2
    assert merged[0].domain == phase2_a.domain
    assert merged[1].domain == phase2_c.domain


def test_open_issues_are_refreshed_from_history_bank() -> None:
    mismatch = Mismatch(
        sentence_index=0,
        nl_sentence="s0",
        mismatch_type="weakening",
        fol_truth=True,
        grounded_truth=False,
        fol_formula_str="p(a)",
        grounded_formula='truth["p(a)"]',
    )
    phase4_a = Phase4Result(
        mismatches=[mismatch],
        all_passed=False,
        evaluations=[],
        witness_index=0,
        witness_side="not_q",
    )
    phase4_b = Phase4Result(
        mismatches=[
            Mismatch(
                sentence_index=0,
                nl_sentence="s0 later",
                mismatch_type="strengthening",
                fol_truth=False,
                grounded_truth=True,
                fol_formula_str="q(a)",
                grounded_formula='truth["q(a)"]',
            )
        ],
        all_passed=False,
        evaluations=[],
        witness_index=1,
        witness_side="q",
    )
    history = [
        WitnessCheckResult(
            witness_index=0,
            phase2=Phase2Result(
                domain={"entities": ["alice"], "sorts": {"Entity": ["alice"]}, "predicates": {}},
                domain_desc_str="",
                model=None,
                witness_side="not_q",
            ),
            phase3=Phase3Result(grounded=[]),
            phase4=phase4_a,
        ),
        WitnessCheckResult(
            witness_index=1,
            phase2=Phase2Result(
                domain={"entities": ["bob"], "sorts": {"Entity": ["bob"]}, "predicates": {}},
                domain_desc_str="",
                model=None,
                witness_side="q",
            ),
            phase3=Phase3Result(grounded=[]),
            phase4=phase4_b,
        ),
    ]

    issues = _open_issues_from_history_witness_results(
        history,
        premises=[z3.BoolVal(True)],
        q=z3.BoolVal(True),
    )

    assert list(issues) == [0]
    assert issues[0][0].nl_sentence == "s0"
    assert issues[0][3]["entities"] == ["alice"]


def test_compute_gap_analysis_marks_pure_missing_link_as_non_bridgeable() -> None:
    entity = z3.DeclareSort("Entity")
    a = z3.Const("a", entity)
    x = z3.Const("x", entity)
    p = z3.Function("p", entity, z3.BoolSort())
    q = z3.Function("q", entity, z3.BoolSort())

    premises = [
        z3.ForAll([x], z3.Implies(p(x), q(x))),
        q(a),
    ]
    gap = compute_gap_analysis(premises, q(a), mismatches=[], background_constraints=[])

    assert gap.missing_links == [("q", "p")]
    assert gap.bridgeable is False
    assert gap.non_bridgeable_reason is not None


def test_compute_gap_analysis_allows_bridge_when_mismatch_localizes_gap() -> None:
    entity = z3.DeclareSort("Entity")
    a = z3.Const("a", entity)
    x = z3.Const("x", entity)
    p = z3.Function("p", entity, z3.BoolSort())
    q = z3.Function("q", entity, z3.BoolSort())

    premises = [
        z3.ForAll([x], z3.Implies(p(x), q(x))),
        q(a),
    ]
    mismatch = Mismatch(
        sentence_index=0,
        nl_sentence="If p then q.",
        mismatch_type="weakening",
        fol_truth=True,
        grounded_truth=False,
        fol_formula_str=str(premises[0]),
        grounded_formula='truth["q(a)"]',
    )
    gap = compute_gap_analysis(premises, q(a), mismatches=[mismatch], background_constraints=[])

    assert gap.missing_links == [("q", "p")]
    assert gap.bridgeable is True
    assert gap.non_bridgeable_reason is None


def test_phase5_to_dict_exposes_bridge_only_and_seed_witness_aliases() -> None:
    repair = RepairEntry(
        sentence_index=0,
        mismatch_type="weakening",
        original_formula_str="p(a)",
        grounded_formula='truth["p(a)"]',
        fol_truth_before=True,
        grounded_truth_expected=False,
        repaired_expr_str="Not(p(a))",
        repaired_formula=z3.BoolVal(True),
        success=True,
        local_validated=True,
    )
    bridge_payload = _phase5_to_dict(
        Phase5Result(
            repaired_premises=[],
            repaired_q=z3.BoolVal(True),
            repairs=[],
            all_repaired=False,
            bridge_axioms=[z3.BoolVal(True)],
        ),
        accepted_bridges=[z3.BoolVal(True)],
    )
    repair_payload = _phase5_to_dict(
        Phase5Result(
            repaired_premises=[z3.BoolVal(True)],
            repaired_q=z3.BoolVal(True),
            repairs=[repair],
            all_repaired=True,
            num_local_validated=1,
        )
    )

    assert bridge_payload["bridge_only"] is True
    assert bridge_payload["num_formula_repairs"] == 0
    assert bridge_payload["num_seed_witness_validated"] == 0
    assert repair_payload["bridge_only"] is False
    assert repair_payload["num_formula_repairs"] == 1
    assert repair_payload["num_seed_witness_validated"] == 1
    assert repair_payload["repairs"][0]["seed_witness_validated"] is True
