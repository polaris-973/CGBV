import z3

from cgbv.core.phase4_check import Mismatch, Phase4Result, SentenceEval
from cgbv.core.pipeline import (
    _carried_actionable_violation_keys,
    _phase4_violation_keys,
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


def test_carried_actionable_violation_keys_track_persisting_issue() -> None:
    person = z3.DeclareSort("Person")
    alice = z3.Const("alice", person)
    is_ready = z3.Function("is_ready", person, z3.BoolSort())

    s = z3.Solver()
    s.add(z3.Not(is_ready(alice)))
    assert s.check() == z3.sat
    model = s.model()

    mismatch = Mismatch(
        sentence_index=0,
        nl_sentence="Alice is ready.",
        mismatch_type="weakening",
        fol_truth=False,
        grounded_truth=True,
        fol_formula_str="is_ready(alice)",
        grounded_formula='truth["is_ready(alice)"]',
    )
    namespace = {"alice": alice, "is_ready": is_ready}

    still_broken = _carried_actionable_violation_keys(
        actionable_mismatches=[mismatch],
        current_mismatch_indices=set(),
        premises=[is_ready(alice)],
        q=z3.BoolVal(True),
        namespace=namespace,
        carried_mismatch_models={0: model},
    )
    resolved = _carried_actionable_violation_keys(
        actionable_mismatches=[mismatch],
        current_mismatch_indices=set(),
        premises=[z3.Not(is_ready(alice))],
        q=z3.BoolVal(True),
        namespace=namespace,
        carried_mismatch_models={0: model},
    )

    assert ("carried", 0) in still_broken
    assert ("carried", 0) not in resolved
