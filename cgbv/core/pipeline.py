from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cgbv.logging as cgbv_log
import z3 as _z3_module  # for premise consistency SAT check (Fix C)
from cgbv.config.settings import ExperimentConfig
from cgbv.core.gap_analysis import compute_gap_analysis
from cgbv.core.multi_witness import MultiWitnessResult, WitnessCheckResult, run_multi_witness
from cgbv.core.phase2_witness import Phase2Result
from cgbv.core.phase1_formalize import Phase1Result, run_phase1, run_phase1_targeted
from cgbv.core.phase3_grounded import GroundedFormula, Phase3Result, retemplate_with_hint
from cgbv.core.semantic_stability import SemanticAuditResult, audit_semantic_stability
from cgbv.solver.model_extractor import format_domain_schema
from cgbv.solver.code_executor import configure_max_workers
from cgbv.core.phase4_check import Mismatch, Phase4Result, run_phase4
from cgbv.core.phase5_repair import Phase5Result, run_phase5
from cgbv.data.base import DataSample
from cgbv.llm.base import LLMClient
from cgbv.llm.factory import create_llm_client
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.finite_evaluator import FiniteModelEvaluator
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNKNOWN, VERDICT_UNCERTAIN

logger = logging.getLogger(__name__)
_finite_evaluator = FiniteModelEvaluator()


def _formula_str(idx: int, premises: list, q: object) -> str:
    """
    Return a stable string fingerprint for the formula at sentence *idx*.
    Used in issue/repair traces and compatibility views derived from the
    current theory state.
    """
    n = len(premises)
    if idx < n:
        return str(premises[idx])
    if idx == n:
        return str(q)
    return ""  # should not occur


@dataclass
class RoundRecord:
    """Record of one repair round (Phase 2+3+4+5)."""
    round_num: int                          # 1-indexed
    num_witnesses: int
    mismatches: list[dict]                  # serialisable mismatch records (current witness only)
    all_passed: bool
    repair_attempted: bool
    repair_success: bool
    verdict_before: str | None = None
    verdict_candidate: str | None = None
    verdict_after: str | None = None
    repair_reverted: bool = False           # P0.4: repair was reverted due to verdict regression
    repair_local_validated: int = 0        # Seed-witness local acceptances; not a cross-witness guarantee
    repair_seed_witness_validated: int = 0 # Alias for observability; same semantics as repair_local_validated
    num_mismatches: int = 0               # total mismatches sent to Phase 5
    num_phase3_detected: int = 0          # structural Phase 3 errors detected this round (pre-reground)
    num_phase3_errors: int = 0            # Phase 3 errors remaining after targeted re-grounding
    num_phase3_reground_success: int = 0  # Phase 3 errors resolved via targeted re-grounding
    carried_issues: list[dict] = field(default_factory=list)  # persisted from previous rounds
    num_carried: int = 0                   # count of carried issues injected this round


@dataclass
class PipelineResult:
    sample_id: str
    dataset: str
    label: str                              # ground truth
    verdict: str | None                     # final solver verdict (after all repairs); None when execution failed
    verdict_pre_bridge: str | None          # Phase 1 verdict BEFORE Phase 1.5 bridge repair; None on exec failure
    verified: bool                          # True if no mismatch found in any witness
    num_rounds: int
    verdict_post_bridge: str | None = None  # verdict AFTER Phase 1.5 bridge repair, before repair loop
    rounds: list[RoundRecord] = field(default_factory=list)
    phase1_raw_code: str = ""
    phase1_repeated_failure: bool = False
    acceptance_state: str = "needs_repair"  # accepted | needs_repair | failed
    diagnostic_tags: list[str] = field(default_factory=list)
    semantic_stable: bool | None = None
    initial_obligation_count: int = 0
    final_obligation_count: int = 0
    # Execution/verification status (separate concepts — see DESIGN NOTES below)
    #
    # execution_status: did the pipeline produce a valid semantic verdict?
    #   "success"        — Phase 1 produced a verdict (Entailed/Refuted/Uncertain)
    #   "phase1_error"   — Phase 1 failed (code error, all retries exhausted)
    #   "solver_unknown" — Phase 1 code ran but Z3 returned Unknown (timeout/undecidable)
    #   "pipeline_error" — unexpected exception that prevented pipeline completion
    #
    # verification_status: what did the CGBV verification chain conclude?
    #   (only meaningful when execution_status == "success")
    #   "verified"         — all witnesses clean, no mismatches, no carried issues
    #   "exhausted_rounds" — mismatches remained after R_max repair rounds
    #   "witness_failed"   — Phase 2 witness construction failed in the final round
    #   "semantic_unstable" — witness bank is clean but independent locked-symbol audit
    #                         still disagrees on a query-relevant formula
    #   "not_run"          — verification never started (execution failed before it)
    execution_status: str = "success"
    verification_status: str = "not_run"
    # P6: Verification confidence grading
    #   "high"   — verified with ≥1 witness, no mismatches, no open issues
    #   "medium" — all repairs committed, no regressions, but unverified
    #   "low"    — some repairs failed or reverted, or open issues remain
    #   "none"   — no verification attempted (execution failed or no witnesses)
    verification_confidence: str = "none"
    error: str | None = None


class CGBVPipeline:
    """
    Main CGBV pipeline controller.

    Orchestrates Phase 1 → R_max rounds of (Multi-Witness Phase 2+3+4 → Phase 5) loops.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm: LLMClient = create_llm_client(config.llm)
        self.solver = Z3Solver(timeout_ms=config.pipeline.solver_timeout * 1000)
        self.prompt_engine = PromptEngine(
            templates_dir=config.prompts.templates_dir,
            few_shot_dir=config.prompts.few_shot_dir,
        )
        self._results_base = config.output_dir
        configure_max_workers(config.pipeline.max_exec_workers)

    async def run(self, sample: DataSample) -> PipelineResult:
        """
        Run the full CGBV pipeline on one sample.

        Algorithm:
        1. Phase 1: NL → FOL → solver verdict (with Distinct constraints)
        2. for round in 1..R_max:
             Phase 2+3+4: Multi-Witness (¬q side + q side for Uncertain)
             if no mismatch: return verified
             Phase 5: repair with local acceptance check
             re-solve; if verdict regressed and was initially correct → revert (P0.4)
        3. return unverified (exhausted R_max rounds)
        """
        out_dir = self._results_base / sample.dataset / sample.id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Phase 1: Formalize & Solve
        # ----------------------------------------------------------------
        cgbv_log.update_phase("phase1")
        p1 = await run_phase1(
            premises_nl=sample.premises,
            conclusion_nl=sample.conclusion,
            llm=self.llm,
            solver=self.solver,
            prompt_engine=self.prompt_engine,
            dataset=sample.dataset,
            max_retries=self.config.pipeline.formalize_retries,
            task_type=sample.task_type,
            code_exec_timeout=self.config.pipeline.code_exec_timeout,
            world_assumption=self.config.pipeline.world_assumption,
            bridge_retries=self.config.pipeline.bridge_retries,
            enable_bridge=self.config.pipeline.enable_phase1_bridge,
        )
        self._write_json(out_dir / "phase1.json", _phase1_to_dict(p1, sample))

        if p1.error or p1.q is None:
            result = PipelineResult(
                sample_id=sample.id,
                dataset=sample.dataset,
                label=sample.label,
                verdict=None,
                verdict_pre_bridge=None,
                verdict_post_bridge=None,
                verified=False,
                num_rounds=0,
                phase1_raw_code=p1.raw_code,
                phase1_repeated_failure=p1.repeated_failure,
                acceptance_state="failed",
                diagnostic_tags=["compile_error"],
                semantic_stable=None,
                initial_obligation_count=0,
                final_obligation_count=0,
                execution_status="phase1_error",
                verification_status="not_run",
                error=p1.error or "Phase 1 produced no formula",
            )
            self._write_json(out_dir / "result.json", asdict(result))
            return result

        # If Phase 1 ran but the solver returned Unknown (timeout / undecidable),
        # there is no meaningful semantic verdict to verify or repair.  Treat this
        # as a distinct execution failure rather than letting it silently enter the
        # multi-witness loop and emerge as execution_status="success".
        if p1.verdict == VERDICT_UNKNOWN:
            result = PipelineResult(
                sample_id=sample.id,
                dataset=sample.dataset,
                label=sample.label,
                verdict=None,
                verdict_pre_bridge=None,
                verdict_post_bridge=None,
                verified=False,
                num_rounds=0,
                phase1_raw_code=p1.raw_code,
                phase1_repeated_failure=p1.repeated_failure,
                acceptance_state="failed",
                diagnostic_tags=["solver_unknown"],
                semantic_stable=None,
                initial_obligation_count=0,
                final_obligation_count=0,
                execution_status="solver_unknown",
                verification_status="not_run",
                error="Phase 1 solver returned Unknown (timeout or undecidable formula)",
            )
            self._write_json(out_dir / "result.json", asdict(result))
            return result

        # Working copies
        premises = list(p1.premises)               # NL-only formulas
        background_constraints = list(p1.background_constraints)
        bound_var_names = p1.bound_var_names       # P0.1: ForAll/Exists variable names
        q = p1.q
        verdict = p1.verdict
        # Three-level verdict separation:
        #   verdict_pre_bridge  = raw Phase 1 output BEFORE Phase 1.5 → used by bridge audit
        #   verdict_post_bridge = after Phase 1.5, before repair loop → regression guard baseline
        #   verdict (final)     = updated by Phase 5 repairs
        verdict_pre_bridge = p1.verdict_pre_bridge or p1.verdict  # snapshot; never updated
        verdict_post_bridge = verdict                              # snapshot; never updated
        namespace = p1.namespace
        model_info = p1.model_info
        model_info_q = p1.model_info_q             # P1.1: q-side model for Uncertain
        initial_gap = compute_gap_analysis(premises, q, [], background_constraints)
        initial_obligation_count = initial_gap.obligation_count

        sentences = sample.premises + [sample.conclusion]
        rounds: list[RoundRecord] = []

        # Tracks the verification chain outcome for the final round.
        # Overwritten each round; the last value is used in the final PipelineResult.
        _final_verification_status = "exhausted_rounds"

        # Derived compatibility view: one representative mismatch context per
        # sentence, rebuilt from the authoritative history witness bank every round.
        # Value: (Mismatch, formula_fingerprint, model, domain)
        open_issues: dict[int, tuple] = {}
        issue_round_counts: dict[int, int] = {}

        # Template cache: reuse Phase 3 templates across rounds.
        # Only sentences whose FOL was repaired need regeneration.
        # Cleared entirely on theory rewrite (run_phase1_targeted).
        template_cache: list | None = None  # list[GroundingTemplate] | None
        regenerate_indices: set[int] | None = None  # filled from Phase 5 repairs

        # History witness bank: stores distinct Phase 2 witness worlds across rounds.
        # We do NOT persist old Phase 3/4 outputs here because templates may evolve;
        # instead we re-run Phase 4 on the bank using the current templates.
        history_witness_bank: list[Phase2Result] = []

        # Revert counter: tracks how many times Phase 5 repairs for a sentence
        # have been reverted by the semantic audit. Used by the pre-Phase5
        # re-grounding gate (§5) to redirect to template re-generation.
        revert_counts: dict[int, int] = {}

        # ----------------------------------------------------------------
        # Repair loop (up to R_max rounds)
        # ----------------------------------------------------------------
        for round_num in range(1, self.config.pipeline.r_max + 1):
            logger.info(
                "Pipeline sample=%s round=%d/%d verdict=%s",
                sample.id, round_num, self.config.pipeline.r_max, verdict,
            )

            # Phase 2+3+4: Multi-Witness
            cgbv_log.update_phase("phase2", f"round {round_num}/{self.config.pipeline.r_max}")
            # P5: For Uncertain verdicts, use at least 2 witnesses to improve
            # verification coverage (both P∧q and P∧¬q sides need checking).
            effective_k = (
                max(self.config.pipeline.num_witnesses, self.config.pipeline.min_uncertain_witnesses)
                if verdict == VERDICT_UNCERTAIN
                else self.config.pipeline.num_witnesses
            )
            mw = await run_multi_witness(
                verdict=verdict,
                model_info=model_info,
                model_info_q=model_info_q,
                premises=premises,
                q=q,
                namespace=namespace,
                sentences=sentences,
                solver=self.solver,
                llm=self.llm,
                prompt_engine=self.prompt_engine,
                background_constraints=background_constraints,
                bound_var_names=bound_var_names,
                num_witnesses=effective_k,
                grounding_retries=self.config.pipeline.grounding_retries,
                world_assumption=self.config.pipeline.world_assumption,
                batch_grounding_size=self.config.pipeline.batch_grounding_size,
                prev_templates=template_cache,
                regenerate_indices=regenerate_indices,
            )
            # Update template cache from this round's templates
            template_cache = mw.templates if mw.templates else None
            regenerate_indices = None  # reset for next round
            self._write_json(
                out_dir / f"round{round_num}_witness.json",
                _mw_to_dict(mw),
            )
            for wr in mw.witness_results:
                k = wr.witness_index
                self._write_json(
                    out_dir / f"round{round_num}_w{k}_phase3.json",
                    _phase3_to_dict(wr.phase3),
                )
                self._write_json(
                    out_dir / f"round{round_num}_w{k}_phase4.json",
                    _phase4_to_dict(wr.phase4),
                )

            history_witness_bank = _merge_witness_bank(history_witness_bank, mw.witness_results)
            history_witness_results = _rerun_phase4_on_witness_bank(
                history_witness_bank,
                template_cache or [],
                sentences,
                premises,
                q,
                namespace,
                self.solver,
            )
            history_violation_keys = _phase4_violation_keys(
                [wr.phase4 for wr in history_witness_results]
            )
            prev_issue_round_counts = dict(issue_round_counts)
            open_issues = _open_issues_from_history_witness_results(
                history_witness_results,
                premises,
                q,
            )
            issue_round_counts = {
                idx: (
                    prev_issue_round_counts[idx] + 1
                    if idx in prev_issue_round_counts else 0
                )
                for idx in open_issues
            }

            # --- Compute carried issues ---
            # Issues in the history-derived compatibility view that the current
            # witness did NOT catch.
            # These are distinct from current witness mismatches and logged
            # separately so the log accurately reflects what each witness saw.
            current_mismatch_indices = {m.sentence_index for m in mw.mismatches}
            carried: list[Mismatch] = [
                issue[0] for idx, issue in open_issues.items()
                if idx not in current_mismatch_indices
            ]

            for m in mw.mismatches:
                m.persist_rounds = issue_round_counts.get(m.sentence_index, 0)

            # Build per-witness model/domain maps for this round.
            witness_models: dict[int, Any] = {
                wr.witness_index: wr.phase2.model
                for wr in mw.witness_results
                if wr.phase2.model is not None
            }
            witness_domains: dict[int, dict] = {
                wr.witness_index: wr.phase2.domain
                for wr in mw.witness_results
                if wr.phase2.domain is not None
            }
            # Per-sentence-index model/domain for carried issues.
            # Keyed by sentence_index (NOT witness_index) to avoid the collision
            # where witness_index is renumbered 0..N-1 each round — a carried
            # issue with witness_index=2 from round k would be silently replaced
            # by round k+1's entirely different witness 2 under a witness_index
            # keyed map.  Phase 5 uses these as primary lookups (before falling
            # back to the per-witness-index maps for current-round mismatches).
            carried_mismatch_models: dict[int, Any] = {
                idx: issue[2]
                for idx, issue in open_issues.items()
                if idx not in current_mismatch_indices
                and len(issue) >= 3 and issue[2] is not None
            }
            carried_mismatch_domains: dict[int, dict] = {
                idx: issue[3]
                for idx, issue in open_issues.items()
                if idx not in current_mismatch_indices
                and len(issue) >= 4 and issue[3] is not None
            }

            round_record = RoundRecord(
                round_num=round_num,
                num_witnesses=mw.num_witnesses_constructed,
                mismatches=[_mismatch_to_dict(m) for m in mw.mismatches],
                all_passed=mw.all_passed,
                repair_attempted=False,
                repair_success=False,
                verdict_before=verdict,
                verdict_after=verdict,
                num_mismatches=len(mw.mismatches),
                carried_issues=[_mismatch_to_dict(m) for m in carried],
                num_carried=len(carried),
            )

            # --- Verified check ---
            # Verification is driven by the accumulated witness bank, not just
            # the current round's de-duplicated mismatch summary.  This closes
            # the branch where a sample could be accepted even though an older
            # or sibling witness remained unverifiable.
            #
            # Gap analysis routing: even with 0 mismatches, if the theory has
            # structural gaps (ungrounded rule antecedents with missing links),
            # we defer verification and enter Phase 5 bridge-only mode.  This
            # catches cases like "bee → animal" where the FOL is structurally
            # incomplete but Phase 3-4 can't detect it (correlated LLM bias).
            if history_witness_results and not history_violation_keys:
                # Gap analysis only needed for Uncertain: guards against correlated
                # LLM bias on incomplete theories.  For Refuted/Entailed the solver
                # already has a definitive proof — no structural gap check required.
                gap = (
                    compute_gap_analysis(premises, q, [], background_constraints)
                    if verdict == VERDICT_UNCERTAIN
                    else None
                )
                if gap is None or not gap.missing_links:
                    action, semantic_audit, p1_new = await self._try_verify_via_audit(
                        sentences=sentences, premises=premises, q=q,
                        namespace=namespace, raw_code=p1.raw_code,
                        witness_results=history_witness_results,
                        sample=sample, round_num=round_num, out_dir=out_dir,
                    )
                    if action == "failed":
                        rounds.append(round_record)
                        final_gap = compute_gap_analysis(premises, q, [], background_constraints)
                        result = PipelineResult(
                            sample_id=sample.id,
                            dataset=sample.dataset,
                            label=sample.label,
                            verdict=verdict,
                            verdict_pre_bridge=verdict_pre_bridge,
                            verdict_post_bridge=verdict_post_bridge,
                            verified=False,
                            num_rounds=round_num,
                            rounds=rounds,
                            phase1_raw_code=p1.raw_code,
                            phase1_repeated_failure=p1.repeated_failure,
                            acceptance_state="needs_repair",
                            diagnostic_tags=["semantic_unstable"],
                            semantic_stable=False,
                            initial_obligation_count=initial_obligation_count,
                            final_obligation_count=final_gap.obligation_count,
                            execution_status="success",
                            verification_status="semantic_unstable",
                            verification_confidence="low",
                            error="Semantic stability audit rejected an otherwise clean witness bank.",
                        )
                        self._write_json(out_dir / "result.json", asdict(result))
                        return result

                    if action == "reformalized":
                        premises = list(p1_new.premises)
                        q = p1_new.q
                        background_constraints = list(p1_new.background_constraints)
                        bound_var_names = p1_new.bound_var_names
                        namespace = p1_new.namespace
                        verdict = p1_new.verdict
                        model_info = p1_new.model_info
                        model_info_q = p1_new.model_info_q
                        p1 = p1_new
                        template_cache = None  # theory rewrite → invalidate all templates
                        revert_counts.clear()  # stale revert history invalid after rewrite
                        history_witness_bank.clear()
                        round_record.verdict_after = verdict
                        rounds.append(round_record)
                        continue

                    # action == "verified"
                    final_gap = compute_gap_analysis(premises, q, [], background_constraints)
                    rounds.append(round_record)
                    result = PipelineResult(
                        sample_id=sample.id,
                        dataset=sample.dataset,
                        label=sample.label,
                        verdict=verdict,
                        verdict_pre_bridge=verdict_pre_bridge,
                        verdict_post_bridge=verdict_post_bridge,
                        verified=True,
                        num_rounds=round_num,
                        rounds=rounds,
                        phase1_raw_code=p1.raw_code,
                        phase1_repeated_failure=p1.repeated_failure,
                        acceptance_state="accepted",
                        diagnostic_tags=[],
                        semantic_stable=True,
                        initial_obligation_count=initial_obligation_count,
                        final_obligation_count=final_gap.obligation_count,
                        execution_status="success",
                        verification_status="verified",
                        verification_confidence="high",
                    )
                    self._write_json(out_dir / "result.json", asdict(result))
                    logger.info("Pipeline sample=%s VERIFIED at round %d", sample.id, round_num)
                    return result

                if not gap.bridgeable:
                    p1_gap = await self._try_reformalize_non_bridgeable_gap(
                        sentences=sentences,
                        premises=premises,
                        q=q,
                        raw_code=p1.raw_code,
                        gap=gap,
                        sample=sample,
                        round_num=round_num,
                    )
                    if p1_gap is not None and p1_gap.verdict != VERDICT_UNKNOWN:
                        premises = list(p1_gap.premises)
                        q = p1_gap.q
                        background_constraints = list(p1_gap.background_constraints)
                        bound_var_names = p1_gap.bound_var_names
                        namespace = p1_gap.namespace
                        verdict = p1_gap.verdict
                        model_info = p1_gap.model_info
                        model_info_q = p1_gap.model_info_q
                        p1 = p1_gap
                        template_cache = None
                        revert_counts.clear()
                        history_witness_bank.clear()
                        round_record.verdict_after = verdict
                    else:
                        logger.warning(
                            "Pipeline sample=%s round=%d: non-bridgeable gap re-formalization "
                            "failed; leaving theory unchanged",
                            sample.id, round_num,
                        )
                    rounds.append(round_record)
                    continue

                # Gap analysis found structural gaps — enter bridge-only Phase 5
                logger.info(
                    "Pipeline sample=%s round=%d: 0 mismatches but gap analysis "
                    "found %d missing link(s): %s — entering bridge-only Phase 5",
                    sample.id, round_num, len(gap.missing_links), gap.missing_links,
                )
                first_wr = mw.witness_results[0] if mw.witness_results else None
                first_witness_domain = first_wr.phase2.domain if first_wr else {}

                cgbv_log.update_phase("phase5", f"bridge-only round {round_num}")
                p5_gap = await run_phase5(
                    mismatches=[],
                    premises=premises,
                    q=q,
                    namespace=namespace,
                    raw_code=p1.raw_code,
                    domain=first_witness_domain,
                    llm=self.llm,
                    prompt_engine=self.prompt_engine,
                    max_retries=self.config.pipeline.repair_retries,
                    models=witness_models,
                    domains=witness_domains,
                    solver=self.solver,
                    world_assumption=self.config.pipeline.world_assumption,
                    gap_analysis=gap,
                    sparse_witness_format=self.config.pipeline.sparse_witness_format,
                )

                # Process bridge axioms from gap-triggered Phase 5
                _gap_committed: list = []
                if p5_gap.bridge_axioms:
                    # Filter out bridges that would make the premise set UNSAT
                    # (ex falso quodlibet — an inconsistent bridge lets Z3 prove
                    # any conclusion vacuously, corrupting the verdict).
                    accepted_gap_bridges = _filter_obligation_reducing_bridges(
                        bridges=list(p5_gap.bridge_axioms),
                        premises=list(premises),
                        q=q,
                        mismatches=[],
                        background_constraints=list(background_constraints),
                    )
                    if accepted_gap_bridges:
                        background_constraints.extend(accepted_gap_bridges)

                        solver_premises = list(premises) + background_constraints
                        if sample.task_type == "three_class":
                            new_verdict, new_model_info, new_model_info_q = (
                                self.solver.check_entailment_three_class(solver_premises, q)
                            )
                        else:
                            new_verdict, new_model_info = self.solver.check_entailment(
                                solver_premises, q
                            )
                            new_model_info_q = None

                        # Unknown guard
                        new_is_unknown = (new_verdict == VERDICT_UNKNOWN)
                        old_is_unknown = (verdict == VERDICT_UNKNOWN)
                        reject_reason: str | None = None
                        if new_is_unknown and not old_is_unknown:
                            reject_reason = "gap bridge degraded solver to Unknown"
                        else:
                            new_history_violation_keys = _bridge_violation_keys_on_witness_bank(
                                bridges=accepted_gap_bridges,
                                witness_bank=history_witness_bank,
                                templates=template_cache or [],
                                sentences=sentences,
                                premises=premises,
                                q=q,
                                namespace=namespace,
                                solver=self.solver,
                            )
                            if new_history_violation_keys:
                                reject_reason = (
                                    "gap bridge introduced unresolved witness-bank "
                                    f"violations ({len(new_history_violation_keys)})"
                                )
                        if reject_reason is not None:
                            for _ in accepted_gap_bridges:
                                if background_constraints:
                                    background_constraints.pop()
                            logger.warning(
                                "Pipeline sample=%s round=%d: %s; reverted",
                                sample.id, round_num, reject_reason,
                            )
                        else:
                            _gap_committed = accepted_gap_bridges
                            old_verdict_for_log = verdict
                            verdict = new_verdict
                            model_info = new_model_info
                            model_info_q = new_model_info_q
                            round_record.verdict_after = new_verdict
                            round_record.repair_success = True

                            bridge_notes = [
                                f"# Gap bridge axiom: {str(b)}"
                                for b in accepted_gap_bridges
                            ]
                            updated_code = (
                                p1.raw_code.rstrip()
                                + "\n\n# ---- Gap analysis bridges (round "
                                + str(round_num) + ") ----\n"
                                + "\n".join(bridge_notes)
                            )
                            p1 = _patched_p1(
                                p1, premises, q, new_verdict, new_model_info,
                                new_model_info_q, raw_code=updated_code,
                                background_constraints=list(background_constraints),
                            )
                            logger.info(
                                "Pipeline sample=%s round=%d: %d gap bridge(s) "
                                "committed, verdict %s→%s",
                                sample.id, round_num,
                                len(accepted_gap_bridges), old_verdict_for_log, verdict,
                            )
                    else:
                        logger.info(
                            "Pipeline sample=%s round=%d: no gap bridge reduced obligations; no commit",
                            sample.id, round_num,
                        )

                self._write_json(
                    out_dir / f"round{round_num}_repair.json",
                    _phase5_to_dict(p5_gap, accepted_bridges=_gap_committed),
                )
                rounds.append(round_record)
                # Continue to next round for verification of the new theory
                continue

            if mw.num_witnesses_constructed == 0:
                logger.warning(
                    "Pipeline sample=%s round=%d: all witness constructions failed",
                    sample.id, round_num,
                )
                _final_verification_status = "witness_failed"
                rounds.append(round_record)
                continue

            _final_verification_status = "exhausted_rounds"

            # Effective mismatches for repair = current witness + carried issues.
            effective_mismatches: list[Mismatch] = list(mw.mismatches) + carried

            # Conclusion mismatches: is_phase3_error is a HINT (not a definitive
            # attribution).  We try retemplate first; unresolved conclusion mismatches
            # are escalated to Phase 5 / run_phase1_targeted — never dead-ended.
            conclusion_mismatches: list[Mismatch] = [m for m in effective_mismatches if m.is_phase3_error]
            actionable_mismatches: list[Mismatch] = [m for m in effective_mismatches if not m.is_phase3_error]
            # num_mismatches tracks all mismatches for logging purposes.
            round_record.num_mismatches = len(effective_mismatches)

            # --- Targeted Phase 3 re-grounding for conclusion mismatches ---
            # For each detected conclusion mismatch, attempt a targeted re-grounding
            # using a NEUTRAL hint (no target truth value — symmetric distrust).
            # Unresolved conclusion mismatches are escalated to Phase 5 / theory rewrite.
            round_record.num_phase3_detected = len(conclusion_mismatches)

            if conclusion_mismatches:
                logger.warning(
                    "Pipeline sample=%s round=%d: %d conclusion mismatch(es) "
                    "detected — attempting neutral re-grounding before escalation: %s",
                    sample.id, round_num, len(conclusion_mismatches),
                    [(m.witness_side, m.mismatch_type, m.grounded_formula[:40])
                     for m in conclusion_mismatches],
                )
                cgbv_log.update_phase("phase3", f"re-grounding round {round_num}")
                still_broken: list[Mismatch] = []
                reground_records: list[dict] = []
                for m in conclusion_mismatches:
                    expected_truth = (m.witness_side == "q")  # q→True, not_q→False
                    idx_key = m.sentence_index
                    if idx_key in open_issues and idx_key not in current_mismatch_indices:
                        stored = open_issues[idx_key]
                        witness_domain = stored[3] if len(stored) >= 4 else None
                    else:
                        witness_domain = witness_domains.get(m.witness_index)
                    if witness_domain is None:
                        still_broken.append(m)
                        continue
                    # NEUTRAL hint: do NOT specify target truth value (symmetric distrust).
                    # The template or the FOL could be wrong — let the LLM re-derive
                    # purely from the NL sentence meaning.
                    hint = (
                        f"Your previous template `{m.grounded_formula}` disagrees with "
                        f"the FOL formula on a boundary witness. One of them may be wrong. "
                        f"Re-derive the template purely from the natural language meaning: "
                        f"'{sentences[m.sentence_index]}'. "
                        f"Do NOT try to match any specific truth value."
                    )
                    schema_str = format_domain_schema(witness_domain)
                    new_tmpl = await retemplate_with_hint(
                        idx=m.sentence_index,
                        sentence=sentences[m.sentence_index],
                        domain_schema_str=schema_str,
                        domain=witness_domain,
                        current_template=m.grounded_formula,
                        hint=hint,
                        llm=self.llm,
                        prompt_engine=self.prompt_engine,
                        max_retries=self.config.pipeline.grounding_retries,
                        world_assumption=self.config.pipeline.world_assumption,
                        solver=self.solver,
                    )
                    resolved = False
                    if not new_tmpl.failed:
                        new_truth = self.solver.evaluate_grounded_formula(
                            witness_domain, new_tmpl.template_code
                        )
                        if new_truth == expected_truth:
                            logger.info(
                                "Pipeline sample=%s round=%d: re-templating resolved "
                                "sentence %d (%s→%s)",
                                sample.id, round_num, m.sentence_index,
                                m.grounded_formula[:40], new_tmpl.template_code[:40],
                            )
                            resolved = True
                            # Update template cache with corrected template
                            if template_cache is not None and m.sentence_index < len(template_cache):
                                template_cache[m.sentence_index] = new_tmpl
                    reground_records.append({
                        "sentence_index": m.sentence_index,
                        "witness_side": m.witness_side,
                        "old_formula": m.grounded_formula,
                        "new_formula": new_tmpl.template_code if not new_tmpl.failed else "",
                        "resolved": resolved,
                        "error": new_tmpl.error if new_tmpl.failed else None,
                    })
                    if not resolved:
                        still_broken.append(m)

                resolved_count = len(conclusion_mismatches) - len(still_broken)
                round_record.num_phase3_reground_success = resolved_count
                conclusion_mismatches = still_broken
                self._write_json(
                    out_dir / f"round{round_num}_phase3_reground.json",
                    {"resolved": resolved_count, "attempts": reground_records},
                )

            round_record.num_phase3_errors = len(conclusion_mismatches)
            if conclusion_mismatches:
                # Split: mismatches on sentence indices < n go to Phase 5 as usual.
                # Mismatches on the conclusion index (== n) are routed directly to
                # run_phase1_targeted — Phase 5 lacks the NL context to correctly
                # repair q and should never modify the conclusion formula.
                n_sentences = len(sentences) - 1  # n = number of premises
                final_q_mismatches = [m for m in conclusion_mismatches if m.sentence_index >= n_sentences]
                premise_conclusion_mismatches = [m for m in conclusion_mismatches if m.sentence_index < n_sentences]

                if premise_conclusion_mismatches:
                    logger.info(
                        "Pipeline sample=%s round=%d: %d premise-level conclusion mismatch(es) "
                        "escalated to Phase 5.",
                        sample.id, round_num, len(premise_conclusion_mismatches),
                    )
                    actionable_mismatches.extend(premise_conclusion_mismatches)

                if final_q_mismatches:
                    logger.info(
                        "Pipeline sample=%s round=%d: %d conclusion mismatch(es) on q — "
                        "routing to run_phase1_targeted (bypassing Phase 5).",
                        sample.id, round_num, len(final_q_mismatches),
                    )
                    m0 = final_q_mismatches[0]
                    conclusion_nl = sentences[-1] if sentences else ""
                    q_hint = (
                        f"The conclusion '{conclusion_nl}' has a cross-granularity mismatch: "
                        f"the current FOL formula q does not faithfully represent the NL conclusion "
                        f"in the boundary world (grounded formula: {m0.grounded_formula!r}). "
                        f"Re-formalize the conclusion formula only, preserving all premises."
                    )
                    p1_targeted = await run_phase1_targeted(
                        original_code=p1.raw_code,
                        failed_repairs=[(m0, q_hint)],
                        premises_nl=sample.premises,
                        conclusion_nl=sample.conclusion,
                        llm=self.llm,
                        solver=self.solver,
                        prompt_engine=self.prompt_engine,
                        task_type=sample.task_type,
                        code_exec_timeout=self.config.pipeline.code_exec_timeout,
                        world_assumption=self.config.pipeline.world_assumption,
                        max_retries=self.config.pipeline.formalize_retries,
                    )
                    if p1_targeted is not None and p1_targeted.verdict != VERDICT_UNKNOWN:
                        logger.info(
                            "Pipeline sample=%s round=%d: conclusion re-formalization succeeded "
                            "(verdict %s → %s).",
                            sample.id, round_num, verdict, p1_targeted.verdict,
                        )
                        p1 = p1_targeted
                        premises = list(p1.premises)
                        q = p1.q
                        background_constraints = list(p1.background_constraints)
                        bound_var_names = p1.bound_var_names
                        namespace = p1.namespace
                        verdict = p1.verdict
                        model_info = p1.model_info
                        model_info_q = p1.model_info_q
                        template_cache = None
                        revert_counts.clear()
                        history_witness_bank.clear()
                        round_record.verdict_after = verdict
                        rounds.append(round_record)
                        continue
                    else:
                        logger.warning(
                            "Pipeline sample=%s round=%d: conclusion re-formalization failed — "
                            "continuing with existing q.",
                            sample.id, round_num,
                        )

            # --- Post-reground verified check ---
            # If targeted re-grounding resolved ALL Phase 3 errors AND there are no
            # actionable mismatches or carried issues, this round is now clean.
            # We must perform this check here (after re-grounding) rather than relying
            # on the pre-reground verified check at the top of the loop, which fired
            # before Phase 3 errors were detected and resolved.
            # Also apply gap analysis routing: defer verification if structural gaps exist.
            history_witness_results = _rerun_phase4_on_witness_bank(
                history_witness_bank,
                template_cache or [],
                sentences,
                premises,
                q,
                namespace,
                self.solver,
            )
            history_violation_keys = _phase4_violation_keys(
                [wr.phase4 for wr in history_witness_results]
            )
            if history_witness_results and not history_violation_keys:
                gap_post = (
                    compute_gap_analysis(premises, q, [], background_constraints)
                    if verdict == VERDICT_UNCERTAIN
                    else None
                )
                if (gap_post is None or not gap_post.missing_links) and not conclusion_mismatches:
                    round_record.all_passed = True
                    # Drop the resolved Phase 3 entries from the mismatches snapshot —
                    # they are no longer errors and should not appear in the final record.
                    round_record.mismatches = [
                        m for m in round_record.mismatches
                        if not m.get("is_phase3_error", False)
                    ]
                    action, semantic_audit, p1_new = await self._try_verify_via_audit(
                        sentences=sentences, premises=premises, q=q,
                        namespace=namespace, raw_code=p1.raw_code,
                        witness_results=history_witness_results,
                        sample=sample, round_num=round_num, out_dir=out_dir,
                    )
                    if action == "failed":
                        rounds.append(round_record)
                        final_gap = compute_gap_analysis(premises, q, [], background_constraints)
                        result = PipelineResult(
                            sample_id=sample.id,
                            dataset=sample.dataset,
                            label=sample.label,
                            verdict=verdict,
                            verdict_pre_bridge=verdict_pre_bridge,
                            verdict_post_bridge=verdict_post_bridge,
                            verified=False,
                            num_rounds=round_num,
                            rounds=rounds,
                            phase1_raw_code=p1.raw_code,
                            phase1_repeated_failure=p1.repeated_failure,
                            acceptance_state="needs_repair",
                            diagnostic_tags=["semantic_unstable"],
                            semantic_stable=False,
                            initial_obligation_count=initial_obligation_count,
                            final_obligation_count=final_gap.obligation_count,
                            execution_status="success",
                            verification_status="semantic_unstable",
                            verification_confidence="low",
                            error="Semantic stability audit rejected an otherwise clean witness bank.",
                        )
                        self._write_json(out_dir / "result.json", asdict(result))
                        return result

                    if action == "reformalized":
                        premises = list(p1_new.premises)
                        q = p1_new.q
                        background_constraints = list(p1_new.background_constraints)
                        bound_var_names = p1_new.bound_var_names
                        namespace = p1_new.namespace
                        verdict = p1_new.verdict
                        model_info = p1_new.model_info
                        model_info_q = p1_new.model_info_q
                        p1 = p1_new
                        template_cache = None  # theory rewrite → invalidate all templates
                        revert_counts.clear()  # stale revert history invalid after rewrite
                        history_witness_bank.clear()
                        round_record.verdict_after = verdict
                        rounds.append(round_record)
                        continue

                    # action == "verified"
                    final_gap = compute_gap_analysis(premises, q, [], background_constraints)
                    rounds.append(round_record)
                    result = PipelineResult(
                        sample_id=sample.id,
                        dataset=sample.dataset,
                        label=sample.label,
                        verdict=verdict,
                        verdict_pre_bridge=verdict_pre_bridge,
                        verdict_post_bridge=verdict_post_bridge,
                        verified=True,
                        num_rounds=round_num,
                        rounds=rounds,
                        phase1_raw_code=p1.raw_code,
                        phase1_repeated_failure=p1.repeated_failure,
                        acceptance_state="accepted",
                        diagnostic_tags=[],
                        semantic_stable=True,
                        initial_obligation_count=initial_obligation_count,
                        final_obligation_count=final_gap.obligation_count,
                        execution_status="success",
                        verification_status="verified",
                        verification_confidence="high",
                    )
                    self._write_json(out_dir / "result.json", asdict(result))
                    logger.info(
                        "Pipeline sample=%s VERIFIED at round %d "
                        "(all Phase 3 grounding errors resolved via targeted re-grounding)",
                        sample.id, round_num,
                    )
                    return result
                elif gap_post is not None and gap_post.missing_links:
                    if not gap_post.bridgeable:
                        p1_gap = await self._try_reformalize_non_bridgeable_gap(
                            sentences=sentences,
                            premises=premises,
                            q=q,
                            raw_code=p1.raw_code,
                            gap=gap_post,
                            sample=sample,
                            round_num=round_num,
                        )
                        if p1_gap is not None and p1_gap.verdict != VERDICT_UNKNOWN:
                            premises = list(p1_gap.premises)
                            q = p1_gap.q
                            background_constraints = list(p1_gap.background_constraints)
                            bound_var_names = p1_gap.bound_var_names
                            namespace = p1_gap.namespace
                            verdict = p1_gap.verdict
                            model_info = p1_gap.model_info
                            model_info_q = p1_gap.model_info_q
                            p1 = p1_gap
                            template_cache = None
                            revert_counts.clear()
                            history_witness_bank.clear()
                            round_record.verdict_after = verdict
                        else:
                            logger.warning(
                                "Pipeline sample=%s round=%d: non-bridgeable post-Phase4 gap "
                                "re-formalization failed; leaving theory unchanged",
                                sample.id, round_num,
                            )
                        rounds.append(round_record)
                        continue

                    # Structural gap found after Phase 3/4 (possibly alongside Phase 3
                    # errors on the conclusion).  Fire bridge-only Phase 5 to commit
                    # the missing link — the updated theory may resolve Phase 3 errors
                    # and change the verdict in the next round.
                    logger.info(
                        "Pipeline sample=%s round=%d: %d missing link(s) found "
                        "after Phase 3/4 — entering bridge-only Phase 5%s",
                        sample.id, round_num, len(gap_post.missing_links),
                        " (conclusion mismatches present)" if conclusion_mismatches else "",
                    )
                    first_wr = mw.witness_results[0] if mw.witness_results else None
                    first_witness_domain = first_wr.phase2.domain if first_wr else {}

                    cgbv_log.update_phase("phase5", f"bridge-only round {round_num}")
                    p5_gap2 = await run_phase5(
                        mismatches=[],
                        premises=premises,
                        q=q,
                        namespace=namespace,
                        raw_code=p1.raw_code,
                        domain=first_witness_domain,
                        llm=self.llm,
                        prompt_engine=self.prompt_engine,
                        max_retries=self.config.pipeline.repair_retries,
                        models=witness_models,
                        domains=witness_domains,
                        solver=self.solver,
                        world_assumption=self.config.pipeline.world_assumption,
                        gap_analysis=gap_post,
                        sparse_witness_format=self.config.pipeline.sparse_witness_format,
                    )

                    _gap2_committed: list = []
                    if p5_gap2.bridge_axioms:
                        accepted_gap2_bridges = _filter_obligation_reducing_bridges(
                            bridges=list(p5_gap2.bridge_axioms),
                            premises=list(premises),
                            q=q,
                            mismatches=[],
                            background_constraints=list(background_constraints),
                        )
                        if accepted_gap2_bridges:
                            background_constraints.extend(accepted_gap2_bridges)

                            solver_premises = list(premises) + background_constraints
                            if sample.task_type == "three_class":
                                new_verdict, new_model_info, new_model_info_q = (
                                    self.solver.check_entailment_three_class(solver_premises, q)
                                )
                            else:
                                new_verdict, new_model_info = self.solver.check_entailment(
                                    solver_premises, q
                                )
                                new_model_info_q = None

                            new_is_unknown = (new_verdict == VERDICT_UNKNOWN)
                            old_is_unknown = (verdict == VERDICT_UNKNOWN)
                            reject_reason: str | None = None
                            if new_is_unknown and not old_is_unknown:
                                reject_reason = "post-Phase4 gap bridge degraded solver to Unknown"
                            else:
                                new_history_violation_keys = _bridge_violation_keys_on_witness_bank(
                                    bridges=accepted_gap2_bridges,
                                    witness_bank=history_witness_bank,
                                    templates=template_cache or [],
                                    sentences=sentences,
                                    premises=premises,
                                    q=q,
                                    namespace=namespace,
                                    solver=self.solver,
                                )
                                if new_history_violation_keys:
                                    reject_reason = (
                                        "post-Phase4 gap bridge introduced unresolved witness-bank "
                                        f"violations ({len(new_history_violation_keys)})"
                                    )
                            if reject_reason is not None:
                                for _ in accepted_gap2_bridges:
                                    if background_constraints:
                                        background_constraints.pop()
                                logger.warning(
                                    "Pipeline sample=%s round=%d: %s; reverted",
                                    sample.id, round_num, reject_reason,
                                )
                            else:
                                _gap2_committed = accepted_gap2_bridges
                                old_verdict_for_log = verdict
                                verdict = new_verdict
                                model_info = new_model_info
                                model_info_q = new_model_info_q
                                round_record.verdict_after = new_verdict
                                round_record.repair_success = True

                                bridge_notes = [
                                    f"# Gap bridge axiom (post-Phase4): {str(b)}"
                                    for b in accepted_gap2_bridges
                                ]
                                updated_code = (
                                    p1.raw_code.rstrip()
                                    + "\n\n# ---- Gap analysis bridges (round "
                                    + str(round_num) + ") ----\n"
                                    + "\n".join(bridge_notes)
                                )
                                p1 = _patched_p1(
                                    p1, premises, q, new_verdict, new_model_info,
                                    new_model_info_q, raw_code=updated_code,
                                    background_constraints=list(background_constraints),
                                )
                                logger.info(
                                    "Pipeline sample=%s round=%d: %d post-Phase4 gap "
                                    "bridge(s) committed, verdict %s→%s",
                                    sample.id, round_num,
                                    len(accepted_gap2_bridges), old_verdict_for_log, verdict,
                                )
                        else:
                            logger.info(
                                "Pipeline sample=%s round=%d: no post-Phase4 gap bridge reduced obligations; no commit",
                                sample.id, round_num,
                            )

                    self._write_json(
                        out_dir / f"round{round_num}_repair.json",
                        _phase5_to_dict(p5_gap2, accepted_bridges=_gap2_committed),
                    )
                    rounds.append(round_record)
                    continue
                # else: conclusion mismatches remain but no structural gap → fall through

            # Use the first witness for repair prompt context (fallback domain)
            first_wr = mw.witness_results[0] if mw.witness_results else None
            first_witness_domain = first_wr.phase2.domain if first_wr else {}

            if actionable_mismatches:
                # -----------------------------------------------------------
                # Pre-Phase5 re-grounding gate (§5 dead-loop breaker)
                # For mismatches that were previously reverted by semantic audit,
                # try re-generating the template BEFORE sending to Phase 5.
                # If the template was wrong (not the FOL), fixing it here avoids
                # the Phase5→audit-revert→loop cycle.
                # -----------------------------------------------------------
                revert_candidates = [
                    m for m in actionable_mismatches
                    if revert_counts.get(m.sentence_index, 0) >= 1
                ]
                if revert_candidates:
                    logger.info(
                        "Pipeline sample=%s round=%d: %d mismatch(es) have prior "
                        "revert(s) — attempting Phase 3 re-templating before Phase 5: %s",
                        sample.id, round_num, len(revert_candidates),
                        [m.sentence_index for m in revert_candidates],
                    )
                    retemplate_resolved: list[int] = []
                    for m in revert_candidates:
                        expected_truth = (m.witness_side == "q")
                        idx_key = m.sentence_index
                        if idx_key in open_issues and idx_key not in current_mismatch_indices:
                            stored = open_issues[idx_key]
                            w_domain = stored[3] if len(stored) >= 4 else None
                        else:
                            w_domain = witness_domains.get(m.witness_index)
                        if w_domain is None:
                            continue
                        hint = (
                            f"Your previous template `{m.grounded_formula}` led to a "
                            f"mismatch that was repaired at the FOL level but reverted "
                            f"by semantic audit ({revert_counts[m.sentence_index]} time(s)). "
                            f"This suggests the TEMPLATE is wrong, not the FOL. "
                            f"Re-examine the logical structure of the sentence."
                        )
                        schema_str = format_domain_schema(w_domain)
                        new_tmpl = await retemplate_with_hint(
                            idx=m.sentence_index,
                            sentence=sentences[m.sentence_index],
                            domain_schema_str=schema_str,
                            domain=w_domain,
                            current_template=m.grounded_formula,
                            hint=hint,
                            llm=self.llm,
                            prompt_engine=self.prompt_engine,
                            max_retries=self.config.pipeline.grounding_retries,
                            world_assumption=self.config.pipeline.world_assumption,
                            solver=self.solver,
                        )
                        if not new_tmpl.failed:
                            new_truth = self.solver.evaluate_grounded_formula(
                                w_domain, new_tmpl.template_code
                            )
                            if new_truth == expected_truth:
                                logger.info(
                                    "Pipeline sample=%s round=%d: pre-Phase5 re-templating "
                                    "resolved sentence %d (revert_count=%d)",
                                    sample.id, round_num, m.sentence_index,
                                    revert_counts[m.sentence_index],
                                )
                                retemplate_resolved.append(m.sentence_index)
                                revert_counts.pop(m.sentence_index, None)
                                if template_cache is not None and m.sentence_index < len(template_cache):
                                    template_cache[m.sentence_index] = new_tmpl
                    # Remove resolved mismatches from actionable list
                    if retemplate_resolved:
                        actionable_mismatches = [
                            m for m in actionable_mismatches
                            if m.sentence_index not in retemplate_resolved
                        ]
                        logger.info(
                            "Pipeline sample=%s round=%d: pre-Phase5 re-templating resolved "
                            "%d mismatch(es); %d remain for Phase 5",
                            sample.id, round_num, len(retemplate_resolved),
                            len(actionable_mismatches),
                        )

            if actionable_mismatches:
                gap = compute_gap_analysis(premises, q, actionable_mismatches, background_constraints)

                cgbv_log.update_phase("phase5", f"round {round_num}/{self.config.pipeline.r_max}")
                p5 = await run_phase5(
                    mismatches=actionable_mismatches,
                    premises=premises,
                    q=q,
                    namespace=namespace,
                    raw_code=p1.raw_code,
                    domain=first_witness_domain,      # fallback domain for prompt
                    llm=self.llm,
                    prompt_engine=self.prompt_engine,
                    max_retries=self.config.pipeline.repair_retries,
                    models=witness_models,            # current-round witnesses (witness_index keyed)
                    domains=witness_domains,
                    mismatch_models=carried_mismatch_models,   # carried overrides (sentence_index keyed)
                    mismatch_domains=carried_mismatch_domains,
                    solver=self.solver,
                    world_assumption=self.config.pipeline.world_assumption,
                    gap_analysis=gap,
                    sparse_witness_format=self.config.pipeline.sparse_witness_format,
                )
                # Phase 5 repair.json written after bridge decisions so it
                # reflects which bridges were actually committed (Fix #4).
                _committed_bridges: list = []

                round_record.repair_attempted = True
                round_record.repair_local_validated = p5.num_local_validated
                round_record.repair_seed_witness_validated = p5.num_local_validated

                # 9.2: Partial commit — commit successful repairs even if some failed.
                # p5.repaired_premises/q already contain successful repairs mixed with
                # unchanged originals (run_phase5 applies repairs incrementally).
                any_succeeded = any(r.success for r in p5.repairs)
                round_record.repair_success = any_succeeded

                if any_succeeded:
                    # Snapshot for possible revert
                    old_premises = list(premises)
                    old_q = q
                    old_verdict = verdict
                    old_violation_keys = set(history_violation_keys)

                    premises = p5.repaired_premises
                    q = p5.repaired_q

                    # Fix C: Premise consistency check (Principle 5).
                    # After repair, verify the premise set is still satisfiable.
                    # A contradictory premise set lets Z3 prove anything (ex falso
                    # quodlibet), producing a spurious Entailed verdict.
                    _sat_check_solver = _z3_module.Solver()
                    for _p in premises:
                        _sat_check_solver.add(_p)
                    for _bc in background_constraints:
                        _sat_check_solver.add(_bc)
                    if _sat_check_solver.check() == _z3_module.unsat:
                        logger.warning(
                            "Pipeline sample=%s round=%d: repaired premises are UNSAT — "
                            "contradiction detected, rejecting repair and escalating",
                            sample.id, round_num,
                        )
                        # Revert repair
                        premises = old_premises
                        q = old_q
                        round_record.verdict_after = old_verdict
                        round_record.repair_reverted = True
                        round_record.repair_success = False
                        # Escalation: trigger targeted re-formalization immediately
                        cgbv_log.update_phase("phase1", f"contradiction-escalation round {round_num}")
                        contradiction_hint = (
                            f"Phase 5 repair created a contradiction in the premise set "
                            f"(premises are UNSAT after modifying sentences "
                            f"{[r.sentence_index for r in p5.repairs if r.success]}). "
                            f"The formalization has structural errors that cannot be "
                            f"patched. Rewrite the full theory."
                        )
                        p1_new = await run_phase1_targeted(
                            original_code=p1.raw_code,
                            failed_repairs=[
                                (
                                    Mismatch(
                                        sentence_index=r.sentence_index,
                                        nl_sentence=sentences[r.sentence_index] if r.sentence_index < len(sentences) else "",
                                        mismatch_type="strengthening",
                                        fol_truth=False,
                                        grounded_truth=True,
                                        fol_formula_str="",
                                        grounded_formula="",
                                    ),
                                    contradiction_hint,
                                )
                                for r in p5.repairs if r.success
                            ],
                            premises_nl=sample.premises,
                            conclusion_nl=sample.conclusion,
                            llm=self.llm,
                            solver=self.solver,
                            prompt_engine=self.prompt_engine,
                            task_type=sample.task_type,
                            code_exec_timeout=self.config.pipeline.code_exec_timeout,
                            world_assumption=self.config.pipeline.world_assumption,
                            max_retries=self.config.pipeline.formalize_retries,
                        )
                        if p1_new is not None:
                            old_is_unknown = (verdict == VERDICT_UNKNOWN)
                            if p1_new.verdict == VERDICT_UNKNOWN and not old_is_unknown:
                                logger.warning(
                                    "Pipeline sample=%s round=%d: contradiction escalation "
                                    "degraded solver to Unknown; keeping old theory",
                                    sample.id, round_num,
                                )
                            else:
                                premises = list(p1_new.premises)
                                q = p1_new.q
                                background_constraints = list(p1_new.background_constraints)
                                bound_var_names = p1_new.bound_var_names
                                namespace = p1_new.namespace
                                verdict = p1_new.verdict
                                model_info = p1_new.model_info
                                model_info_q = p1_new.model_info_q
                                p1 = p1_new
                                template_cache = None
                                revert_counts.clear()
                                history_witness_bank.clear()
                                round_record.verdict_after = verdict
                        self._write_json(
                            out_dir / f"round{round_num}_contradiction_escalation.json",
                            {
                                "triggered": True,
                                "reformalize_success": p1_new is not None,
                                "new_verdict": p1_new.verdict if p1_new else None,
                            },
                        )
                        # Skip rest of repair processing for this round
                        self._write_json(
                            out_dir / f"round{round_num}_repair.json",
                            _phase5_to_dict(p5, accepted_bridges=[]),
                        )
                        rounds.append(round_record)
                        continue

                    # Per-bridge filter: only commit bridges that strictly reduce
                    # unresolved obligations. Bridges that don't improve theory
                    # adequacy stay out of background_constraints and raw_code.
                    accepted_bridges = _filter_obligation_reducing_bridges(
                        bridges=list(p5.bridge_axioms),
                        premises=list(premises),
                        q=q,
                        mismatches=actionable_mismatches,
                        background_constraints=list(background_constraints),
                    ) if p5.bridge_axioms else []
                    n_stripped = len(p5.bridge_axioms) - len(accepted_bridges)
                    if accepted_bridges:
                        background_constraints.extend(accepted_bridges)
                        logger.info(
                            "Pipeline sample=%s round=%d: %d/%d bridge(s) accepted "
                            "(%d stripped — no obligation reduction)",
                            sample.id, round_num,
                            len(accepted_bridges), len(p5.bridge_axioms), n_stripped,
                        )
                    elif p5.bridge_axioms:
                        logger.info(
                            "Pipeline sample=%s round=%d: all %d bridge(s) stripped "
                            "(no obligation reduction)",
                            sample.id, round_num, len(p5.bridge_axioms),
                        )

                    solver_premises = list(premises) + background_constraints
                    if sample.task_type == "three_class":
                        new_verdict, new_model_info, new_model_info_q = (
                            self.solver.check_entailment_three_class(solver_premises, q)
                        )
                    else:
                        new_verdict, new_model_info = self.solver.check_entailment(
                            solver_premises, q
                        )
                        new_model_info_q = None

                    # Label-free verdict stability guard: only revert if the
                    # repair caused the solver to degrade to Unknown (solver
                    # failure / timeout).  All other verdict changes — including
                    # Entailed↔Refuted, Entailed↔Uncertain, etc. — are accepted
                    # as the system's honest belief after repair.
                    # NOTE: we intentionally do NOT use sample.label here to
                    # avoid oracle leakage that would invalidate experiments.
                    round_record.verdict_candidate = new_verdict
                    new_is_unknown = (new_verdict == VERDICT_UNKNOWN)
                    old_is_unknown = (old_verdict == VERDICT_UNKNOWN)
                    reject_reason: str | None = None
                    if new_is_unknown and not old_is_unknown:
                        reject_reason = (
                            f"repair degraded solver verdict to Unknown "
                            f"({old_verdict} → {new_verdict})"
                        )
                    else:
                        tentative_history = _rerun_phase4_on_witness_bank(
                            history_witness_bank,
                            template_cache or [],
                            sentences,
                            premises,
                            q,
                            namespace,
                            self.solver,
                        )
                        eliminated_history = {
                            idx
                            for idx, p2_hist in enumerate(history_witness_bank)
                            if _witness_eliminated_by_bridges(accepted_bridges, p2_hist.model)
                        }
                        new_violation_keys = _phase4_violation_keys(
                            [wr.phase4 for wr in tentative_history],
                            eliminated_history,
                        )
                        changed_indices = [
                            i for i, (old_f, new_f) in enumerate(
                                zip(list(old_premises) + [old_q], list(premises) + [q])
                            )
                            if str(old_f) != str(new_f)
                        ]
                        touched_violation_keys = (
                            _phase4_violation_keys_for_sentences(
                                [wr.phase4 for wr in tentative_history],
                                changed_indices,
                                eliminated_history,
                            )
                            if changed_indices else set()
                        )
                        if touched_violation_keys:
                            reject_reason = (
                                "candidate repair leaves touched sentence obligations unresolved "
                                f"on the witness bank ({len(touched_violation_keys)} remaining)"
                            )
                        elif not new_violation_keys < old_violation_keys:
                            reject_reason = (
                                "candidate repair did not strictly shrink the current "
                                f"violation set ({len(old_violation_keys)} → "
                                f"{len(new_violation_keys)})"
                            )
                        if reject_reason is None:
                            if changed_indices:
                                # Deterministic gate: if template_cache is available,
                                # compare repaired FOL against cached templates on all
                                # witnesses.  This avoids the LLM re-generation randomness
                                # that caused correct repairs to be rejected (Violation 2).
                                all_new = list(premises) + [q]
                                if template_cache is not None:
                                    for idx in changed_indices:
                                        if reject_reason is not None:
                                            break
                                        if idx >= len(template_cache) or template_cache[idx] is None:
                                            continue  # no cache for this index → skip (conservative accept)
                                        cached_tmpl = template_cache[idx]
                                        if cached_tmpl.failed:
                                            continue
                                        for wr in tentative_history:
                                            if wr.phase2.model is None or wr.phase2.domain is None:
                                                continue
                                            fol_val = _finite_evaluator.evaluate(
                                                wr.phase2.model, all_new[idx], namespace=namespace,
                                            )
                                            tmpl_val = self.solver.evaluate_grounded_formula(
                                                wr.phase2.domain, cached_tmpl.template_code,
                                            )
                                            if fol_val is not None and tmpl_val is not None and fol_val != tmpl_val:
                                                reject_reason = (
                                                    f"Repaired FOL for sentence {idx} disagrees with "
                                                    f"cached template on witness {wr.witness_index} "
                                                    f"(FOL={fol_val}, template={tmpl_val})"
                                                )
                                                break
                                    self._write_json(
                                        out_dir / f"round{round_num}_semantic_audit.json",
                                        {
                                            "audit_type": "deterministic_template_cache",
                                            "changed_indices": changed_indices,
                                            "stable": reject_reason is None,
                                            "reject_reason": reject_reason,
                                        },
                                    )
                                else:
                                    # No template cache (first round) → fall back to LLM audit
                                    semantic_audit = await audit_semantic_stability(
                                        sentences=sentences,
                                        premises=premises,
                                        q=q,
                                        namespace=namespace,
                                        raw_code=p1.raw_code,
                                        witness_results=tentative_history,
                                        llm=self.llm,
                                        prompt_engine=self.prompt_engine,
                                        indices=changed_indices,
                                    )
                                    self._write_json(
                                        out_dir / f"round{round_num}_semantic_audit.json",
                                        _semantic_audit_to_dict(semantic_audit),
                                    )
                                    if not semantic_audit.stable:
                                        reject_reason = (
                                            "semantic stability audit rejected the candidate patch "
                                            f"({len(semantic_audit.issues)} issue(s))"
                                        )

                    if reject_reason is not None:
                        logger.warning(
                            "Pipeline sample=%s round=%d: %s; reverting",
                            sample.id, round_num, reject_reason,
                        )
                        premises = old_premises
                        q = old_q
                        verdict = old_verdict
                        round_record.verdict_after = old_verdict
                        round_record.repair_reverted = True
                        round_record.repair_success = False
                        regenerate_indices = None  # revert → FOL unchanged, templates still valid

                        # Revert accepted bridge axioms committed this round
                        for _ in accepted_bridges:
                            if background_constraints:
                                background_constraints.pop()

                        # Repair reverted — only revert-count state is persistent.
                        for m in actionable_mismatches:
                            revert_counts[m.sentence_index] = revert_counts.get(m.sentence_index, 0) + 1
                    else:
                        verdict = new_verdict
                        round_record.verdict_after = new_verdict
                        model_info = new_model_info
                        model_info_q = new_model_info_q
                        _committed_bridges = accepted_bridges
                        # Append Phase 5 repair notes to raw_code so subsequent
                        # repair prompts reflect the current formula state.
                        repair_notes_p5 = []
                        all_old = list(old_premises) + [old_q]
                        all_new = list(premises) + [q]
                        n_prem = len(old_premises)
                        for i, (old_f, new_f) in enumerate(zip(all_old, all_new)):
                            if str(new_f) != str(old_f):
                                label = "q" if i == n_prem else f"premises[{i}]"
                                repair_notes_p5.append(
                                    f"# Phase 5 repair: {label} = {str(new_f)}"
                                )
                        # Only annotate bridges that were actually accepted
                        for bridge in accepted_bridges:
                            repair_notes_p5.append(
                                f"# Phase 5 bridge axiom: {str(bridge)}"
                            )
                        updated_code = p1.raw_code
                        if repair_notes_p5:
                            updated_code = (
                                p1.raw_code.rstrip()
                                + "\n\n# ---- Phase 5 repairs (round "
                                + str(round_num) + ") ----\n"
                                + "\n".join(repair_notes_p5)
                            )
                        p1 = _patched_p1(
                            p1, premises, q, new_verdict, new_model_info, new_model_info_q,
                            raw_code=updated_code,
                            background_constraints=list(background_constraints),
                        )
                        logger.info(
                            "Pipeline sample=%s round=%d: repair committed "
                            "(%d/%d succeeded, %d bridge(s) kept), new verdict=%s",
                            sample.id, round_num,
                            sum(1 for r in p5.repairs if r.success),
                            len(p5.repairs), len(accepted_bridges), verdict,
                        )
                        repaired_indices = {
                            r.sentence_index for r in p5.repairs if r.success
                        }
                        # Template cache: mark repaired indices for regeneration
                        # in the next round (their FOL changed, template may need updating)
                        regenerate_indices = repaired_indices if repaired_indices else None
                else:
                    logger.warning(
                        "Pipeline sample=%s round=%d: all repairs failed",
                        sample.id, round_num,
                    )
                    old_violation_keys = set(history_violation_keys)
                    # Per-bridge filter: only commit bridges that strictly reduce
                    # unresolved obligations. Others stay out of
                    # background_constraints and raw_code entirely.
                    bridge_accepted = False
                    accepted_bridges_bo: list = []
                    if p5.bridge_axioms:
                        accepted_bridges_bo = _filter_obligation_reducing_bridges(
                            bridges=list(p5.bridge_axioms),
                            premises=list(premises),
                            q=q,
                            mismatches=actionable_mismatches,
                            background_constraints=list(background_constraints),
                        )
                        n_stripped_bo = len(p5.bridge_axioms) - len(accepted_bridges_bo)
                        if not accepted_bridges_bo:
                            logger.info(
                                "Pipeline sample=%s round=%d: all %d bridge(s) stripped "
                                "(no obligation reduction); no commit",
                                sample.id, round_num, len(p5.bridge_axioms),
                            )
                        else:
                            background_constraints.extend(accepted_bridges_bo)
                            bridge_notes = [
                                f"# Phase 5 bridge axiom: {str(b)}"
                                for b in accepted_bridges_bo
                            ]
                            updated_code = (
                                p1.raw_code.rstrip()
                                + "\n\n# ---- Phase 5 bridges (round "
                                + str(round_num) + ") ----\n"
                                + "\n".join(bridge_notes)
                            )
                            # Re-solve with accepted bridges so verdict/model_info
                            # are fresh for the next round's witness construction.
                            solver_premises = list(premises) + background_constraints
                            if sample.task_type == "three_class":
                                new_verdict, new_model_info, new_model_info_q = (
                                    self.solver.check_entailment_three_class(
                                        solver_premises, q)
                                )
                            else:
                                new_verdict, new_model_info = (
                                    self.solver.check_entailment(solver_premises, q)
                                )
                                new_model_info_q = None

                            # Unknown guard: reject bridges if they degrade verdict
                            new_is_unknown = (new_verdict == VERDICT_UNKNOWN)
                            old_is_unknown = (verdict == VERDICT_UNKNOWN)
                            reject_reason: str | None = None
                            if new_is_unknown and not old_is_unknown:
                                reject_reason = "bridge-only degraded solver to Unknown"
                            else:
                                tentative_history = _rerun_phase4_on_witness_bank(
                                    history_witness_bank,
                                    template_cache or [],
                                    sentences,
                                    premises,
                                    q,
                                    namespace,
                                    self.solver,
                                )
                                eliminated_history = {
                                    idx
                                    for idx, p2_hist in enumerate(history_witness_bank)
                                    if _witness_eliminated_by_bridges(
                                        accepted_bridges_bo, p2_hist.model
                                    )
                                }
                                new_violation_keys = _phase4_violation_keys(
                                    [wr.phase4 for wr in tentative_history],
                                    eliminated_history,
                                )
                                if not new_violation_keys < old_violation_keys:
                                    reject_reason = (
                                        "bridge-only candidate did not strictly shrink the "
                                        f"current violation set ({len(old_violation_keys)} → "
                                        f"{len(new_violation_keys)})"
                                    )
                            if reject_reason is not None:
                                for _ in accepted_bridges_bo:
                                    if background_constraints:
                                        background_constraints.pop()
                                accepted_bridges_bo = []
                                logger.warning(
                                    "Pipeline sample=%s round=%d: %s; reverted",
                                    sample.id, round_num, reject_reason,
                                )
                            else:
                                bridge_accepted = True
                                old_verdict_for_log = verdict  # Fix #5: capture before update
                                verdict = new_verdict
                                model_info = new_model_info
                                model_info_q = new_model_info_q
                                round_record.verdict_after = new_verdict
                                round_record.repair_success = True  # Fix #3
                                _committed_bridges = accepted_bridges_bo  # Fix #4
                                p1 = _patched_p1(
                                    p1, premises, q, new_verdict, new_model_info,
                                    new_model_info_q, raw_code=updated_code,
                                    background_constraints=list(background_constraints),
                                )
                                logger.info(
                                    "Pipeline sample=%s round=%d: %d/%d bridge(s) "
                                    "committed (%d stripped), verdict %s→%s",
                                    sample.id, round_num,
                                    len(accepted_bridges_bo), len(p5.bridge_axioms),
                                    n_stripped_bo, old_verdict_for_log, verdict,
                                )
                # Write Phase 5 repair.json after bridge decisions so
                # accepted_bridge_axioms reflects only what was committed (Fix #4).
                self._write_json(
                    out_dir / f"round{round_num}_repair.json",
                    _phase5_to_dict(p5, accepted_bridges=_committed_bridges),
                )

                # --- Targeted re-formalization ---
                # When Phase 5 fails on some mismatches, the detect-repair loop is
                # incomplete: Phase 4 detected the problem but Phase 5 couldn't fix
                # it within the existing namespace (e.g., missing entity constants,
                # wrong predicate decomposition).  Targeted re-formalization rewrites
                # the entire Z3 theory to close this gap.
                mismatch_by_idx_local = {m.sentence_index: m for m in actionable_mismatches}
                failed_repairs = [
                    (mismatch_by_idx_local[r.sentence_index], r.error)
                    for r in p5.repairs
                    if not r.success and r.sentence_index in mismatch_by_idx_local
                ]
                if failed_repairs and not round_record.repair_reverted:
                    cgbv_log.update_phase("phase1", f"re-formalize round {round_num}")
                    p1_new = await run_phase1_targeted(
                        original_code=p1.raw_code,
                        failed_repairs=failed_repairs,
                        premises_nl=sample.premises,
                        conclusion_nl=sample.conclusion,
                        llm=self.llm,
                        solver=self.solver,
                        prompt_engine=self.prompt_engine,
                        task_type=sample.task_type,
                        code_exec_timeout=self.config.pipeline.code_exec_timeout,
                        world_assumption=self.config.pipeline.world_assumption,
                        max_retries=self.config.pipeline.formalize_retries,
                    )
                    if p1_new is not None:
                        # Unknown guard: only accept if it doesn't degrade to Unknown
                        old_is_unknown = (verdict == VERDICT_UNKNOWN)
                        if p1_new.verdict == VERDICT_UNKNOWN and not old_is_unknown:
                            logger.warning(
                                "Pipeline sample=%s round=%d: targeted re-formalization "
                                "degraded solver to Unknown; discarding",
                                sample.id, round_num,
                            )
                        else:
                            logger.info(
                                "Pipeline sample=%s round=%d: targeted re-formalization "
                                "succeeded, verdict %s→%s",
                                sample.id, round_num, verdict, p1_new.verdict,
                            )
                            # Replace the full theory state
                            premises = list(p1_new.premises)
                            q = p1_new.q
                            background_constraints = list(p1_new.background_constraints)
                            bound_var_names = p1_new.bound_var_names
                            namespace = p1_new.namespace
                            verdict = p1_new.verdict
                            model_info = p1_new.model_info
                            model_info_q = p1_new.model_info_q
                            sentences = sample.premises + [sample.conclusion]
                            p1 = p1_new
                            template_cache = None  # theory rewrite → invalidate all templates
                            revert_counts.clear()  # stale revert history invalid after rewrite
                            history_witness_bank.clear()
                            round_record.verdict_after = verdict
                    self._write_json(
                        out_dir / f"round{round_num}_reformalize.json",
                        {
                            "success": p1_new is not None,
                            "verdict": p1_new.verdict if p1_new else None,
                            "raw_code": p1_new.raw_code if p1_new else None,
                            "failed_mismatches": [
                                {
                                    "sentence_index": m.sentence_index,
                                    "nl_sentence": m.nl_sentence,
                                    "mismatch_type": m.mismatch_type,
                                    "repair_error": err,
                                }
                                for m, err in failed_repairs
                            ],
                        },
                    )
            else:
                # No actionable mismatches remain (all resolved via re-grounding
                # or no mismatches were detected at this stage).
                pass

            rounds.append(round_record)

        # Exhausted R_max rounds without full verification
        # P6: Compute verification confidence for unverified results
        final_history_results = _rerun_phase4_on_witness_bank(
            history_witness_bank,
            template_cache or [],
            sentences,
            premises,
            q,
            namespace,
            self.solver,
        )
        final_history_violation_keys = _phase4_violation_keys(
            [wr.phase4 for wr in final_history_results]
        )
        if _final_verification_status == "witness_failed":
            _confidence = "none"
        elif rounds:
            any_reverted = any(r.repair_reverted for r in rounds)
            any_repair_succeeded = any(r.repair_success for r in rounds)
            has_boundary_failures = bool(final_history_violation_keys)
            if any_repair_succeeded and not any_reverted and not has_boundary_failures:
                _confidence = "medium"
            else:
                _confidence = "low"
        else:
            _confidence = "none"

        final_gap = compute_gap_analysis(
            premises,
            q,
            [],
            background_constraints,
        )
        diagnostic_tags: list[str] = []
        if verdict == VERDICT_UNCERTAIN and final_gap.obligation_count > 0:
            diagnostic_tags.append("underformalized")
        if _final_verification_status == "witness_failed":
            diagnostic_tags.append("witness_failed")
        if final_history_violation_keys:
            diagnostic_tags.append("boundary_failed")
        acceptance_state = (
            "accepted"
            if _final_verification_status == "verified"
            and "underformalized" not in diagnostic_tags
            else "needs_repair"
        )

        result = PipelineResult(
            sample_id=sample.id,
            dataset=sample.dataset,
            label=sample.label,
            verdict=verdict,
            verdict_pre_bridge=verdict_pre_bridge,
            verdict_post_bridge=verdict_post_bridge,
            verified=False,
            num_rounds=self.config.pipeline.r_max,
            rounds=rounds,
            phase1_raw_code=p1.raw_code,
            phase1_repeated_failure=p1.repeated_failure,
            acceptance_state=acceptance_state,
            diagnostic_tags=diagnostic_tags,
            semantic_stable=None,
            initial_obligation_count=initial_obligation_count,
            final_obligation_count=final_gap.obligation_count,
            execution_status="success",
            verification_status=_final_verification_status,
            verification_confidence=_confidence,
        )
        self._write_json(out_dir / "result.json", asdict(result))
        logger.info(
            "Pipeline sample=%s UNVERIFIED after %d rounds",
            sample.id, self.config.pipeline.r_max,
        )
        return result

    async def _try_verify_via_audit(
        self,
        sentences: list[str],
        premises: list,
        q: object,
        namespace: dict,
        raw_code: str,
        witness_results: list,
        sample: DataSample,
        round_num: int,
        out_dir: Path,
    ) -> tuple[str, SemanticAuditResult, Phase1Result | None]:
        """Run semantic audit and, if unstable, attempt targeted re-formalization.

        Returns:
            (action, audit, p1_new) where action is one of:
              - "verified": audit passed, caller should return verified result
              - "reformalized": audit failed but re-formalization succeeded;
                caller should update state from p1_new and continue loop
              - "failed": audit failed and re-formalization failed/degraded;
                caller should return semantic_unstable result
        """
        semantic_audit = await audit_semantic_stability(
            sentences=sentences,
            premises=premises,
            q=q,
            namespace=namespace,
            raw_code=raw_code,
            witness_results=witness_results,
            llm=self.llm,
            prompt_engine=self.prompt_engine,
        )
        self._write_json(
            out_dir / f"round{round_num}_semantic_audit.json",
            _semantic_audit_to_dict(semantic_audit),
        )

        if semantic_audit.stable:
            return "verified", semantic_audit, None

        logger.warning(
            "Pipeline sample=%s round=%d: semantic stability audit "
            "flagged %d issue(s); triggering targeted re-formalization",
            sample.id, round_num, len(semantic_audit.issues),
        )
        cgbv_log.update_phase("phase1", f"semantic-audit round {round_num}")
        p1_new = await run_phase1_targeted(
            original_code=raw_code,
            failed_repairs=[
                (issue, "Semantic stability audit diverged on the current witness bank.")
                for issue in semantic_audit.issues
            ],
            premises_nl=sample.premises,
            conclusion_nl=sample.conclusion,
            llm=self.llm,
            solver=self.solver,
            prompt_engine=self.prompt_engine,
            task_type=sample.task_type,
            code_exec_timeout=self.config.pipeline.code_exec_timeout,
            world_assumption=self.config.pipeline.world_assumption,
            max_retries=self.config.pipeline.formalize_retries,
        )

        if p1_new is None or p1_new.verdict == VERDICT_UNKNOWN:
            return "failed", semantic_audit, None

        return "reformalized", semantic_audit, p1_new

    async def _try_reformalize_non_bridgeable_gap(
        self,
        *,
        sentences: list[str],
        premises: list,
        q: object,
        raw_code: str,
        gap: Any,
        sample: DataSample,
        round_num: int,
    ) -> Phase1Result | None:
        """
        Conservative fallback for gap-only states that should NOT be bridged.

        This keeps the pipeline from inventing new bridge axioms when the gap
        lacks grounded evidence to reconnect; instead we ask Phase 1 targeted
        re-formalization to rewrite the theory structure.
        """
        target_idx = (
            gap.query_relevant_premise_indices[0]
            if getattr(gap, "query_relevant_premise_indices", None)
            else len(premises)
        )
        nl_sentence = sentences[target_idx] if 0 <= target_idx < len(sentences) else ""
        hint = (
            "Gap analysis found non-bridgeable structural missing links. "
            f"Reason: {getattr(gap, 'non_bridgeable_reason', 'prefer re-formalization')}. "
            f"Missing links: {getattr(gap, 'missing_links', [])}. "
            "Do not add a new bridge axiom. Rewrite the formalization so the "
            "query-relevant proof path is represented directly."
        )
        logger.info(
            "Pipeline sample=%s round=%d: non-bridgeable gap routed to targeted "
            "re-formalization (%s)",
            sample.id, round_num, getattr(gap, "non_bridgeable_reason", "no reason"),
        )
        cgbv_log.update_phase("phase1", f"gap-reformalize round {round_num}")
        synthetic_mismatch = Mismatch(
            sentence_index=target_idx,
            nl_sentence=nl_sentence,
            mismatch_type="strengthening",
            fol_truth=False,
            grounded_truth=True,
            fol_formula_str=_formula_str(target_idx, premises, q),
            grounded_formula="",
        )
        return await run_phase1_targeted(
            original_code=raw_code,
            failed_repairs=[(synthetic_mismatch, hint)],
            premises_nl=sample.premises,
            conclusion_nl=sample.conclusion,
            llm=self.llm,
            solver=self.solver,
            prompt_engine=self.prompt_engine,
            task_type=sample.task_type,
            code_exec_timeout=self.config.pipeline.code_exec_timeout,
            world_assumption=self.config.pipeline.world_assumption,
            max_retries=self.config.pipeline.formalize_retries,
        )

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_make_serialisable(data), f, indent=2, ensure_ascii=False, default=_json_default)
        except Exception as e:
            logger.warning("Failed to write %s: %s", path, e)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _phase1_to_dict(p1: Phase1Result, sample: DataSample | None = None) -> dict:
    return {
        "verdict": p1.verdict,
        "verdict_pre_bridge": p1.verdict_pre_bridge,
        "raw_code": p1.raw_code,
        "repeated_failure": p1.repeated_failure,
        "premises_nl": list(sample.premises) if sample is not None else None,
        "conclusion_nl": sample.conclusion if sample is not None else None,
        "num_premises_nl": len(sample.premises) if sample is not None else None,
        "num_premises_formulas": len(p1.premises),
        "premises": [str(f) for f in p1.premises],
        "background_constraints": [str(f) for f in p1.background_constraints],
        "bound_var_names": sorted(p1.bound_var_names),
        "q": str(p1.q) if p1.q is not None else None,
        "attempts": [
            {
                "attempt_num": a.attempt_num,
                "messages": a.messages,
                "raw_output": a.raw_output,
                "code_exec_error": a.code_exec_error,
                "validation_error": a.validation_error,
                "solver_error": a.solver_error,
                "verdict": a.verdict,
                "diagnostic": a.diagnostic,
            }
            for a in p1.attempts
        ],
        "error": p1.error,
    }


def _mismatch_to_dict(m: Mismatch) -> dict:
    return {
        "sentence_index": m.sentence_index,
        "nl_sentence": m.nl_sentence,
        "mismatch_type": m.mismatch_type,
        "fol_truth": m.fol_truth,
        "grounded_truth": m.grounded_truth,
        "fol_formula_str": m.fol_formula_str,
        "grounded_formula": m.grounded_formula,
        "witness_index": m.witness_index,
        "witness_side": m.witness_side,
        "is_phase3_error": m.is_phase3_error,
        "fol_quantifier_layout": m.fol_quantifier_layout,
    }


def _mw_to_dict(mw: MultiWitnessResult) -> dict:
    return {
        "num_witnesses_constructed": mw.num_witnesses_constructed,
        "all_passed": mw.all_passed,
        "mismatches": [_mismatch_to_dict(m) for m in mw.mismatches],
        "witnesses": [
            {
                "witness_index": wr.witness_index,
                "witness_side": wr.phase2.witness_side,
                "domain_summary": _domain_summary(wr.phase2.domain),
                "domain": wr.phase2.domain,
                "phase4_all_passed": wr.phase4.all_passed,
                "phase4_mismatches": len(wr.phase4.mismatches),
                "phase4_unverifiable": wr.phase4.num_unverifiable,
            }
            for wr in mw.witness_results
        ],
    }


def _phase3_to_dict(p3: Phase3Result) -> dict:
    return {
        "grounded": [
            {
                "sentence_index": g.sentence_index,
                "nl_sentence": g.nl_sentence,
                "formula_code": g.formula_code,
                "failed": g.failed,
                "attempts": [
                    {
                        "attempt_num": a.attempt_num,
                        "messages": a.messages,
                        "raw_output": a.raw_output,
                        "extracted_formula": a.extracted_formula,
                        "validation_error": a.validation_error,
                        "accepted": a.accepted,
                    }
                    for a in g.attempts
                ],
                "error": g.error,
            }
            for g in p3.grounded
        ],
    }


def _phase4_to_dict(p4: Phase4Result) -> dict:
    return {
        "all_passed": p4.all_passed,
        "witness_index": p4.witness_index,
        "witness_side": p4.witness_side,
        "num_unverifiable": p4.num_unverifiable,
        "mismatches": [_mismatch_to_dict(m) for m in p4.mismatches],
        "evaluations": [
            {
                "sentence_index": e.sentence_index,
                "nl_sentence": e.nl_sentence,
                "fol_truth": e.fol_truth,
                "grounded_truth": e.grounded_truth,
                "mismatch": e.mismatch,
                "mismatch_type": e.mismatch_type,
                "grounding_failed": e.grounding_failed,
                "fol_formula_str": e.fol_formula_str,
                "grounded_formula": e.grounded_formula,
                "fol_eval_repr": e.fol_eval_repr,
                "witness_index": e.witness_index,
                "witness_side": e.witness_side,
                "error": e.error,
            }
            for e in p4.evaluations
        ],
        "error": p4.error,
    }


def _phase5_to_dict(p5: Phase5Result, accepted_bridges: list | None = None) -> dict:
    """Serialise Phase5Result to a JSON-compatible dict.

    Field semantics for bridge axioms (both fields are always present):
      bridge_axioms          — all axioms the LLM proposed this round
                               (may include ones later stripped by the pipeline)
      accepted_bridge_axioms — the subset actually committed to background_constraints
                               and annotated in raw_code; this is the authoritative
                               signal for "what the solver currently knows about bridges"

    Downstream scripts should read accepted_bridge_axioms for the committed state
    and bridge_axioms only to inspect the full LLM proposal.
    """
    return {
        "all_repaired": p5.all_repaired,
        "num_local_validated": p5.num_local_validated,
        "num_seed_witness_validated": p5.num_local_validated,
        "num_formula_repairs": len(p5.repairs),
        "bridge_only": (not p5.repairs) and bool(
            accepted_bridges if accepted_bridges is not None else p5.bridge_axioms
        ),
        "error": p5.error,
        "repairs": [
            {
                "sentence_index": r.sentence_index,
                "mismatch_type": r.mismatch_type,
                "witness_index": r.witness_index,
                "witness_side": r.witness_side,
                "original": r.original_formula_str,
                "grounded_formula": r.grounded_formula,
                "fol_truth_before": r.fol_truth_before,
                "grounded_truth_expected": r.grounded_truth_expected,
                "repaired": r.repaired_expr_str,
                "success": r.success,
                "local_validated": r.local_validated,
                "seed_witness_validated": r.local_validated,
                "attempts": [
                    {
                        "attempt_num": a.attempt_num,
                        "messages": a.messages,
                        "raw_output": a.raw_output,
                        "extracted_expression": a.extracted_expression,
                        "eval_error": a.eval_error,
                        "local_validation_error": a.local_validation_error,
                        "local_validation_truth": a.local_validation_truth,
                        "accepted": a.accepted,
                    }
                    for a in r.attempts
                ],
                "error": r.error,
            }
            for r in p5.repairs
        ],
        "bridge_axioms": [str(b) for b in p5.bridge_axioms],
        "accepted_bridge_axioms": [
            str(b) for b in (
                accepted_bridges if accepted_bridges is not None else p5.bridge_axioms
            )
        ],
        "unified_attempts": p5.unified_attempts,
    }


def _semantic_audit_to_dict(audit: SemanticAuditResult) -> dict:
    return {
        "stable": audit.stable,
        "checked_indices": list(audit.checked_indices),
        "issues": [
            {
                "sentence_index": issue.sentence_index,
                "nl_sentence": issue.nl_sentence,
                "current_formula": issue.current_formula_str,
                "audited_formula": issue.audited_formula_str,
                "differing_witnesses": issue.differing_witnesses,
                "error": issue.error,
            }
            for issue in audit.issues
        ],
    }


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _filter_obligation_reducing_bridges(
    *,
    bridges: list,
    premises: list,
    q: object,
    mismatches: list,
    background_constraints: list,
) -> list:
    """
    Greedily accept bridges that strictly reduce unresolved query-relevant obligations.

    This keeps bridge selection tied to theory adequacy rather than to the current
    witness geometry. Consistency is still enforced first to avoid UNSAT bridges.
    """
    accepted: list = []
    current_bg = list(background_constraints)
    current_gap = compute_gap_analysis(premises, q, mismatches, current_bg)
    current_score = current_gap.obligation_count

    for bridge in bridges:
        if not _is_bridge_consistent(bridge, list(premises) + current_bg):
            continue
        trial_bg = current_bg + [bridge]
        trial_gap = compute_gap_analysis(premises, q, mismatches, trial_bg)
        if trial_gap.obligation_count < current_score:
            accepted.append(bridge)
            current_bg = trial_bg
            current_score = trial_gap.obligation_count
            logger.info(
                "Accepted bridge by obligation reduction: %d -> %d",
                current_gap.obligation_count, trial_gap.obligation_count,
            )
            current_gap = trial_gap
        else:
            logger.info(
                "Rejected bridge without obligation reduction: %.80s",
                str(bridge),
            )
    return accepted


def _is_bridge_consistent(
    bridge: object,
    existing_premises: list,
) -> bool:
    import z3
    s_test = z3.Solver()
    for p in existing_premises:
        s_test.add(p)
    s_test.add(bridge)
    if s_test.check() == z3.unsat:
        logger.warning(
            "Bridge rejected (inconsistent with premises): %.80s",
            str(bridge),
        )
        return False
    return True


def _witness_eliminated_by_bridges(bridges: list, model: Any | None) -> bool:
    """Return True iff any candidate bridge contradicts the given witness model."""
    if not bridges or model is None:
        return False
    import z3
    for bridge in bridges:
        try:
            if z3.is_false(model.evaluate(bridge)):
                return True
        except Exception:
            pass
    return False


def _bridge_violation_keys_on_witness_bank(
    *,
    bridges: list,
    witness_bank: list[Phase2Result],
    templates: list,
    sentences: list[str],
    premises: list,
    q: object,
    namespace: dict[str, Any],
    solver: Z3Solver,
) -> set[tuple[str, int, int, str]]:
    """
    Re-evaluate the current theory on the witness bank after adding bridges.

    Bridge-only commits do not rewrite formulas, so the only principled bank-side
    effect is witness elimination. The committed bridge set is acceptable iff the
    remaining bank stays clean.
    """
    tentative_history = _rerun_phase4_on_witness_bank(
        witness_bank,
        templates,
        sentences,
        premises,
        q,
        namespace,
        solver,
    )
    eliminated_history = {
        idx
        for idx, p2_hist in enumerate(witness_bank)
        if _witness_eliminated_by_bridges(bridges, p2_hist.model)
    }
    return _phase4_violation_keys(
        [wr.phase4 for wr in tentative_history],
        eliminated_history,
    )


def _witness_bank_fingerprint(p2: Phase2Result) -> str:
    """Stable fingerprint for de-duplicating witness worlds across rounds."""
    payload = {
        "witness_side": p2.witness_side,
        "domain": p2.domain,
    }
    return json.dumps(
        _make_serialisable(payload),
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )


def _merge_witness_bank(
    history_bank: list[Phase2Result],
    witness_results: list[WitnessCheckResult],
) -> list[Phase2Result]:
    """Append only genuinely new witness worlds to the history bank."""
    merged = list(history_bank)
    seen = {
        _witness_bank_fingerprint(p2)
        for p2 in merged
        if p2.domain is not None
    }
    for wr in witness_results:
        p2 = wr.phase2
        if p2.domain is None:
            continue
        fp = _witness_bank_fingerprint(p2)
        if fp in seen:
            continue
        seen.add(fp)
        merged.append(p2)
    return merged


def _grounded_from_templates(templates: list) -> list[GroundedFormula]:
    """Build Phase 4-compatible grounded formulas from the current template set."""
    return [
        GroundedFormula(
            sentence_index=tmpl.sentence_index,
            nl_sentence=tmpl.nl_sentence,
            formula_code=tmpl.template_code,
            failed=tmpl.failed,
            attempts=tmpl.attempts,
            error=tmpl.error,
        )
        for tmpl in templates
    ]


def _rerun_phase4_on_witness_bank(
    witness_bank: list[Phase2Result],
    templates: list,
    sentences: list[str],
    premises: list,
    q: object,
    namespace: dict[str, Any],
    solver: Z3Solver,
) -> list[WitnessCheckResult]:
    """
    Re-run Phase 4 on the accumulated witness bank using the CURRENT templates.

    The bank stores only Phase 2 witness worlds. This avoids freezing old
    Phase 3/4 outputs when templates are corrected in later rounds.
    """
    if not witness_bank or not templates or len(templates) != len(sentences):
        return []

    grounded_formulas = _grounded_from_templates(templates)
    fol_formulas = list(premises) + [q]
    results: list[WitnessCheckResult] = []
    for bank_idx, p2 in enumerate(witness_bank):
        if p2.model is None or p2.domain is None:
            continue
        p3 = Phase3Result(grounded=grounded_formulas)
        p4 = run_phase4(
            sentences=sentences,
            fol_formulas=fol_formulas,
            model=p2.model,
            domain=p2.domain,
            grounded_formulas=p3.grounded,
            solver=solver,
            namespace=namespace,
            witness_index=bank_idx,
            witness_side=p2.witness_side,
        )
        results.append(WitnessCheckResult(
            witness_index=bank_idx,
            phase2=p2,
            phase3=p3,
            phase4=p4,
        ))
    return results


def _phase4_violation_keys_for_sentences(
    phase4_results: list[Phase4Result],
    sentence_indices: list[int] | set[int],
    eliminated_witness_indices: set[int] | None = None,
) -> set[tuple[str, int, int, str]]:
    """Filter Phase 4 violation keys down to a sentence slice."""
    wanted = set(sentence_indices)
    if not wanted:
        return set()
    return {
        key for key in _phase4_violation_keys(
            phase4_results,
            eliminated_witness_indices=eliminated_witness_indices,
        )
        if key[1] in wanted
    }


def _open_issues_from_history_witness_results(
    history_witness_results: list[WitnessCheckResult],
    premises: list,
    q: object,
) -> dict[int, tuple]:
    """
    Build a per-sentence issue context from the authoritative history witness bank.

    This keeps `open_issues` as a compatibility view only; it is rebuilt every
    round instead of being manually persisted across repairs and rewrites.
    """
    issues: dict[int, tuple] = {}
    for wr in history_witness_results:
        for m in wr.phase4.mismatches:
            issues.setdefault(
                m.sentence_index,
                (
                    m,
                    _formula_str(m.sentence_index, premises, q),
                    wr.phase2.model,
                    wr.phase2.domain,
                ),
            )
    return issues


def _phase4_violation_keys(
    phase4_results: list[Phase4Result],
    eliminated_witness_indices: set[int] | None = None,
) -> set[tuple[str, int, int, str]]:
    """Collect per-witness violation keys from Phase 4 results."""
    eliminated = eliminated_witness_indices or set()
    keys: set[tuple[str, int, int, str]] = set()
    for p4 in phase4_results:
        if p4.witness_index in eliminated:
            continue
        for m in p4.mismatches:
            kind = "phase3_error" if m.is_phase3_error else "mismatch"
            keys.add((kind, m.sentence_index, p4.witness_index, p4.witness_side))
        for e in p4.evaluations:
            if e.grounding_failed or e.error is not None or e.fol_truth is None:
                keys.add(("unverifiable", e.sentence_index, p4.witness_index, p4.witness_side))
    return keys


def _make_serialisable(obj):
    if isinstance(obj, dict):
        return {
            (", ".join(str(k) for k in key) if isinstance(key, tuple) else str(key) if not isinstance(key, str) else key): _make_serialisable(val)
            for key, val in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    return obj


def _domain_summary(domain: dict | None) -> dict:
    """Compact witness summary for faster result inspection."""
    if not domain:
        return {}

    sorts = domain.get("sorts", {})
    predicates = domain.get("predicates", {})
    true_atoms = 0
    false_atoms = 0
    for interp in predicates.values():
        for val in interp.values():
            if val:
                true_atoms += 1
            else:
                false_atoms += 1

    return {
        "num_entities": len(domain.get("entities", [])),
        "entities_by_sort": {sort_name: len(entities) for sort_name, entities in sorts.items()},
        "num_predicates": len(predicates),
        "true_atoms": true_atoms,
        "false_atoms": false_atoms,
    }


def _patched_p1(
    p1: Phase1Result,
    premises: list,
    q: object,
    verdict: str,
    model_info: object,
    model_info_q: object | None = None,
    raw_code: str | None = None,
    background_constraints: list | None = None,
) -> Phase1Result:
    """Return a shallow copy of p1 with updated premises/q/verdict/model_info.

    raw_code: if provided, replaces p1.raw_code (used to append repair notes so
    downstream Phase 5 prompts show current formula state).
    background_constraints: if provided, replaces p1.background_constraints so
    the p1 snapshot stays consistent with the pipeline's live solver state.
    """
    from cgbv.core.phase1_formalize import Phase1Result as P1
    return P1(
        verdict=verdict,
        premises=premises,
        background_constraints=(
            background_constraints if background_constraints is not None
            else p1.background_constraints
        ),
        bound_var_names=p1.bound_var_names,
        q=q,
        model_info=model_info,
        model_info_q=model_info_q,
        namespace=p1.namespace,
        raw_code=raw_code if raw_code is not None else p1.raw_code,
        verdict_pre_bridge=p1.verdict_pre_bridge,  # preserve original pre-bridge snapshot
        attempts=p1.attempts,
        repeated_failure=p1.repeated_failure,
        error=None,
    )
