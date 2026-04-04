from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cgbv.logging as cgbv_log
from cgbv.config.settings import ExperimentConfig
from cgbv.core.multi_witness import MultiWitnessResult, run_multi_witness
from cgbv.core.phase1_formalize import Phase1Result, run_phase1
from cgbv.core.phase3_grounded import Phase3Result, reground_with_hint
from cgbv.solver.model_extractor import format_domain_desc
from cgbv.core.phase4_check import Mismatch, Phase4Result
from cgbv.core.phase5_repair import Phase5Result, run_phase5
from cgbv.data.base import DataSample
from cgbv.llm.base import LLMClient
from cgbv.llm.factory import create_llm_client
from cgbv.prompts.prompt_engine import PromptEngine
from cgbv.solver.z3_solver import Z3Solver, VERDICT_UNKNOWN, VERDICT_REFUTED, VERDICT_UNCERTAIN

logger = logging.getLogger(__name__)


def _normalise_verdict(v: str) -> str:
    """Lightweight verdict normaliser for pipeline-internal use.

    "unknown" is intentionally NOT mapped to "uncertain":
      - "uncertain" = semantic (P∧q and P∧¬q are both satisfiable)
      - "unknown"   = solver/execution failure (Z3 timeout, Phase 1 error)
    Keeping them distinct prevents errored samples from spuriously matching
    Uncertain-labelled ground truth.
    """
    s = str(v).strip().lower()
    if s in ("true", "entailed", "yes"):
        return "true"
    if s in ("false", "not entailed", "no", "refuted"):
        return "false"
    if s in ("uncertain", "neither"):
        return "uncertain"
    return s  # "unknown" → "unknown" (does not match any label)


def _formula_str(idx: int, premises: list, q: object) -> str:
    """
    Return a stable string fingerprint for the formula at sentence *idx*.
    Used by the open_issues staleness check: if the formula changes (repaired),
    the fingerprint changes and the stale issue is pruned.
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
    repair_local_validated: int = 0        # P0.4: mismatches that passed local acceptance
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
    #   "not_run"          — verification never started (execution failed before it)
    execution_status: str = "success"
    verification_status: str = "not_run"
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

        sentences = sample.premises + [sample.conclusion]
        rounds: list[RoundRecord] = []

        label_norm = _normalise_verdict(sample.label)
        # Regression guard baseline = verdict_post_bridge.
        # Bridge has already happened; Phase 5 must not revert a bridge fix.
        initial_correct = (_normalise_verdict(verdict_post_bridge) == label_norm)
        # Dynamic guard: starts same as initial_correct, but gets set to True
        # whenever any round produces the correct verdict.  This prevents
        # "round 1 fixes it, round 2 breaks it again" regressions that the
        # static initial_correct alone cannot catch.
        best_correct = initial_correct

        # Tracks the verification chain outcome for the final round.
        # Overwritten each round; the last value is used in the final PipelineResult.
        _final_verification_status = "exhausted_rounds"

        # Fix 4: open_issues — persisted unresolved mismatches across rounds.
        # Key: sentence_index.
        # Value: (Mismatch, formula_fingerprint, model, domain)
        #   formula_fingerprint — staleness check (formula changed = issue pruned)
        #   model, domain       — witness world captured at detection time so
        #                         Phase 5 validates carried issues against the
        #                         correct world, not a different-round witness.
        open_issues: dict[int, tuple] = {}

        # ----------------------------------------------------------------
        # Repair loop (up to R_max rounds)
        # ----------------------------------------------------------------
        for round_num in range(1, self.config.pipeline.r_max + 1):
            logger.info(
                "Pipeline sample=%s round=%d/%d verdict=%s",
                sample.id, round_num, self.config.pipeline.r_max, verdict,
            )

            # --- Prune stale open_issues ---
            # An open_issue is stale when the formula for that sentence has
            # changed (i.e. a previous repair modified it).  We compare the
            # stored formula string against the current live formula.
            stale = [
                idx for idx, issue in open_issues.items()
                if _formula_str(idx, premises, q) != issue[1]
            ]
            for idx in stale:
                logger.debug(
                    "Pipeline sample=%s: pruning stale open_issue for sentence %d",
                    sample.id, idx,
                )
                del open_issues[idx]

            # Phase 2+3+4: Multi-Witness
            cgbv_log.update_phase("phase2", f"round {round_num}/{self.config.pipeline.r_max}")
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
                num_witnesses=self.config.pipeline.num_witnesses,
                grounding_retries=self.config.pipeline.grounding_retries,
                world_assumption=self.config.pipeline.world_assumption,
            )
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

            # --- Compute carried issues ---
            # Issues in open_issues that the current witness did NOT catch.
            # These are distinct from current witness mismatches and logged
            # separately so the log accurately reflects what each witness saw.
            current_mismatch_indices = {m.sentence_index for m in mw.mismatches}
            carried: list[Mismatch] = [
                issue[0] for idx, issue in open_issues.items()
                if idx not in current_mismatch_indices
            ]

            # Build per-witness model/domain maps for this round.
            # Placed here (not inside the repair branch) so they're available
            # when updating open_issues in any code path.
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
            # A sample is verified only when the current witness is clean AND
            # there are no persisted open issues (which a new witness may simply
            # not surface, even though the formulas are unchanged).
            # This includes Phase 3 errors persisted in open_issues — they must
            # be resolved via re-grounding before verification can succeed.
            if mw.all_passed and not carried and not open_issues:
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
                    execution_status="success",
                    verification_status="verified",
                )
                self._write_json(out_dir / "result.json", asdict(result))
                logger.info("Pipeline sample=%s VERIFIED at round %d", sample.id, round_num)
                return result

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

            # Split Phase 3 structural errors from actionable (Phase 1 FOL) mismatches.
            # A mismatch on the conclusion that contradicts the witness's expected truth
            # value is definitively a Phase 3 grounding error (see phase4_check.py for
            # the exact symmetric rule).  Phase 5 cannot fix these — patching the FOL
            # formula would change the verdict, not fix the grounding.
            phase3_errors: list[Mismatch] = [m for m in effective_mismatches if m.is_phase3_error]
            actionable_mismatches: list[Mismatch] = [m for m in effective_mismatches if not m.is_phase3_error]
            # num_mismatches tracks only actionable (Phase 5-eligible) mismatches so
            # that repair_local_fix_rate = num_local_validated / num_mismatches reflects
            # the actual Phase 5 workload.  Phase 3 errors are excluded because they are
            # never sent to Phase 5 and would artificially deflate the rate.
            round_record.num_mismatches = len(actionable_mismatches)

            # --- Targeted Phase 3 re-grounding for structural errors ---
            # For each detected Phase 3 error, attempt a targeted re-grounding pass
            # using the semantic mismatch as the error hint.  This avoids exhausting
            # R_max on the same grounding error without ever giving Phase 3 the signal
            # it needs to correct the comparison direction / quantifier structure.
            # Snapshot the original detected count BEFORE re-grounding overwrites phase3_errors.
            # num_phase3_errors (set after re-grounding) records the remaining count;
            # num_phase3_detected records the total detected — the correct denominator for
            # phase3_reground_rate = num_phase3_reground_success / num_phase3_detected.
            round_record.num_phase3_detected = len(phase3_errors)

            # Persist Phase 3 errors in open_issues so they survive across rounds.
            # Without this, a structural grounding error detected in round N can
            # "disappear" in round N+1 (different witness / fallback formula) and
            # cause a false verification — the "whitewashing" bug.
            for m in phase3_errors:
                wm, wd = _pick_world_for_issue(
                    m, current_mismatch_indices,
                    open_issues, witness_models, witness_domains,
                )
                open_issues[m.sentence_index] = (
                    m, _formula_str(m.sentence_index, premises, q), wm, wd,
                )

            if phase3_errors:
                logger.warning(
                    "Pipeline sample=%s round=%d: %d structural Phase 3 grounding error(s) "
                    "detected (conclusion mismatch contradicts witness construction) — "
                    "attempting targeted re-grounding: %s",
                    sample.id, round_num, len(phase3_errors),
                    [(m.witness_side, m.mismatch_type, m.grounded_formula[:40])
                     for m in phase3_errors],
                )
                cgbv_log.update_phase("phase3", f"re-grounding round {round_num}")
                still_broken: list[Mismatch] = []
                reground_records: list[dict] = []
                for m in phase3_errors:
                    expected_truth = (m.witness_side == "q")  # q→True, not_q→False
                    # For carried mismatches, witness_index refers to a *previous round*
                    # where indices were renumbered — use the stored domain instead.
                    # For current-round mismatches, use this round's witness domain.
                    idx_key = m.sentence_index
                    if idx_key in open_issues and idx_key not in current_mismatch_indices:
                        stored = open_issues[idx_key]
                        witness_domain = stored[3] if len(stored) >= 4 else None
                    else:
                        witness_domain = witness_domains.get(m.witness_index)
                    if witness_domain is None:
                        still_broken.append(m)
                        continue
                    new_gf = await reground_with_hint(
                        idx=m.sentence_index,
                        sentence=sentences[m.sentence_index],
                        domain_desc_str=format_domain_desc(witness_domain),
                        domain=witness_domain,
                        current_formula=m.grounded_formula,
                        expected_truth=expected_truth,
                        llm=self.llm,
                        prompt_engine=self.prompt_engine,
                        max_retries=self.config.pipeline.grounding_retries,
                        world_assumption=self.config.pipeline.world_assumption,
                    )
                    resolved = False
                    if not new_gf.failed:
                        new_truth = self.solver.evaluate_grounded_formula(
                            witness_domain, new_gf.formula_code
                        )
                        if new_truth == expected_truth:
                            logger.info(
                                "Pipeline sample=%s round=%d: re-grounding resolved "
                                "sentence %d (%s→%s)",
                                sample.id, round_num, m.sentence_index,
                                m.grounded_formula[:40], new_gf.formula_code[:40],
                            )
                            resolved = True
                            # Remove resolved Phase 3 error from open_issues
                            open_issues.pop(m.sentence_index, None)
                    reground_records.append({
                        "sentence_index": m.sentence_index,
                        "witness_side": m.witness_side,
                        "old_formula": m.grounded_formula,
                        "new_formula": new_gf.formula_code if not new_gf.failed else "",
                        "resolved": resolved,
                        "error": new_gf.error if new_gf.failed else None,
                    })
                    if not resolved:
                        still_broken.append(m)

                resolved_count = len(phase3_errors) - len(still_broken)
                round_record.num_phase3_reground_success = resolved_count
                phase3_errors = still_broken
                self._write_json(
                    out_dir / f"round{round_num}_phase3_reground.json",
                    {"resolved": resolved_count, "attempts": reground_records},
                )

            round_record.num_phase3_errors = len(phase3_errors)
            if phase3_errors:
                logger.warning(
                    "Pipeline sample=%s round=%d: %d Phase 3 error(s) remain unresolved "
                    "after targeted re-grounding; skipping Phase 5 for these.",
                    sample.id, round_num, len(phase3_errors),
                )

            # --- Post-reground verified check ---
            # If targeted re-grounding resolved ALL Phase 3 errors AND there are no
            # actionable mismatches or carried issues, this round is now clean.
            # We must perform this check here (after re-grounding) rather than relying
            # on the pre-reground verified check at the top of the loop, which fired
            # before Phase 3 errors were detected and resolved.
            if not phase3_errors and not actionable_mismatches and not carried and not open_issues:
                round_record.all_passed = True
                # Drop the resolved Phase 3 entries from the mismatches snapshot —
                # they are no longer errors and should not appear in the final record.
                round_record.mismatches = [
                    m for m in round_record.mismatches
                    if not m.get("is_phase3_error", False)
                ]
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
                    execution_status="success",
                    verification_status="verified",
                )
                self._write_json(out_dir / "result.json", asdict(result))
                logger.info(
                    "Pipeline sample=%s VERIFIED at round %d "
                    "(all Phase 3 grounding errors resolved via targeted re-grounding)",
                    sample.id, round_num,
                )
                return result

            if round_num < self.config.pipeline.r_max and actionable_mismatches:
                # Use the first witness for repair prompt context (fallback domain)
                first_wr = mw.witness_results[0] if mw.witness_results else None
                first_witness_domain = first_wr.phase2.domain if first_wr else {}

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
                )
                self._write_json(
                    out_dir / f"round{round_num}_repair.json",
                    _phase5_to_dict(p5),
                )

                round_record.repair_attempted = True
                round_record.repair_success = p5.all_repaired
                round_record.repair_local_validated = p5.num_local_validated

                if p5.all_repaired:
                    # Snapshot for possible revert
                    old_premises = list(premises)
                    old_q = q
                    old_verdict = verdict

                    premises = p5.repaired_premises
                    q = p5.repaired_q

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

                    # P0.4 Global revert: undo repair if it regressed from the best
                    # known correct verdict (dynamic guard — catches round-1-fixes/
                    # round-2-breaks regressions that static initial_correct misses).
                    new_correct = (_normalise_verdict(new_verdict) == label_norm)
                    round_record.verdict_candidate = new_verdict
                    if best_correct and not new_correct:
                        logger.warning(
                            "Pipeline sample=%s round=%d: repair regressed verdict "
                            "(%s → %s vs label=%s); reverting",
                            sample.id, round_num, old_verdict, new_verdict, sample.label,
                        )
                        premises = old_premises
                        q = old_q
                        verdict = old_verdict
                        round_record.verdict_after = old_verdict
                        round_record.repair_reverted = True
                        round_record.repair_success = False

                        # Repair reverted — persist actionable mismatches only.
                        # Phase 3 errors are never added to open_issues because
                        # Phase 5 cannot repair them (they need re-grounding, not
                        # FOL patching).
                        for m in actionable_mismatches:
                            wm, wd = _pick_world_for_issue(
                                m, current_mismatch_indices,
                                open_issues, witness_models, witness_domains,
                            )
                            open_issues[m.sentence_index] = (
                                m, _formula_str(m.sentence_index, premises, q), wm, wd,
                            )
                    else:
                        verdict = new_verdict
                        round_record.verdict_after = new_verdict
                        model_info = new_model_info
                        model_info_q = new_model_info_q
                        # Once we land on the correct verdict, lock in the guard so
                        # no subsequent round can silently undo this improvement.
                        if new_correct:
                            best_correct = True
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
                        )
                        logger.info(
                            "Pipeline sample=%s round=%d: repair committed, new verdict=%s",
                            sample.id, round_num, verdict,
                        )
                        # Repair committed — clear resolved open_issues.
                        # (Phase 3 errors were never added to open_issues.)
                        for m in actionable_mismatches:
                            open_issues.pop(m.sentence_index, None)
                else:
                    logger.warning(
                        "Pipeline sample=%s round=%d: repair partially failed",
                        sample.id, round_num,
                    )
                    # Partial failure — persist actionable mismatches only.
                    for m in actionable_mismatches:
                        wm, wd = _pick_world_for_issue(
                            m, current_mismatch_indices,
                            open_issues, witness_models, witness_domains,
                        )
                        open_issues[m.sentence_index] = (
                            m, _formula_str(m.sentence_index, premises, q), wm, wd,
                        )
            else:
                # Last round or no actionable mismatches to repair.
                # Phase 3 errors are excluded — they cannot be repaired by Phase 5.
                for m in actionable_mismatches:
                    wm, wd = _pick_world_for_issue(
                        m, current_mismatch_indices,
                        open_issues, witness_models, witness_domains,
                    )
                    open_issues[m.sentence_index] = (
                        m, _formula_str(m.sentence_index, premises, q), wm, wd,
                    )

            rounds.append(round_record)

        # Exhausted R_max rounds without full verification
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
            execution_status="success",
            verification_status=_final_verification_status,
        )
        self._write_json(out_dir / "result.json", asdict(result))
        logger.info(
            "Pipeline sample=%s UNVERIFIED after %d rounds",
            sample.id, self.config.pipeline.r_max,
        )
        return result

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
                "solver_error": a.solver_error,
                "verdict": a.verdict,
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


def _phase5_to_dict(p5: Phase5Result) -> dict:
    return {
        "all_repaired": p5.all_repaired,
        "num_local_validated": p5.num_local_validated,
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
    }


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


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


def _pick_world_for_issue(
    m: "Mismatch",
    current_mismatch_indices: set[int],
    open_issues: dict,
    witness_models: dict,
    witness_domains: dict,
) -> tuple:
    """
    Return ``(model, domain)`` to store alongside *m* in ``open_issues``.

    * Current-round mismatches (sentence_index in current_mismatch_indices):
      use this round's witness — it is the freshest world for this sentence.
    * Carried mismatches: preserve the model/domain from their original
      detection round.  Using the current-round witness would silently corrupt
      the stored world because witness_index is renumbered 0..N-1 every round,
      so witness_index=k from a previous round aliases a *different* world in
      the current round.
    """
    idx = m.sentence_index
    if idx not in current_mismatch_indices:
        prev = open_issues.get(idx)
        if prev is not None and len(prev) >= 3 and prev[2] is not None:
            return prev[2], (prev[3] if len(prev) >= 4 else None)
    return witness_models.get(m.witness_index), witness_domains.get(m.witness_index)


def _patched_p1(
    p1: Phase1Result,
    premises: list,
    q: object,
    verdict: str,
    model_info: object,
    model_info_q: object | None = None,
    raw_code: str | None = None,
) -> Phase1Result:
    """Return a shallow copy of p1 with updated premises/q/verdict/model_info.

    raw_code: if provided, replaces p1.raw_code (used to append repair notes so
    downstream Phase 5 prompts show current formula state).
    """
    from cgbv.core.phase1_formalize import Phase1Result as P1
    return P1(
        verdict=verdict,
        premises=premises,
        background_constraints=p1.background_constraints,
        bound_var_names=p1.bound_var_names,
        q=q,
        model_info=model_info,
        model_info_q=model_info_q,
        namespace=p1.namespace,
        raw_code=raw_code if raw_code is not None else p1.raw_code,
        verdict_pre_bridge=p1.verdict_pre_bridge,  # preserve original pre-bridge snapshot
        attempts=p1.attempts,
        error=None,
    )
