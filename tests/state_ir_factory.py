from __future__ import annotations

import numpy as np

from iris.schema import (
    AnchorRef,
    ApplicabilityAudit,
    Branch,
    BudgetState,
    ConstraintRelation,
    ConsistencySummary,
    ControlAction,
    ControlState,
    LemmaBinding,
    Obligation,
    ProblemAssumption,
    ProblemFrame,
    RequiredOutput,
    ScopeRef,
    StateIR,
    StateRef,
    StrategyCandidate,
    Subgoal,
    SymbolEntry,
    VerifierEvidence,
)


def make_state_ir(
    hidden_dim: int = 8,
    *,
    seed: int = 0,
    include_lm: bool = True,
    include_vs: bool = True,
) -> StateIR:
    rng = np.random.default_rng(seed)

    def vec() -> np.ndarray:
        return rng.normal(size=(hidden_dim,)).astype(np.float32)

    anchor = AnchorRef(
        anchor_id="anchor-0",
        anchor_role="problem_statement",
        support_kind="text_span",
        confidence=0.99,
    )
    scope = ScopeRef(scope_kind="global", scope_id="scope-0")
    pf = ProblemFrame(
        task_type="proof",
        target_spec="prove x = x",
        required_output=RequiredOutput(
            output_kind="proof",
            answer_channel="structured_object",
            formality_level="informal",
            verifier_mode="stacked",
        ),
        problem_assumptions=(
            ProblemAssumption(
                assumption_id="asm-0",
                normalized_claim="x is a valid symbol",
                origin_kind="problem_statement",
                status="active",
                source_anchor_refs=(anchor,),
            ),
        ),
        domain_tags=("algebra", "bootstrap"),
        source_anchor_refs=(anchor,),
        frame_status="grounded",
        vector=vec(),
    )
    sy = (
        SymbolEntry(
            sy_id="sy-x",
            surface_form="x",
            entity_kind="variable",
            scope_ref=scope,
            binding_state="bound",
            type_status="typed",
            canonical_name="x",
            type_expr="Element",
            source_anchor_refs=(anchor,),
            vector=vec(),
        ),
        SymbolEntry(
            sy_id="sy-eq",
            surface_form="=",
            entity_kind="relation_symbol",
            scope_ref=scope,
            binding_state="bound",
            type_status="typed",
            canonical_name="eq",
            type_expr="Relation",
            source_anchor_refs=(anchor,),
            vector=vec(),
        ),
    )
    cg = (
        ConstraintRelation(
            cg_id="cg-0",
            relation_type="equality",
            arguments=("sy-x", "sy-x"),
            relation_status="supported",
            qualifiers={"source": "axiom"},
            source_anchor_refs=(anchor,),
            vector=vec(),
        ),
    )
    branch = Branch(
        branch_id="br-0",
        branch_status="active",
        local_scope_ref=scope,
        strategy_family="direct_proof",
        summary="identity branch",
        vector=vec(),
    )
    subgoal = Subgoal(
        subgoal_id="sg-0",
        branch_id="br-0",
        goal_kind="prove",
        target_payload="x = x",
        goal_status="open",
        vector=vec(),
    )
    obligation = Obligation(
        obligation_id="obl-0",
        branch_id="br-0",
        attached_to_ref=StateRef(slot="FR", entry_id="sg-0"),
        obligation_kind="justify_step",
        obligation_status="open",
        required_evidence_class="local_validity",
        vector=vec(),
    )
    strategy = StrategyCandidate(
        strategy_id="strat-0",
        branch_id="br-0",
        strategy_family="direct_proof",
        candidate_status="selected",
        precondition_refs=(StateRef(slot="CG", entry_id="cg-0"),),
        score=0.75,
        vector=vec(),
    )
    fr = (branch, subgoal, obligation, strategy)

    lm = ()
    if include_lm:
        lm = (
            LemmaBinding(
                lm_id="lm-0",
                memory_kind="retrieved_lemma",
                source_ref="lemma://reflexive-equality",
                claim_signature="forall x, x = x",
                binding_map={"x": "sy-x"},
                applicability_audit=ApplicabilityAudit(
                    audit_status="applicable",
                    required_conditions=("typed_symbol",),
                    satisfied_condition_refs=(StateRef(slot="SY", entry_id="sy-x"),),
                ),
                retrieval_signal=0.82,
                source_anchor_refs=(anchor,),
                vector=vec(),
            ),
        )

    vs = ()
    if include_vs:
        evidence = VerifierEvidence(
            vs_id="vs-0",
            evidence_class="local_validity",
            target_ref=StateRef(slot="FR", entry_id="sg-0"),
            verdict="supported",
            polarity="positive",
            coverage_scope="step_local",
            strength=0.88,
            provenance_ref="verifier-stack-v1",
            linked_obligation_refs=(StateRef(slot="FR", entry_id="obl-0"),),
            vector=vec(),
        )
        vs = (
            evidence,
            ConsistencySummary(
                vs_id="vs-1",
                summary_kind="consistency",
                target_ref=StateRef(slot="FR", entry_id="sg-0"),
                based_on_vs_ids=(evidence.vs_id,),
                consistency_status="consistent",
                confidence=0.91,
                provenance_ref="verifier-stack-v1",
                vector=vec(),
            ),
        )

    continue_action = ControlAction(
        action_id="action-0",
        action_type="continue",
        target_level="L3",
        selection_score=0.62,
        action_status="selected",
    )
    cs = ControlState(
        selected_action=continue_action,
        budget_state=BudgetState(
            global_step_budget_remaining=8,
            branch_expansion_budget_remaining=2,
            verifier_probe_budget_remaining=1,
            reparse_budget_remaining=1,
        ),
        uncertainty_state="bounded",
        escalation_state="inactive",
        action_candidates=(
            continue_action,
            ControlAction(
                action_id="action-1",
                action_type="switch_strategy",
                target_level="L3",
                selection_score=0.25,
                action_status="candidate",
            ),
        ),
        recovery_target="F_PROC",
        vector=vec(),
    )
    return StateIR(PF=pf, SY=sy, CG=cg, FR=fr, LM=lm, VS=vs, CS=cs)
