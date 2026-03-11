from __future__ import annotations

import hashlib

import numpy as np

from ..schema import (
    ApplicabilityAudit,
    Branch,
    BudgetState,
    ConstraintRelation,
    ConsistencySummary,
    ControlAction,
    ControlState,
    LemmaBinding,
    ProblemFrame,
    RequiredOutput,
    ScopeRef,
    StateIR,
    StateRef,
    Subgoal,
    SymbolEntry,
)


def dataset_slice_id_for_segment(segment_id: int) -> str:
    return f"slice-{segment_id:06d}"


def _stable_seed(
    run_id: str,
    dataset_slice_id: str,
    segment_id: int,
    micro_step_idx: int,
    data_seed: int,
) -> int:
    payload = (
        f"{run_id}|{dataset_slice_id}|{segment_id}|{micro_step_idx}|{data_seed}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def generate_synthetic_state(
    run_id: str,
    dataset_slice_id: str,
    segment_id: int,
    micro_step_idx: int,
    hidden_dim: int,
    data_seed: int,
) -> StateIR:
    seed = _stable_seed(
        run_id=run_id,
        dataset_slice_id=dataset_slice_id,
        segment_id=segment_id,
        micro_step_idx=micro_step_idx,
        data_seed=data_seed,
    )
    rng = np.random.default_rng(seed)
    symbol_count = int(rng.integers(1, 4))
    relation_count = int(rng.integers(0, 4))
    subgoal_count = int(rng.integers(0, 3))
    memory_count = int(rng.integers(0, 2))
    verifier_count = int(rng.integers(0, 2))
    problem_scope = ScopeRef(scope_kind="problem_global", scope_id="scope-problem")
    branch_scope = ScopeRef(
        scope_kind="branch_local",
        scope_id="branch-0",
        parent_scope_id=problem_scope.scope_id,
    )
    pf_vector = rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32)
    cs_vector = rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32)
    return StateIR(
        PF=ProblemFrame(
            task_type="prove",
            target_spec=f"synthetic-{segment_id}-{micro_step_idx}",
            required_output=RequiredOutput(
                output_kind="proof",
                answer_channel="structured_object",
                formality_level="semi-formal",
            ),
            problem_assumptions=(),
            domain_tags=("synthetic", "bootstrap"),
            source_anchor_refs=(),
            frame_status="active",
            vector=pf_vector,
        ),
        SY=tuple(
            SymbolEntry(
                sy_id=f"sy-{index}",
                surface_form=f"x_{index}",
                entity_kind="variable",
                scope_ref=problem_scope,
                binding_state="free",
                type_status="unknown",
                vector=rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32),
            )
            for index in range(symbol_count)
        ),
        CG=tuple(
            ConstraintRelation(
                cg_id=f"cg-{index}",
                relation_type="dependency",
                arguments=(f"x_{index}", f"x_{(index + 1) % max(symbol_count, 1)}"),
                relation_status="candidate",
                vector=rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32),
            )
            for index in range(relation_count)
        ),
        FR=tuple(
            [
                Branch(
                    branch_id="branch-0",
                    branch_status="active",
                    local_scope_ref=branch_scope,
                    strategy_family="synthetic-search",
                    summary="synthetic bootstrap branch",
                    vector=rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32),
                )
            ]
            + [
                Subgoal(
                    subgoal_id=f"subgoal-{index}",
                    branch_id="branch-0",
                    goal_kind="prove_claim",
                    target_payload=f"claim_{index}",
                    goal_status="candidate",
                    vector=rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32),
                )
                for index in range(subgoal_count)
            ]
        ),
        LM=tuple(
            LemmaBinding(
                lm_id=f"lm-{index}",
                memory_kind="pattern",
                source_ref="synthetic-memory",
                claim_signature=f"pattern-{index}",
                binding_map={},
                applicability_audit=ApplicabilityAudit(
                    audit_status="unchecked",
                    required_conditions=(),
                ),
                retrieval_signal=float(rng.random()),
                vector=rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32),
            )
            for index in range(memory_count)
        ),
        VS=tuple(
            ConsistencySummary(
                vs_id=f"vs-{index}",
                summary_kind="branch_consistency",
                target_ref=StateRef(slot="FR", entry_id="branch-0"),
                based_on_vs_ids=(f"seed-{index}",),
                consistency_status="stable",
                confidence=float(rng.random()),
                provenance_ref="synthetic-verifier-v1",
                vector=rng.normal(0.0, 1.0, (hidden_dim,)).astype(np.float32),
            )
            for index in range(verifier_count)
        ),
        CS=ControlState(
            selected_action=ControlAction(
                action_id="action-continue",
                action_type="continue",
                action_status="selected",
                target_ref=StateRef(slot="FR", entry_id="branch-0"),
            ),
            budget_state=BudgetState(
                global_step_budget_remaining=int(rng.integers(0, 8)),
                branch_expansion_budget_remaining=int(rng.integers(0, 4)),
                verifier_probe_budget_remaining=int(rng.integers(0, 3)),
            ),
            uncertainty_state="synthetic_estimate",
            escalation_state="inactive",
            vector=cs_vector,
        ),
    )
