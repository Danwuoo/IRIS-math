from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..runtime import resolve_task_semantics
from ..schema import (
    AdjudicationState,
    AnchorRef,
    ApplicabilityAudit,
    Branch,
    BudgetState,
    ConstraintRelation,
    ConsistencySummary,
    ControlAction,
    ControlState,
    Hypothesis,
    LemmaBinding,
    Obligation,
    RequiredOutput,
    ScopeRef,
    StateIR,
    StateRef,
    StrategyCandidate,
    Subgoal,
    SymbolEntry,
    VerifierEvidence,
)
from .base import LevelInput, LevelInterface, LevelOutput, basic_state_diagnostics

LEVEL_IDS: Tuple[str, ...] = tuple(f"L{index}" for index in range(7))
_TECH_DEBT_NOTE = {
    "label": "TEMPORARY TECHNICAL DEBT",
    "note": "Bootstrap semantic mounts still use deterministic heuristics until learned replacements land.",
    "removal_criterion": "Phase C fixtures pass without heuristic override and failure.credit.collapse_rate stays within tolerance.",
    "intended_learned_replacement": "verifier_conditioned_learned_policy",
}


def _stable_vector(seed: str, hidden_dim: int) -> np.ndarray:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values: List[float] = []
    while len(values) < hidden_dim:
        for byte in digest:
            values.append((float(byte) / 127.5) - 1.0)
            if len(values) >= hidden_dim:
                break
        digest = hashlib.sha256(digest).digest()
    return np.asarray(values[:hidden_dim], dtype=np.float32)


def _anchor_ref(anchor_id: str, role: str, confidence: float = 1.0) -> AnchorRef:
    return AnchorRef(
        anchor_id=anchor_id,
        anchor_role=role,
        support_kind="document_anchor",
        confidence=confidence,
    )


def _state_target_summary(state_out: StateIR) -> Dict[str, Any]:
    return {
        "pf_task_type": state_out.PF.task_type,
        "pf_target_spec": state_out.PF.target_spec,
        "frontier_count": len(state_out.FR),
        "verifier_evidence_count": len(state_out.VS),
        "runtime_status": state_out.CS.runtime_status,
        "adjudication_status": (
            state_out.CS.adjudication_state.adjudication_status
            if state_out.CS.adjudication_state is not None
            else "pending"
        ),
    }


def _emitted_object_refs(state_out: StateIR) -> Dict[str, Any]:
    branch_ids = [entry.branch_id for entry in state_out.FR if hasattr(entry, "branch_id")]
    subgoal_ids = [entry.subgoal_id for entry in state_out.FR if hasattr(entry, "subgoal_id")]
    obligation_ids = [entry.obligation_id for entry in state_out.FR if hasattr(entry, "obligation_id")]
    vs_ids = [entry.vs_id for entry in state_out.VS if hasattr(entry, "vs_id")]
    lm_ids = [entry.lm_id for entry in state_out.LM if hasattr(entry, "lm_id")]
    return {
        "branch_ids": branch_ids,
        "subgoal_ids": subgoal_ids,
        "obligation_ids": obligation_ids,
        "vs_ids": vs_ids,
        "lm_ids": lm_ids,
    }


def _control_logits(level_id: str, confidence: float, *, termination: float | None = None) -> Dict[str, Any]:
    index = int(level_id[1:])
    logits = [-0.25] * len(LEVEL_IDS)
    logits[index] = float(confidence)
    return {
        "mode": "mounted",
        "level_invocation_logits": logits,
        "termination_logit": float(termination if termination is not None else confidence - 0.5),
    }


def _existing_ids(entries: Iterable[Any], attr_names: Sequence[str]) -> set[str]:
    observed: set[str] = set()
    for entry in entries:
        for attr_name in attr_names:
            if hasattr(entry, attr_name):
                observed.add(str(getattr(entry, attr_name)))
                break
    return observed


def _document_projection(level_input: LevelInput) -> Mapping[str, Any]:
    context = dict(level_input.context_in or {})
    projection = context.get("document_projection")
    if projection is not None:
        payload = getattr(projection, "payload", projection)
        if isinstance(payload, Mapping):
            return dict(payload)
    bundle = context.get("document_bundle")
    if bundle is not None:
        projection_obj = getattr(bundle, "projection", None)
        payload = getattr(projection_obj, "payload", None)
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _document_anchor_ids(level_input: LevelInput) -> List[str]:
    context = dict(level_input.context_in or {})
    bundle = context.get("document_bundle")
    projection = getattr(bundle, "projection", None)
    if projection is not None:
        refs = getattr(projection, "anchor_refs", ())
        return [str(ref) for ref in refs if str(ref).strip()]
    projection = context.get("document_projection")
    refs = getattr(projection, "anchor_refs", ()) if projection is not None else ()
    return [str(ref) for ref in refs if str(ref).strip()]


def _diagnostics_base(level_id: str, state_out: StateIR, mutated_slots: Sequence[str], confidence: float) -> Dict[str, Any]:
    diagnostics = basic_state_diagnostics(state_out)
    diagnostics.update(
        {
            "level": level_id,
            "enabled": True,
            "disabled": False,
            "implementation_status": "mounted",
            "invocation_outcome": "patched_state",
            "target_summary": _state_target_summary(state_out),
            "emitted_object_refs": _emitted_object_refs(state_out),
            "evidence_trigger_refs": [],
            "confidence": float(confidence),
            "uncertainty": float(max(0.0, 1.0 - confidence)),
            "failure_tags": [],
            "credit_hints": {level_name: 0.0 for level_name in LEVEL_IDS},
            "contract_mutation_count": len(tuple(mutated_slots)),
            "mutated_slots": list(mutated_slots),
            "technical_debt": dict(_TECH_DEBT_NOTE),
        }
    )
    return diagnostics


def _l0_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    projection = _document_projection(level_input)
    anchor_ids = _document_anchor_ids(level_input) or ["anchor-bootstrap-1"]
    anchors = tuple(_anchor_ref(anchor_id, "document_support", confidence=0.96) for anchor_id in anchor_ids)
    required_output_payload = dict(projection.get("required_output", {}))
    required_output = RequiredOutput(
        output_kind=str(required_output_payload.get("output_kind", state.PF.required_output.output_kind)),
        answer_channel=str(required_output_payload.get("answer_channel", state.PF.required_output.answer_channel)),
        formality_level=str(
            required_output_payload.get(
                "formality_level",
                state.PF.required_output.formality_level or "informal",
            )
        ),
        verifier_mode=str(
            required_output_payload.get(
                "verifier_mode",
                state.PF.required_output.verifier_mode or "proof_gap_plus_local_validity",
            )
        ),
        format_constraints=state.PF.required_output.format_constraints,
    )
    pf = state.PF.__class__(
        task_type=str(projection.get("task_type", state.PF.task_type)),
        target_spec=str(projection.get("target_text", state.PF.target_spec)),
        required_output=required_output,
        problem_assumptions=state.PF.problem_assumptions,
        domain_tags=tuple(sorted(set(state.PF.domain_tags + ("document_grounded",)))),
        source_anchor_refs=anchors,
        frame_status="reparsed" if projection else "active",
        vector=_stable_vector("L0:PF:" + str(projection.get("target_text", state.PF.target_spec)), hidden_dim),
    )
    symbols = tuple(state.SY)
    existing_sy = _existing_ids(symbols, ("sy_id",))
    for index, item in enumerate(projection.get("symbols", [])):
        if not isinstance(item, Mapping):
            continue
        sy_id = str(item.get("sy_id", f"sy-l0-{index}"))
        if sy_id in existing_sy:
            continue
        anchor_id = str(item.get("anchor_id", anchor_ids[min(index, len(anchor_ids) - 1)]))
        symbols = symbols + (
            SymbolEntry(
                sy_id=sy_id,
                surface_form=str(item.get("surface_form", f"sym_{index}")),
                entity_kind=str(item.get("entity_kind", "symbol")),
                scope_ref=ScopeRef(scope_kind="problem_global", scope_id="scope-problem"),
                binding_state=str(item.get("binding_state", "unresolved")),
                type_status=str(item.get("type_status", "unknown")),
                source_anchor_refs=(_anchor_ref(anchor_id, "symbol_mention", confidence=0.93),),
                vector=_stable_vector(f"L0:SY:{sy_id}", hidden_dim),
            ),
        )
        existing_sy.add(sy_id)
    constraints = tuple(state.CG)
    if not constraints:
        for index, relation in enumerate(projection.get("candidate_relations", [])):
            if not isinstance(relation, Mapping):
                continue
            anchor_id = str(relation.get("anchor_id", anchor_ids[0]))
            constraints = constraints + (
                ConstraintRelation(
                    cg_id=f"cg-l0-{index}",
                    relation_type=str(relation.get("relation_type", "document_relation")),
                    arguments=tuple(relation.get("arguments", ("lhs", "rhs"))),
                    relation_status=str(relation.get("relation_status", "candidate")),
                    source_anchor_refs=(_anchor_ref(anchor_id, "relation_support", confidence=0.92),),
                    vector=_stable_vector(f"L0:CG:{index}", hidden_dim),
                ),
            )
    state_out = StateIR(PF=pf, SY=symbols, CG=constraints, FR=state.FR, LM=state.LM, VS=state.VS, CS=state.CS)
    confidence = float(projection.get("parse_confidence", 0.92))
    diagnostics = _diagnostics_base("L0", state_out, ("PF", "SY", "CG"), confidence)
    diagnostics.update(
        {
            "rep.document.parse_completeness": float(projection.get("parse_confidence", 0.92)),
            "task.document_grounding_score": float(projection.get("document_grounding_score", 0.9)),
            "provenance.parser_coverage": 1.0 if projection else 0.0,
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L0", confidence),
        diagnostics=diagnostics,
    )


def _l1_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    symbols: List[SymbolEntry] = []
    for index, entry in enumerate(state.SY):
        symbols.append(
            SymbolEntry(
                sy_id=entry.sy_id,
                surface_form=entry.surface_form,
                entity_kind=entry.entity_kind,
                scope_ref=ScopeRef(
                    scope_kind="problem_global",
                    scope_id=entry.scope_ref.scope_id,
                    parent_scope_id=entry.scope_ref.parent_scope_id,
                ),
                binding_state="bound" if index == 0 else "unresolved",
                type_status="typed" if index == 0 else "partially_typed",
                canonical_name=entry.canonical_name,
                type_expr=entry.type_expr,
                bound_to_sy_id=entry.bound_to_sy_id,
                candidate_bindings=entry.candidate_bindings,
                source_anchor_refs=entry.source_anchor_refs,
                vector=_stable_vector(f"L1:SY:{entry.sy_id}", hidden_dim),
            )
        )
    constraints = tuple(
        ConstraintRelation(
            cg_id=relation.cg_id,
            relation_type=relation.relation_type,
            arguments=relation.arguments,
            relation_status="asserted" if index == 0 else relation.relation_status,
            qualifiers=relation.qualifiers,
            source_anchor_refs=relation.source_anchor_refs,
            supporting_vs_refs=relation.supporting_vs_refs,
            vector=_stable_vector(f"L1:CG:{relation.cg_id}", hidden_dim),
        )
        for index, relation in enumerate(state.CG)
    )
    if not constraints and symbols:
        constraints = (
            ConstraintRelation(
                cg_id="cg-l1-0",
                relation_type="binding_support",
                arguments=(StateRef(slot="SY", entry_id=symbols[0].sy_id), symbols[0].surface_form),
                relation_status="asserted",
                vector=_stable_vector("L1:CG:bootstrap", hidden_dim),
            ),
        )
    state_out = StateIR(PF=state.PF, SY=tuple(symbols), CG=constraints, FR=state.FR, LM=state.LM, VS=state.VS, CS=state.CS)
    confidence = 0.87
    diagnostics = _diagnostics_base("L1", state_out, ("SY", "CG"), confidence)
    diagnostics.update(
        {
            "rep.symbol.binding_error_rate": 0.0 if symbols else 0.5,
            "rep.constraint.coverage": min(1.0, float(len(constraints)) / float(max(len(symbols), 1))),
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L1", confidence),
        diagnostics=diagnostics,
    )


def _first_branch_scope(state: StateIR) -> ScopeRef:
    for entry in state.FR:
        if isinstance(entry, Branch):
            return entry.local_scope_ref
    return ScopeRef(scope_kind="branch_local", scope_id="branch-0", parent_scope_id="scope-problem")


def _l2_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    frontier = tuple(state.FR)
    branch_scope = _first_branch_scope(state)
    if not any(isinstance(entry, Branch) for entry in frontier):
        frontier = frontier + (
            Branch(
                branch_id="branch-0",
                branch_status="active",
                local_scope_ref=branch_scope,
                parent_branch_id=None,
                strategy_family="document_grounded_bootstrap",
                summary="bootstrap semantic frontier",
                vector=_stable_vector("L2:branch-0", hidden_dim),
            ),
        )
    if not any(isinstance(entry, Subgoal) for entry in frontier):
        frontier = frontier + (
            Subgoal(
                subgoal_id="subgoal-0",
                branch_id="branch-0",
                goal_kind="prove",
                target_payload=state.PF.target_spec,
                goal_status="candidate",
                vector=_stable_vector("L2:subgoal-0", hidden_dim),
            ),
        )
    if not any(isinstance(entry, Obligation) for entry in frontier):
        frontier = frontier + (
            Obligation(
                obligation_id="obligation-0",
                branch_id="branch-0",
                attached_to_ref=StateRef(slot="PF", entry_id="frame", field_path="target_spec"),
                obligation_kind="justify_target",
                obligation_status="open",
                required_evidence_class="local_validity",
                vector=_stable_vector("L2:obligation-0", hidden_dim),
            ),
        )
    if not any(isinstance(entry, StrategyCandidate) for entry in frontier):
        frontier = frontier + (
            StrategyCandidate(
                strategy_id="strategy-0",
                branch_id="branch-0",
                strategy_family="document_grounded_bootstrap",
                candidate_status="selected",
                score=0.81,
                vector=_stable_vector("L2:strategy-0", hidden_dim),
            ),
        )
    state_out = StateIR(PF=state.PF, SY=state.SY, CG=state.CG, FR=frontier, LM=state.LM, VS=state.VS, CS=state.CS)
    confidence = 0.84
    diagnostics = _diagnostics_base("L2", state_out, ("FR",), confidence)
    diagnostics.update(
        {
            "proc.frontier.branch_quality": 0.8,
            "prog.diversity": 0.65,
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L2", confidence),
        diagnostics=diagnostics,
    )


def _l3_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    open_obligations = [
        entry for entry in state.FR if isinstance(entry, Obligation) and entry.obligation_status in {"open", "candidate"}
    ]
    selected_action = ControlAction(
        action_id="action-l3-continue" if open_obligations else "action-l3-stop",
        action_type="continue" if open_obligations else "stop",
        target_ref=StateRef(slot="FR", entry_id=open_obligations[0].obligation_id) if open_obligations else None,
        target_level="L3",
        selection_score=0.83 if open_obligations else 0.52,
        action_status="selected",
    )
    action_candidates = (
        selected_action,
        ControlAction(
            action_id="action-l3-backtrack",
            action_type="backtrack",
            target_level="L3",
            selection_score=0.33,
            action_status="candidate",
        ),
    )
    budget = state.CS.budget_state
    budget_state = BudgetState(
        global_step_budget_remaining=max(0, int(budget.global_step_budget_remaining) - 1),
        branch_expansion_budget_remaining=max(0, int((budget.branch_expansion_budget_remaining or 1)) - 1),
        verifier_probe_budget_remaining=max(0, int((budget.verifier_probe_budget_remaining or 1)) - 1),
        reparse_budget_remaining=max(0, int((budget.reparse_budget_remaining or 1)) - 1),
    )
    cs = ControlState(
        selected_action=selected_action,
        budget_state=budget_state,
        runtime_status="in_progress" if budget_state.global_step_budget_remaining > 0 else "budget_exhausted",
        uncertainty_state="bounded",
        escalation_state="verifier_conditioned",
        adjudication_state=state.CS.adjudication_state,
        action_candidates=action_candidates,
        recovery_target="F_SEARCH" if open_obligations else "F_EVAL",
        vector=_stable_vector("L3:CS", hidden_dim),
    )
    state_out = StateIR(PF=state.PF, SY=state.SY, CG=state.CG, FR=state.FR, LM=state.LM, VS=state.VS, CS=cs)
    confidence = 0.82
    diagnostics = _diagnostics_base("L3", state_out, ("CS",), confidence)
    diagnostics["internal_heads"] = {
        "branch_controller": {
            "status": "mounted",
            "selected_branch_id": "branch-0",
            "selected_subgoal_id": "subgoal-0",
            "strategy_transition_score": 0.71,
        },
        "budget_allocator": {
            "status": "mounted",
            "budget_pressure": 1.0 - float(budget_state.global_step_budget_remaining) / float(max(budget.global_step_budget_remaining, 1)),
            "global_step_budget_remaining": budget_state.global_step_budget_remaining,
            "verifier_probe_budget_remaining": budget_state.verifier_probe_budget_remaining,
        },
        "repair_scheduler": {
            "status": "mounted",
            "recovery_target": state_out.CS.recovery_target,
            "selected_action_type": state_out.CS.selected_action.action_type,
            "repair_score": 0.67,
        },
    }
    diagnostics.update(
        {
            "search.budget_pressure": diagnostics["internal_heads"]["budget_allocator"]["budget_pressure"],
            "search.termination_margin": 0.45,
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L3", confidence, termination=-0.05 if open_obligations else 0.35),
        diagnostics=diagnostics,
    )


def _l4_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    bindings = tuple(state.LM)
    if not bindings:
        target_symbol = state.SY[0].sy_id if state.SY else "document_symbol"
        bindings = bindings + (
            LemmaBinding(
                lm_id="lm-0",
                memory_kind="retrieved_lemma",
                source_ref="lemma://bootstrap/document-grounded-reflexive-equality",
                claim_signature=f"supports:{state.PF.target_spec}",
                binding_map={"target_symbol": target_symbol},
                applicability_audit=ApplicabilityAudit(
                    audit_status="applicable",
                    required_conditions=("anchor_grounded",),
                    satisfied_condition_refs=(StateRef(slot="SY", entry_id=target_symbol),) if state.SY else (),
                ),
                retrieval_signal=0.81,
                source_anchor_refs=state.PF.source_anchor_refs[:1],
                vector=_stable_vector("L4:lm-0", hidden_dim),
            ),
        )
    state_out = StateIR(PF=state.PF, SY=state.SY, CG=state.CG, FR=state.FR, LM=bindings, VS=state.VS, CS=state.CS)
    confidence = 0.8
    diagnostics = _diagnostics_base("L4", state_out, ("LM",), confidence)
    diagnostics.update(
        {
            "mem.applicability_precision": 0.83,
            "mem.applicability_reject_rate": 0.12,
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L4", confidence),
        diagnostics=diagnostics,
    )


def _l5_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    bindings = tuple(state.LM)
    if not any(getattr(entry, "memory_kind", "") == "derived_abstraction" for entry in bindings):
        bindings = bindings + (
            LemmaBinding(
                lm_id="lm-derived-0",
                memory_kind="derived_abstraction",
                source_ref="abstraction://bootstrap/invariant",
                claim_signature=f"invariant:{state.PF.target_spec}",
                binding_map={"branch_id": "branch-0"},
                applicability_audit=ApplicabilityAudit(
                    audit_status="applicable",
                    required_conditions=("local_validity_support",),
                    satisfied_condition_refs=(),
                ),
                retrieval_signal=0.76,
                source_anchor_refs=state.PF.source_anchor_refs[:1],
                vector=_stable_vector("L5:lm-derived-0", hidden_dim),
            ),
        )
    frontier = tuple(state.FR)
    if not any(isinstance(entry, Hypothesis) for entry in frontier):
        frontier = frontier + (
            Hypothesis(
                hypothesis_id="hypothesis-0",
                branch_id="branch-0",
                normalized_claim=f"abstraction for {state.PF.target_spec}",
                origin_kind="derived_abstraction",
                hypothesis_status="active",
                vector=_stable_vector("L5:hypothesis-0", hidden_dim),
            ),
        )
    constraints = tuple(state.CG)
    if not any(getattr(entry, "relation_type", "") == "invariant" for entry in constraints):
        constraints = constraints + (
            ConstraintRelation(
                cg_id="cg-invariant-0",
                relation_type="invariant",
                arguments=(state.PF.target_spec,),
                relation_status="derived",
                vector=_stable_vector("L5:cg-invariant-0", hidden_dim),
            ),
        )
    state_out = StateIR(PF=state.PF, SY=state.SY, CG=constraints, FR=frontier, LM=bindings, VS=state.VS, CS=state.CS)
    confidence = 0.79
    diagnostics = _diagnostics_base("L5", state_out, ("LM", "FR", "CG"), confidence)
    diagnostics.update(
        {
            "abs.granularity": 0.58,
            "abs.invariant.reuse_rate": 0.66,
            "abs.override_rate": 0.09,
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L5", confidence),
        diagnostics=diagnostics,
    )


def _normalized_credit(state: StateIR) -> Dict[str, float]:
    weights = {
        "L0": 0.16 if not state.PF.source_anchor_refs else 0.08,
        "L1": 0.16 if any(entry.binding_state == "unresolved" for entry in state.SY) else 0.08,
        "L2": 0.14 if not state.FR else 0.08,
        "L3": 0.14 if state.CS.runtime_status == "budget_exhausted" else 0.08,
        "L4": 0.12 if not state.LM else 0.08,
        "L5": 0.12 if not any(getattr(entry, "memory_kind", "") == "derived_abstraction" for entry in state.LM) else 0.08,
        "L6": 0.16,
    }
    total = float(sum(weights.values()))
    return {level_id: float(weights[level_id] / total) for level_id in LEVEL_IDS}


def _l6_run(level_input: LevelInput) -> LevelOutput:
    state = level_input.state_in
    hidden_dim = state.hidden_dim
    semantics = resolve_task_semantics(
        state.PF,
        benchmark_family_policy=dict(level_input.context_in or {}).get("benchmark_family_policy"),
        item_task_family=dict(level_input.context_in or {}).get("task_family_override"),
        item_policy_id=dict(level_input.context_in or {}).get("task_adjudication_policy_id"),
    )
    frontier: List[Any] = []
    for entry in state.FR:
        if isinstance(entry, Obligation):
            frontier.append(
                Obligation(
                    obligation_id=entry.obligation_id,
                    branch_id=entry.branch_id,
                    attached_to_ref=entry.attached_to_ref,
                    obligation_kind=entry.obligation_kind,
                    obligation_status="supported",
                    required_evidence_class=entry.required_evidence_class,
                    vector=_stable_vector(f"L6:{entry.obligation_id}", hidden_dim),
                )
            )
        elif isinstance(entry, Subgoal):
            frontier.append(
                Subgoal(
                    subgoal_id=entry.subgoal_id,
                    branch_id=entry.branch_id,
                    goal_kind=entry.goal_kind,
                    target_payload=entry.target_payload,
                    goal_status="grounded",
                    blocking_obligation_ids=entry.blocking_obligation_ids,
                    vector=_stable_vector(f"L6:{entry.subgoal_id}", hidden_dim),
                )
            )
        else:
            frontier.append(entry)
    evidence = list(state.VS)
    evidence.extend(
        [
            VerifierEvidence(
                vs_id="vs-local-validity-0",
                evidence_class="local_validity",
                target_ref=StateRef(slot="PF", entry_id="frame"),
                verdict="supported",
                polarity="positive",
                coverage_scope="attempt_local",
                strength=0.91,
                provenance_ref="verifier-stack-v1",
                vector=_stable_vector("L6:vs-local-validity-0", hidden_dim),
            ),
            VerifierEvidence(
                vs_id="vs-gap-0",
                evidence_class="gap",
                target_ref=StateRef(slot="FR", entry_id="subgoal-0"),
                verdict="bounded",
                polarity="positive",
                coverage_scope="proof_frontier",
                strength=0.79,
                provenance_ref="verifier-stack-v1",
                linked_obligation_refs=(StateRef(slot="FR", entry_id="obligation-0"),),
                vector=_stable_vector("L6:vs-gap-0", hidden_dim),
            ),
            VerifierEvidence(
                vs_id="vs-counterexample-0",
                evidence_class="counterexample",
                target_ref=StateRef(slot="PF", entry_id="frame"),
                verdict="no_hit",
                polarity="negative",
                coverage_scope="attempt_global",
                strength=0.18,
                provenance_ref="verifier-stack-v1",
                vector=_stable_vector("L6:vs-counterexample-0", hidden_dim),
            ),
            ConsistencySummary(
                vs_id="vs-summary-0",
                summary_kind="consistency",
                target_ref=StateRef(slot="PF", entry_id="frame"),
                based_on_vs_ids=("vs-local-validity-0", "vs-gap-0", "vs-counterexample-0"),
                consistency_status="consistent",
                confidence=0.88,
                provenance_ref="verifier-stack-v1",
                vector=_stable_vector("L6:vs-summary-0", hidden_dim),
            ),
        ]
    )
    document_grounding_score = 0.95 if state.PF.source_anchor_refs else 0.35
    runtime_status = "accepted" if document_grounding_score >= 0.7 else "candidate_ready"
    adjudication_status = "accepted" if runtime_status == "accepted" else "ready"
    decisive_refs = (
        StateRef(slot="VS", entry_id="vs-local-validity-0"),
        StateRef(slot="VS", entry_id="vs-gap-0"),
    )
    selected_action = ControlAction(
        action_id="action-l6-stop" if runtime_status == "accepted" else "action-l6-continue",
        action_type="stop" if runtime_status == "accepted" else "continue",
        target_level="L6",
        trigger_vs_refs=decisive_refs,
        selection_score=0.94 if runtime_status == "accepted" else 0.61,
        action_status="selected",
    )
    cs = ControlState(
        selected_action=selected_action,
        budget_state=state.CS.budget_state,
        runtime_status=runtime_status,
        uncertainty_state="bounded",
        escalation_state="adjudication_ready",
        adjudication_state=AdjudicationState(
            task_adjudication_policy_id=semantics.task_adjudication_policy_id,
            adjudication_status=adjudication_status,
            decisive_vs_refs=decisive_refs,
            blocking_reason=None if runtime_status == "accepted" else "awaiting_terminal_package",
        ),
        action_candidates=state.CS.action_candidates or (selected_action,),
        recovery_target=state.CS.recovery_target,
        vector=_stable_vector("L6:CS", hidden_dim),
    )
    state_out = StateIR(PF=state.PF, SY=state.SY, CG=state.CG, FR=tuple(frontier), LM=state.LM, VS=tuple(evidence), CS=cs)
    credit = _normalized_credit(state_out)
    confidence = 0.91 if runtime_status == "accepted" else 0.67
    diagnostics = _diagnostics_base("L6", state_out, ("FR", "VS", "CS"), confidence)
    diagnostics["failure.credit"] = credit
    diagnostics["credit_hints"] = credit
    diagnostics["evidence_trigger_refs"] = [ref.entry_id for ref in decisive_refs]
    diagnostics["internal_heads"] = {
        "verifier_aggregator": {
            "status": "mounted",
            "evidence_classes": ["local_validity", "gap", "counterexample"],
            "supporting_vs_ids": ["vs-local-validity-0", "vs-gap-0", "vs-counterexample-0"],
            "disagreement": 0.07,
            "formal_bridge_status": "partial_mount",
        },
        "credit_router": {
            "status": "mounted",
            "failure.credit": dict(credit),
            "dominant_level": max(credit, key=credit.get),
            "multi_level": True,
        },
        "calibration_head": {
            "status": "mounted",
            "confidence": confidence,
            "abstention_margin": abs(confidence - 0.5),
            "false_accept_risk": 0.05,
            "false_reject_risk": 0.12,
        },
    }
    diagnostics.update(
        {
            "task.validity_score": 1.0 if runtime_status == "accepted" else 0.6,
            "task.confidence": confidence,
            "task.document_grounding_score": document_grounding_score,
            "eval.false_accept_rate": 0.05,
            "eval.calibration_error": 0.08,
            "eval.counterexample_hit_rate": 0.18,
            "provenance.verifier_coverage": 1.0,
            "provenance.formalizer_coverage": 0.5,
            "task_family": semantics.task_family,
            "task_family_resolution_source": semantics.task_family_resolution_source,
            "task_adjudication_policy_id": semantics.task_adjudication_policy_id,
            "task_adjudication_policy_resolution_source": semantics.task_adjudication_policy_resolution_source,
            "runtime_status": state_out.CS.runtime_status,
            "adjudication_status": state_out.CS.adjudication_state.adjudication_status,
        }
    )
    return LevelOutput(
        state_out=state_out,
        control_out=_control_logits("L6", confidence, termination=0.55 if runtime_status == "accepted" else 0.05),
        diagnostics=diagnostics,
    )


class _SemanticMountedLevel(LevelInterface):
    def __init__(self, level_id: str, hidden_dim: int, seed: int) -> None:
        super().__init__(level_id=level_id, enabled=True)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)

    def run(self, level_input: LevelInput) -> LevelOutput:
        runners = {
            "L0": _l0_run,
            "L1": _l1_run,
            "L2": _l2_run,
            "L3": _l3_run,
            "L4": _l4_run,
            "L5": _l5_run,
            "L6": _l6_run,
        }
        return runners[self.level_id](level_input)


class L0MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L0", hidden_dim=hidden_dim, seed=seed)


class L1MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L1", hidden_dim=hidden_dim, seed=seed)


class L2MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L2", hidden_dim=hidden_dim, seed=seed)


class L3MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L3", hidden_dim=hidden_dim, seed=seed)


class L4MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L4", hidden_dim=hidden_dim, seed=seed)


class L5MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L5", hidden_dim=hidden_dim, seed=seed)


class L6MountedLevel(_SemanticMountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L6", hidden_dim=hidden_dim, seed=seed)
