from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence, Tuple, TypeAlias, Union

import numpy as np

STATE_IR_SLOT_ORDER: Tuple[str, ...] = ("PF", "SY", "CG", "FR", "LM", "VS", "CS")
STATE_IR_TOKEN_ORDER: Tuple[str, ...] = STATE_IR_SLOT_ORDER
CANONICAL_SCOPE_KINDS: Tuple[str, ...] = ("problem_global", "branch_local", "quote_local")
CANONICAL_SCOPE_KIND_ALIASES: Dict[str, str] = {
    "global": "problem_global",
    "problem": "problem_global",
    "branch": "branch_local",
    "quote": "quote_local",
}
CANONICAL_ACTION_TYPES: Tuple[str, ...] = (
    "continue",
    "backtrack",
    "reparse",
    "switch_strategy",
    "stop",
)
CANONICAL_ACTION_TYPE_ALIASES: Dict[str, str] = {
    "switch": "switch_strategy",
    "retry": "reparse",
    "halt": "stop",
}
CANONICAL_RUNTIME_STATUSES: Tuple[str, ...] = (
    "in_progress",
    "candidate_ready",
    "accepted",
    "rejected",
    "abstained",
    "budget_exhausted",
)
CANONICAL_RUNTIME_STATUS_ALIASES: Dict[str, str] = {
    "running": "in_progress",
    "done": "candidate_ready",
    "need_more_check": "candidate_ready",
    "accepted_with_low_margin": "accepted",
    "fail": "rejected",
    "failed": "rejected",
    "timeout": "budget_exhausted",
}
CANONICAL_ADJUDICATION_STATUSES: Tuple[str, ...] = (
    "pending",
    "ready",
    "accepted",
    "rejected",
    "abstained",
    "blocked",
)
CANONICAL_ADJUDICATION_STATUS_ALIASES: Dict[str, str] = {
    "need_more_check": "pending",
    "in_review": "ready",
    "fail": "rejected",
}


class StateIRValidationError(ValueError):
    pass


def _normalize_vector(slot_name: str, entry_id: str, vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise StateIRValidationError(
            f"{slot_name}:{entry_id} vector must be rank-1, got ndim={array.ndim}."
        )
    if array.shape[0] <= 0:
        raise StateIRValidationError(f"{slot_name}:{entry_id} vector must not be empty.")
    return array


def _tuple_of_str(values: Sequence[Any] | None) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _tuple_of_any(values: Sequence[Any] | None) -> Tuple[Any, ...]:
    if values is None:
        return ()
    return tuple(values)


def _require_text(field_name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise StateIRValidationError(f"{field_name} is required.")
    return text


def _normalize_choice(
    field_name: str,
    value: str,
    *,
    allowed: Sequence[str],
    aliases: Mapping[str, str] | None = None,
) -> str:
    text = _require_text(field_name, value)
    canonical = dict(aliases or {}).get(text, text)
    if canonical not in set(allowed):
        raise StateIRValidationError(
            f"{field_name} must be one of {tuple(allowed)}, got {canonical!r}."
        )
    return canonical


@dataclass(frozen=True)
class StateRef:
    slot: str
    entry_id: str
    field_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _require_text("state_ref.slot", self.slot))
        object.__setattr__(self, "entry_id", _require_text("state_ref.entry_id", self.entry_id))
        if self.field_path is not None:
            object.__setattr__(self, "field_path", str(self.field_path))


@dataclass(frozen=True)
class ScopeRef:
    scope_kind: str
    scope_id: str
    parent_scope_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope_kind",
            _normalize_choice(
                "scope_ref.scope_kind",
                self.scope_kind,
                allowed=CANONICAL_SCOPE_KINDS,
                aliases=CANONICAL_SCOPE_KIND_ALIASES,
            ),
        )
        object.__setattr__(self, "scope_id", _require_text("scope_ref.scope_id", self.scope_id))
        if self.parent_scope_id is not None:
            object.__setattr__(self, "parent_scope_id", str(self.parent_scope_id))


@dataclass(frozen=True)
class AnchorRef:
    anchor_id: str
    anchor_role: str
    support_kind: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", _require_text("anchor_ref.anchor_id", self.anchor_id))
        object.__setattr__(self, "anchor_role", _require_text("anchor_ref.anchor_role", self.anchor_role))
        object.__setattr__(self, "support_kind", _require_text("anchor_ref.support_kind", self.support_kind))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class RequiredOutput:
    output_kind: str
    answer_channel: str
    formality_level: str | None = None
    format_constraints: Tuple[str, ...] = ()
    verifier_mode: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_kind",
            _require_text("required_output.output_kind", self.output_kind),
        )
        object.__setattr__(
            self,
            "answer_channel",
            _require_text("required_output.answer_channel", self.answer_channel),
        )
        object.__setattr__(self, "format_constraints", _tuple_of_str(self.format_constraints))
        if self.formality_level is not None:
            object.__setattr__(self, "formality_level", str(self.formality_level))
        if self.verifier_mode is not None:
            object.__setattr__(self, "verifier_mode", str(self.verifier_mode))


@dataclass(frozen=True)
class ProblemAssumption:
    assumption_id: str
    normalized_claim: str
    origin_kind: str
    status: str
    source_anchor_refs: Tuple[AnchorRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assumption_id",
            _require_text("problem_assumption.assumption_id", self.assumption_id),
        )
        object.__setattr__(
            self,
            "normalized_claim",
            _require_text("problem_assumption.normalized_claim", self.normalized_claim),
        )
        object.__setattr__(self, "origin_kind", _require_text("problem_assumption.origin_kind", self.origin_kind))
        object.__setattr__(self, "status", _require_text("problem_assumption.status", self.status))
        object.__setattr__(self, "source_anchor_refs", tuple(self.source_anchor_refs))


@dataclass(frozen=True)
class ProblemFrame:
    task_type: str
    target_spec: str
    required_output: RequiredOutput
    problem_assumptions: Tuple[ProblemAssumption, ...]
    domain_tags: Tuple[str, ...]
    source_anchor_refs: Tuple[AnchorRef, ...]
    frame_status: str
    vector: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_type", _require_text("PF.task_type", self.task_type))
        object.__setattr__(self, "target_spec", _require_text("PF.target_spec", self.target_spec))
        if not isinstance(self.required_output, RequiredOutput):
            raise StateIRValidationError("PF.required_output must be a RequiredOutput object.")
        object.__setattr__(self, "problem_assumptions", tuple(self.problem_assumptions))
        object.__setattr__(self, "domain_tags", _tuple_of_str(self.domain_tags))
        object.__setattr__(self, "source_anchor_refs", tuple(self.source_anchor_refs))
        object.__setattr__(self, "frame_status", _require_text("PF.frame_status", self.frame_status))
        object.__setattr__(self, "vector", _normalize_vector("PF", "frame", self.vector))


@dataclass(frozen=True)
class SymbolEntry:
    sy_id: str
    surface_form: str
    entity_kind: str
    scope_ref: ScopeRef
    binding_state: str
    type_status: str
    canonical_name: str | None = None
    type_expr: str | None = None
    bound_to_sy_id: str | None = None
    candidate_bindings: Tuple[str, ...] = ()
    source_anchor_refs: Tuple[AnchorRef, ...] = ()
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sy_id", _require_text("SY.sy_id", self.sy_id))
        object.__setattr__(self, "surface_form", _require_text("SY.surface_form", self.surface_form))
        object.__setattr__(self, "entity_kind", _require_text("SY.entity_kind", self.entity_kind))
        if not isinstance(self.scope_ref, ScopeRef):
            raise StateIRValidationError("SY.scope_ref must be a ScopeRef object.")
        object.__setattr__(self, "binding_state", _require_text("SY.binding_state", self.binding_state))
        object.__setattr__(self, "type_status", _require_text("SY.type_status", self.type_status))
        object.__setattr__(self, "candidate_bindings", _tuple_of_str(self.candidate_bindings))
        object.__setattr__(self, "source_anchor_refs", tuple(self.source_anchor_refs))
        if self.canonical_name is not None:
            object.__setattr__(self, "canonical_name", str(self.canonical_name))
        if self.type_expr is not None:
            object.__setattr__(self, "type_expr", str(self.type_expr))
        if self.bound_to_sy_id is not None:
            object.__setattr__(self, "bound_to_sy_id", str(self.bound_to_sy_id))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("SY", self.sy_id, vector))


@dataclass(frozen=True)
class ConstraintRelation:
    cg_id: str
    relation_type: str
    arguments: Tuple[Any, ...]
    relation_status: str
    qualifiers: Mapping[str, Any] | None = None
    source_anchor_refs: Tuple[AnchorRef, ...] = ()
    supporting_vs_refs: Tuple[StateRef, ...] = ()
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cg_id", _require_text("CG.cg_id", self.cg_id))
        object.__setattr__(self, "relation_type", _require_text("CG.relation_type", self.relation_type))
        arguments = _tuple_of_any(self.arguments)
        if not arguments:
            raise StateIRValidationError("CG.arguments must not be empty.")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "relation_status", _require_text("CG.relation_status", self.relation_status))
        object.__setattr__(self, "qualifiers", dict(self.qualifiers or {}))
        object.__setattr__(self, "source_anchor_refs", tuple(self.source_anchor_refs))
        object.__setattr__(self, "supporting_vs_refs", tuple(self.supporting_vs_refs))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("CG", self.cg_id, vector))


@dataclass(frozen=True)
class Branch:
    branch_id: str
    branch_status: str
    local_scope_ref: ScopeRef
    parent_branch_id: str | None = None
    strategy_family: str | None = None
    summary: str | None = None
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _require_text("FR.branch_id", self.branch_id))
        object.__setattr__(self, "branch_status", _require_text("FR.branch_status", self.branch_status))
        if not isinstance(self.local_scope_ref, ScopeRef):
            raise StateIRValidationError("FR.branch.local_scope_ref must be a ScopeRef object.")
        if self.parent_branch_id is not None:
            object.__setattr__(self, "parent_branch_id", str(self.parent_branch_id))
        if self.strategy_family is not None:
            object.__setattr__(self, "strategy_family", str(self.strategy_family))
        if self.summary is not None:
            object.__setattr__(self, "summary", str(self.summary))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("FR", self.branch_id, vector))


@dataclass(frozen=True)
class Subgoal:
    subgoal_id: str
    branch_id: str
    goal_kind: str
    target_payload: str
    goal_status: str
    blocking_obligation_ids: Tuple[str, ...] = ()
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subgoal_id", _require_text("FR.subgoal_id", self.subgoal_id))
        object.__setattr__(self, "branch_id", _require_text("FR.subgoal.branch_id", self.branch_id))
        object.__setattr__(self, "goal_kind", _require_text("FR.subgoal.goal_kind", self.goal_kind))
        object.__setattr__(
            self,
            "target_payload",
            _require_text("FR.subgoal.target_payload", self.target_payload),
        )
        object.__setattr__(self, "goal_status", _require_text("FR.subgoal.goal_status", self.goal_status))
        object.__setattr__(self, "blocking_obligation_ids", _tuple_of_str(self.blocking_obligation_ids))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("FR", self.subgoal_id, vector))


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    branch_id: str
    attached_to_ref: StateRef
    obligation_kind: str
    obligation_status: str
    required_evidence_class: str | None = None
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_id",
            _require_text("FR.obligation_id", self.obligation_id),
        )
        object.__setattr__(self, "branch_id", _require_text("FR.obligation.branch_id", self.branch_id))
        if not isinstance(self.attached_to_ref, StateRef):
            raise StateIRValidationError("FR.obligation.attached_to_ref must be a StateRef object.")
        object.__setattr__(
            self,
            "obligation_kind",
            _require_text("FR.obligation.obligation_kind", self.obligation_kind),
        )
        object.__setattr__(
            self,
            "obligation_status",
            _require_text("FR.obligation.obligation_status", self.obligation_status),
        )
        if self.required_evidence_class is not None:
            object.__setattr__(self, "required_evidence_class", str(self.required_evidence_class))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("FR", self.obligation_id, vector))


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    branch_id: str
    normalized_claim: str
    origin_kind: str
    hypothesis_status: str
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hypothesis_id",
            _require_text("FR.hypothesis_id", self.hypothesis_id),
        )
        object.__setattr__(self, "branch_id", _require_text("FR.hypothesis.branch_id", self.branch_id))
        object.__setattr__(
            self,
            "normalized_claim",
            _require_text("FR.hypothesis.normalized_claim", self.normalized_claim),
        )
        object.__setattr__(
            self,
            "origin_kind",
            _require_text("FR.hypothesis.origin_kind", self.origin_kind),
        )
        object.__setattr__(
            self,
            "hypothesis_status",
            _require_text("FR.hypothesis.hypothesis_status", self.hypothesis_status),
        )
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("FR", self.hypothesis_id, vector))


@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id: str
    branch_id: str
    strategy_family: str
    candidate_status: str
    precondition_refs: Tuple[StateRef, ...] = ()
    score: float | None = None
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy_id",
            _require_text("FR.strategy_id", self.strategy_id),
        )
        object.__setattr__(self, "branch_id", _require_text("FR.strategy.branch_id", self.branch_id))
        object.__setattr__(
            self,
            "strategy_family",
            _require_text("FR.strategy.strategy_family", self.strategy_family),
        )
        object.__setattr__(
            self,
            "candidate_status",
            _require_text("FR.strategy.candidate_status", self.candidate_status),
        )
        object.__setattr__(self, "precondition_refs", tuple(self.precondition_refs))
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("FR", self.strategy_id, vector))


FrontierEntry: TypeAlias = Union[Branch, Subgoal, Obligation, Hypothesis, StrategyCandidate]


@dataclass(frozen=True)
class ApplicabilityAudit:
    audit_status: str
    required_conditions: Tuple[str, ...]
    satisfied_condition_refs: Tuple[StateRef, ...] = ()
    mismatch_reasons: Tuple[str, ...] = ()
    verifier_evidence_refs: Tuple[StateRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_status",
            _require_text("LM.applicability_audit.audit_status", self.audit_status),
        )
        object.__setattr__(self, "required_conditions", _tuple_of_str(self.required_conditions))
        object.__setattr__(self, "satisfied_condition_refs", tuple(self.satisfied_condition_refs))
        object.__setattr__(self, "mismatch_reasons", _tuple_of_str(self.mismatch_reasons))
        object.__setattr__(self, "verifier_evidence_refs", tuple(self.verifier_evidence_refs))


@dataclass(frozen=True)
class LemmaBinding:
    lm_id: str
    memory_kind: str
    source_ref: str
    claim_signature: str
    binding_map: Mapping[str, str]
    applicability_audit: ApplicabilityAudit
    retrieval_signal: float | None = None
    source_anchor_refs: Tuple[AnchorRef, ...] = ()
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lm_id", _require_text("LM.lm_id", self.lm_id))
        object.__setattr__(self, "memory_kind", _require_text("LM.memory_kind", self.memory_kind))
        object.__setattr__(self, "source_ref", _require_text("LM.source_ref", self.source_ref))
        object.__setattr__(
            self,
            "claim_signature",
            _require_text("LM.claim_signature", self.claim_signature),
        )
        object.__setattr__(self, "binding_map", {str(key): str(value) for key, value in self.binding_map.items()})
        if not isinstance(self.applicability_audit, ApplicabilityAudit):
            raise StateIRValidationError("LM.applicability_audit must be an ApplicabilityAudit object.")
        if self.retrieval_signal is not None:
            object.__setattr__(self, "retrieval_signal", float(self.retrieval_signal))
        object.__setattr__(self, "source_anchor_refs", tuple(self.source_anchor_refs))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("LM", self.lm_id, vector))


@dataclass(frozen=True)
class VerifierEvidence:
    vs_id: str
    evidence_class: str
    target_ref: StateRef
    verdict: str
    polarity: str
    coverage_scope: str
    strength: float
    provenance_ref: str
    linked_obligation_refs: Tuple[StateRef, ...] = ()
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vs_id", _require_text("VS.vs_id", self.vs_id))
        object.__setattr__(
            self,
            "evidence_class",
            _require_text("VS.evidence_class", self.evidence_class),
        )
        if not isinstance(self.target_ref, StateRef):
            raise StateIRValidationError("VS.target_ref must be a StateRef object.")
        object.__setattr__(self, "verdict", _require_text("VS.verdict", self.verdict))
        object.__setattr__(self, "polarity", _require_text("VS.polarity", self.polarity))
        object.__setattr__(
            self,
            "coverage_scope",
            _require_text("VS.coverage_scope", self.coverage_scope),
        )
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(
            self,
            "provenance_ref",
            _require_text("VS.provenance_ref", self.provenance_ref),
        )
        object.__setattr__(self, "linked_obligation_refs", tuple(self.linked_obligation_refs))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("VS", self.vs_id, vector))


@dataclass(frozen=True)
class ConsistencySummary:
    vs_id: str
    summary_kind: str
    target_ref: StateRef
    based_on_vs_ids: Tuple[str, ...]
    consistency_status: str
    confidence: float
    provenance_ref: str
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vs_id", _require_text("VS.summary.vs_id", self.vs_id))
        object.__setattr__(
            self,
            "summary_kind",
            _require_text("VS.summary.summary_kind", self.summary_kind),
        )
        if not isinstance(self.target_ref, StateRef):
            raise StateIRValidationError("VS.summary.target_ref must be a StateRef object.")
        based_on_vs_ids = _tuple_of_str(self.based_on_vs_ids)
        if not based_on_vs_ids:
            raise StateIRValidationError("VS.summary.based_on_vs_ids must not be empty.")
        object.__setattr__(self, "based_on_vs_ids", based_on_vs_ids)
        object.__setattr__(
            self,
            "consistency_status",
            _require_text("VS.summary.consistency_status", self.consistency_status),
        )
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(
            self,
            "provenance_ref",
            _require_text("VS.summary.provenance_ref", self.provenance_ref),
        )
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("VS", self.vs_id, vector))


VerifierEntry: TypeAlias = Union[VerifierEvidence, ConsistencySummary]


@dataclass(frozen=True)
class ControlAction:
    action_id: str
    action_type: str
    target_ref: StateRef | None = None
    target_level: str | None = None
    trigger_vs_refs: Tuple[StateRef, ...] = ()
    selection_score: float | None = None
    action_status: str = "selected"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_id",
            _require_text("CS.selected_action.action_id", self.action_id),
        )
        object.__setattr__(
            self,
            "action_type",
            _normalize_choice(
                "CS.selected_action.action_type",
                self.action_type,
                allowed=CANONICAL_ACTION_TYPES,
                aliases=CANONICAL_ACTION_TYPE_ALIASES,
            ),
        )
        if self.target_ref is not None and not isinstance(self.target_ref, StateRef):
            raise StateIRValidationError("CS.selected_action.target_ref must be a StateRef object.")
        if self.target_level is not None:
            object.__setattr__(self, "target_level", str(self.target_level))
        object.__setattr__(self, "trigger_vs_refs", tuple(self.trigger_vs_refs))
        if self.selection_score is not None:
            object.__setattr__(self, "selection_score", float(self.selection_score))
        object.__setattr__(
            self,
            "action_status",
            _require_text("CS.selected_action.action_status", self.action_status),
        )


@dataclass(frozen=True)
class BudgetState:
    global_step_budget_remaining: int
    branch_expansion_budget_remaining: int | None = None
    verifier_probe_budget_remaining: int | None = None
    reparse_budget_remaining: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_step_budget_remaining",
            int(self.global_step_budget_remaining),
        )
        if self.global_step_budget_remaining < 0:
            raise StateIRValidationError("CS.budget_state.global_step_budget_remaining must be >= 0.")
        if self.branch_expansion_budget_remaining is not None:
            object.__setattr__(
                self,
                "branch_expansion_budget_remaining",
                int(self.branch_expansion_budget_remaining),
            )
        if self.verifier_probe_budget_remaining is not None:
            object.__setattr__(
                self,
                "verifier_probe_budget_remaining",
                int(self.verifier_probe_budget_remaining),
            )
        if self.reparse_budget_remaining is not None:
            object.__setattr__(self, "reparse_budget_remaining", int(self.reparse_budget_remaining))


@dataclass(frozen=True)
class AdjudicationState:
    task_adjudication_policy_id: str
    adjudication_status: str
    decisive_vs_refs: Tuple[StateRef, ...] = ()
    blocking_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_adjudication_policy_id",
            _require_text(
                "CS.adjudication_state.task_adjudication_policy_id",
                self.task_adjudication_policy_id,
            ),
        )
        object.__setattr__(
            self,
            "adjudication_status",
            _normalize_choice(
                "CS.adjudication_state.adjudication_status",
                self.adjudication_status,
                allowed=CANONICAL_ADJUDICATION_STATUSES,
                aliases=CANONICAL_ADJUDICATION_STATUS_ALIASES,
            ),
        )
        object.__setattr__(self, "decisive_vs_refs", tuple(self.decisive_vs_refs))
        if self.blocking_reason is not None:
            object.__setattr__(self, "blocking_reason", str(self.blocking_reason))


@dataclass(frozen=True)
class ControlState:
    selected_action: ControlAction
    budget_state: BudgetState
    runtime_status: str
    uncertainty_state: str
    escalation_state: str
    adjudication_state: AdjudicationState | None = None
    action_candidates: Tuple[ControlAction, ...] = ()
    recovery_target: str | None = None
    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selected_action, ControlAction):
            raise StateIRValidationError("CS.selected_action must be a ControlAction object.")
        if not isinstance(self.budget_state, BudgetState):
            raise StateIRValidationError("CS.budget_state must be a BudgetState object.")
        object.__setattr__(
            self,
            "runtime_status",
            _normalize_choice(
                "CS.runtime_status",
                self.runtime_status,
                allowed=CANONICAL_RUNTIME_STATUSES,
                aliases=CANONICAL_RUNTIME_STATUS_ALIASES,
            ),
        )
        object.__setattr__(
            self,
            "uncertainty_state",
            _require_text("CS.uncertainty_state", self.uncertainty_state),
        )
        object.__setattr__(
            self,
            "escalation_state",
            _require_text("CS.escalation_state", self.escalation_state),
        )
        if self.adjudication_state is not None and not isinstance(self.adjudication_state, AdjudicationState):
            raise StateIRValidationError("CS.adjudication_state must be an AdjudicationState object.")
        object.__setattr__(self, "action_candidates", tuple(self.action_candidates))
        if self.recovery_target is not None:
            object.__setattr__(self, "recovery_target", str(self.recovery_target))
        vector = self.vector if self.vector is not None else np.zeros((1,), dtype=np.float32)
        object.__setattr__(self, "vector", _normalize_vector("CS", "control", vector))
        self._validate_status_pair()

    def _validate_status_pair(self) -> None:
        if self.adjudication_state is None:
            return
        runtime_status = str(self.runtime_status)
        adjudication_status = str(self.adjudication_state.adjudication_status)
        if runtime_status == "candidate_ready" and adjudication_status not in {"pending", "ready"}:
            raise StateIRValidationError(
                "CS.runtime_status=candidate_ready may pair only with adjudication_status pending or ready."
            )
        if runtime_status in {"accepted", "rejected", "abstained"} and runtime_status != adjudication_status:
            raise StateIRValidationError(
                "Terminal runtime_status must match adjudication_state.adjudication_status."
            )
        if runtime_status == "budget_exhausted" and adjudication_status != "blocked":
            raise StateIRValidationError(
                "CS.runtime_status=budget_exhausted must pair with adjudication_status=blocked."
            )


_SECTION_ENTRY_UNION: TypeAlias = Union[
    ProblemFrame,
    SymbolEntry,
    ConstraintRelation,
    Branch,
    Subgoal,
    Obligation,
    Hypothesis,
    StrategyCandidate,
    LemmaBinding,
    VerifierEvidence,
    ConsistencySummary,
    ControlState,
]


def _vector_of(entry: _SECTION_ENTRY_UNION) -> np.ndarray:
    return np.asarray(getattr(entry, "vector"), dtype=np.float32)


def _replace_vector(entry: _SECTION_ENTRY_UNION, vector: np.ndarray) -> _SECTION_ENTRY_UNION:
    return replace(entry, vector=np.asarray(vector, dtype=np.float32))


def _entry_id(slot_name: str, entry: _SECTION_ENTRY_UNION) -> str:
    field_name_map = {
        "PF": "frame",
        "SY": "sy_id",
        "CG": "cg_id",
        "FR": (
            "branch_id",
            "subgoal_id",
            "obligation_id",
            "hypothesis_id",
            "strategy_id",
        ),
        "LM": "lm_id",
        "VS": "vs_id",
        "CS": "control",
    }
    selector = field_name_map[slot_name]
    if isinstance(selector, tuple):
        for field_name in selector:
            if hasattr(entry, field_name):
                return str(getattr(entry, field_name))
        return entry.__class__.__name__.lower()
    if selector in {"frame", "control"}:
        return selector
    return str(getattr(entry, selector))


def _coerce_section_payload(slot_name: str, payload: Any) -> Tuple[_SECTION_ENTRY_UNION, ...]:
    if slot_name in {"PF", "CS"}:
        return (payload,)
    if payload is None:
        return ()
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise StateIRValidationError(f"{slot_name} payload must be a sequence of entries.")
    return tuple(payload)


@dataclass(frozen=True)
class StateIR:
    PF: ProblemFrame
    SY: Tuple[SymbolEntry, ...] = ()
    CG: Tuple[ConstraintRelation, ...] = ()
    FR: Tuple[FrontierEntry, ...] = ()
    LM: Tuple[LemmaBinding, ...] = ()
    VS: Tuple[VerifierEntry, ...] = ()
    CS: ControlState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.PF, ProblemFrame):
            raise StateIRValidationError("PF must be a ProblemFrame object.")
        object.__setattr__(self, "SY", tuple(self.SY))
        object.__setattr__(self, "CG", tuple(self.CG))
        object.__setattr__(self, "FR", tuple(self.FR))
        object.__setattr__(self, "LM", tuple(self.LM))
        object.__setattr__(self, "VS", tuple(self.VS))
        if self.CS is None:
            raise StateIRValidationError("CS must be a ControlState object.")
        if not isinstance(self.CS, ControlState):
            raise StateIRValidationError("CS must be a ControlState object.")

        hidden_dim = self.PF.vector.shape[0]
        for slot_name, entries in self.to_ordered_sections():
            for entry in entries:
                vector = _vector_of(entry)
                if vector.shape[0] != hidden_dim:
                    raise StateIRValidationError(
                        f"{slot_name}:{_entry_id(slot_name, entry)} hidden dim {vector.shape[0]} "
                        f"does not match PF hidden dim {hidden_dim}."
                    )

    @property
    def hidden_dim(self) -> int:
        return int(self.PF.vector.shape[0])

    @property
    def total_tokens(self) -> int:
        return int(sum(self.section_lengths().values()))

    def section_lengths(self) -> Dict[str, int]:
        return {
            "PF": 1,
            "SY": int(len(self.SY)),
            "CG": int(len(self.CG)),
            "FR": int(len(self.FR)),
            "LM": int(len(self.LM)),
            "VS": int(len(self.VS)),
            "CS": 1,
        }

    def slot_lengths(self) -> Dict[str, int]:
        return self.section_lengths()

    def to_ordered_sections(self) -> Tuple[Tuple[str, Tuple[_SECTION_ENTRY_UNION, ...]], ...]:
        return (
            ("PF", (self.PF,)),
            ("SY", tuple(self.SY)),
            ("CG", tuple(self.CG)),
            ("FR", tuple(self.FR)),
            ("LM", tuple(self.LM)),
            ("VS", tuple(self.VS)),
            ("CS", (self.CS,)),
        )

    def to_ordered_slots(self) -> Tuple[Tuple[str, Tuple[_SECTION_ENTRY_UNION, ...]], ...]:
        return self.to_ordered_sections()

    def to_slot_map(self) -> Dict[str, Any]:
        return {
            "PF": self.PF,
            "SY": tuple(self.SY),
            "CG": tuple(self.CG),
            "FR": tuple(self.FR),
            "LM": tuple(self.LM),
            "VS": tuple(self.VS),
            "CS": self.CS,
        }

    def to_token_map(self) -> Dict[str, Any]:
        return self.to_slot_map()

    def to_canonical_sequence(self) -> np.ndarray:
        rows = [_vector_of(entry) for _, entries in self.to_ordered_sections() for entry in entries]
        return np.stack(rows, axis=0).astype(np.float32)

    def with_updated_sequence(self, updated_sequence: np.ndarray) -> "StateIR":
        sequence = np.asarray(updated_sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise StateIRValidationError("Updated sequence must be rank-2.")
        if sequence.shape[1] != self.hidden_dim:
            raise StateIRValidationError(
                f"Updated sequence hidden dim {sequence.shape[1]} does not match {self.hidden_dim}."
            )

        expected_rows = int(sum(self.section_lengths().values()))
        if sequence.shape[0] != expected_rows:
            raise StateIRValidationError(
                f"Updated sequence token count {sequence.shape[0]} does not match {expected_rows}."
            )

        updated_sections: Dict[str, Tuple[_SECTION_ENTRY_UNION, ...]] = {}
        offset = 0
        for slot_name, entries in self.to_ordered_sections():
            slot_entries = []
            for entry in entries:
                slot_entries.append(_replace_vector(entry, sequence[offset]))
                offset += 1
            updated_sections[slot_name] = tuple(slot_entries)

        return StateIR(
            PF=updated_sections["PF"][0],
            SY=updated_sections["SY"],
            CG=updated_sections["CG"],
            FR=updated_sections["FR"],
            LM=updated_sections["LM"],
            VS=updated_sections["VS"],
            CS=updated_sections["CS"][0],
        )

    @classmethod
    def empty(cls, hidden_dim: int) -> "StateIR":
        zero = np.zeros((hidden_dim,), dtype=np.float32)
        return cls(
            PF=ProblemFrame(
                task_type="draft",
                target_spec="unresolved_target",
                required_output=RequiredOutput(
                    output_kind="proof",
                    answer_channel="structured_object",
                    formality_level="informal",
                ),
                problem_assumptions=(),
                domain_tags=(),
                source_anchor_refs=(),
                frame_status="draft",
                vector=zero,
            ),
            CS=ControlState(
                selected_action=ControlAction(
                    action_id="action-0",
                    action_type="continue",
                    action_status="selected",
                ),
                budget_state=BudgetState(global_step_budget_remaining=0),
                runtime_status="in_progress",
                uncertainty_state="unknown",
                escalation_state="inactive",
                vector=zero,
            ),
        )

    @classmethod
    def from_ordered_sections(cls, ordered_sections: Sequence[Tuple[str, Any]]) -> "StateIR":
        if len(ordered_sections) != len(STATE_IR_SLOT_ORDER):
            raise StateIRValidationError("State IR must provide all canonical slots once.")

        observed_order = tuple(slot_name for slot_name, _ in ordered_sections)
        if observed_order != STATE_IR_SLOT_ORDER:
            raise StateIRValidationError(
                f"Canonical State IR order is {STATE_IR_SLOT_ORDER}, got {observed_order}."
            )

        slot_map: Dict[str, Any] = {}
        for slot_name, payload in ordered_sections:
            entries = _coerce_section_payload(slot_name, payload)
            if slot_name == "PF":
                if len(entries) != 1 or not isinstance(entries[0], ProblemFrame):
                    raise StateIRValidationError("PF must contain exactly one ProblemFrame.")
                slot_map["PF"] = entries[0]
            elif slot_name == "CS":
                if len(entries) != 1 or not isinstance(entries[0], ControlState):
                    raise StateIRValidationError("CS must contain exactly one ControlState.")
                slot_map["CS"] = entries[0]
            else:
                slot_map[slot_name] = entries

        return cls(
            PF=slot_map["PF"],
            SY=tuple(slot_map["SY"]),
            CG=tuple(slot_map["CG"]),
            FR=tuple(slot_map["FR"]),
            LM=tuple(slot_map["LM"]),
            VS=tuple(slot_map["VS"]),
            CS=slot_map["CS"],
        )

    @classmethod
    def from_slot_map(cls, slot_map: Mapping[str, Any]) -> "StateIR":
        provided = set(slot_map.keys())
        expected = set(STATE_IR_SLOT_ORDER)
        unknown = sorted(provided - expected)
        missing = sorted(expected - provided)
        if unknown:
            raise StateIRValidationError(f"Unknown State IR slots are forbidden: {unknown}.")
        if missing:
            raise StateIRValidationError(f"Missing required State IR slots: {missing}.")

        ordered_sections = [(slot_name, slot_map[slot_name]) for slot_name in STATE_IR_SLOT_ORDER]
        return cls.from_ordered_sections(ordered_sections)

    @classmethod
    def from_token_map(cls, token_map: Mapping[str, Any]) -> "StateIR":
        return cls.from_slot_map(token_map)
