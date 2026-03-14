from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..levels import LevelInput, build_level_stack
from ..schema import AdjudicationState, BudgetState, ControlAction, ControlState, ProblemFrame, RequiredOutput, StateIR
from ..train import build_document_pipeline_bundle
from .math_native_fixtures import DocumentEvalFixture, ProofEvalFixture
from .phase_c_gate import GateContext, _safe_float

LEVEL_IDS: Tuple[str, ...] = tuple(f"L{index}" for index in range(7))
SIDECAR_TECH_DEBT_NOTE = (
    "TEMPORARY TECHNICAL DEBT: local image/scanned_note/diagram fixtures use checked-in sidecar "
    "normalization until mounted parser/verifier backends replace fixture-only replay."
)
REPLACEMENT_TARGET = "mounted_multimodal_parser_and_verifier_backends"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"data": payload}


def hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(json_safe(dict(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_document_fixture_root() -> Path:
    return repo_root() / "tests" / "fixtures" / "p1_phase_de" / "document_eval"


def default_proof_fixture_root() -> Path:
    return repo_root() / "tests" / "fixtures" / "p1_phase_de" / "proof_eval"


def phase_root_paths(phase_root: Path) -> Dict[str, Path]:
    phase_root = Path(phase_root)
    with_prefix = {
        "uninterrupted": phase_root / "s8_uninterrupted",
        "execute_crash": phase_root / "s8_execute",
        "pre_commit_crash": phase_root / "s8_pre_commit",
        "post_commit_crash": phase_root / "s8_post_commit",
    }
    if all(path.exists() for path in with_prefix.values()):
        return with_prefix
    return {
        "uninterrupted": phase_root / "uninterrupted",
        "execute_crash": phase_root / "execute_crash",
        "pre_commit_crash": phase_root / "pre_commit_crash",
        "post_commit_crash": phase_root / "post_commit_crash",
    }


def required_output_from_payload(payload: Mapping[str, Any] | None) -> RequiredOutput:
    payload = dict(payload or {})
    return RequiredOutput(
        output_kind=str(payload.get("output_kind", "proof")),
        answer_channel=str(payload.get("answer_channel", "structured_object")),
        formality_level=str(payload.get("formality_level", "informal")),
        verifier_mode=str(payload.get("verifier_mode", "proof_gap_plus_local_validity")),
    )


def initial_state(
    *,
    hidden_dim: int,
    task_type: str,
    target_spec: str,
    required_output: Mapping[str, Any] | None,
) -> StateIR:
    zero = np.zeros((hidden_dim,), dtype=np.float32)
    return StateIR(
        PF=ProblemFrame(
            task_type=str(task_type),
            target_spec=str(target_spec),
            required_output=required_output_from_payload(required_output),
            problem_assumptions=(),
            domain_tags=("document_grounded", "local_eval"),
            source_anchor_refs=(),
            frame_status="draft",
            vector=zero,
        ),
        CS=ControlState(
            selected_action=ControlAction(action_id="action-0", action_type="continue"),
            budget_state=BudgetState(
                global_step_budget_remaining=4,
                branch_expansion_budget_remaining=2,
                verifier_probe_budget_remaining=2,
                reparse_budget_remaining=1,
            ),
            runtime_status="in_progress",
            uncertainty_state="bounded",
            escalation_state="inactive",
            adjudication_state=AdjudicationState(
                task_adjudication_policy_id="task-family-proof-natural-language-default-v1",
                adjudication_status="pending",
            ),
            vector=zero,
        ),
    )


def run_mounted_stack(
    *,
    state: StateIR,
    context_in: Mapping[str, Any],
    hidden_dim: int,
    seed: int,
) -> tuple[StateIR, Dict[str, Dict[str, Any]]]:
    stack = build_level_stack(implementation="mounted", hidden_dim=hidden_dim, seed=seed)
    current_state = state
    diagnostics_by_level: Dict[str, Dict[str, Any]] = {}
    for level_id in LEVEL_IDS:
        output = stack[level_id].run(LevelInput(state_in=current_state, context_in=dict(context_in)))
        current_state = output.state_out
        diagnostics_by_level[level_id] = dict(output.diagnostics)
    return current_state, diagnostics_by_level


def benchmark_policy(bundle: Any, benchmark_family_policy_ref: str | None) -> Any | None:
    if not benchmark_family_policy_ref:
        return None
    return dict(bundle.benchmark_family_policies).get(str(benchmark_family_policy_ref))


def evaluate_document_fixture(
    fixture: DocumentEvalFixture,
    *,
    policy_bundle: Any,
    hidden_dim: int,
    seed: int,
) -> tuple[Dict[str, Any], Any]:
    document_bundle = build_document_pipeline_bundle(
        fixture.resolved_document_path(),
        sidecar_path=fixture.resolved_sidecar_path(),
        source_format_override=fixture.source_format_override,
    )
    current_state, diagnostics = run_mounted_stack(
        state=initial_state(
            hidden_dim=hidden_dim,
            task_type="prove",
            target_spec=f"document fixture {fixture.fixture_id}",
            required_output={"output_kind": "proof", "answer_channel": "structured_object"},
        ),
        context_in={
            "document_bundle": document_bundle,
            "benchmark_family_policy": benchmark_policy(policy_bundle, fixture.benchmark_family_policy_ref),
            "task_family_override": fixture.task_family,
        },
        hidden_dim=hidden_dim,
        seed=seed,
    )
    l0 = diagnostics["L0"]
    l6 = diagnostics["L6"]
    current_slot_targets = tuple(str(item) for item in document_bundle.projection.state_ir_slot_targets)
    parse_completeness = float(l0.get("rep.document.parse_completeness", 0.0))
    grounding_score = float(l6.get("task.document_grounding_score", l0.get("task.document_grounding_score", 0.0)))
    parser_coverage = float(l0.get("provenance.parser_coverage", 0.0))
    anchor_count = len(tuple(document_bundle.projection.anchor_refs))
    runtime_status = str(l6.get("runtime_status", current_state.CS.runtime_status))
    adjudication_state = current_state.CS.adjudication_state
    adjudication_status = str(
        l6.get(
            "adjudication_status",
            adjudication_state.adjudication_status if adjudication_state is not None else "pending",
        )
    )
    reasons: List[str] = []
    if current_slot_targets != tuple(fixture.expected_slot_targets):
        reasons.append(
            "state_ir_slot_targets mismatch: "
            + f"expected {tuple(fixture.expected_slot_targets)!r}, got {current_slot_targets!r}"
        )
    if anchor_count < int(fixture.expected_min_anchor_count):
        reasons.append(f"anchor_count {anchor_count} < expected minimum {fixture.expected_min_anchor_count}")
    if parse_completeness < float(fixture.expected_min_parse_completeness):
        reasons.append(
            "rep.document.parse_completeness "
            + f"{parse_completeness:.6g} < expected minimum {fixture.expected_min_parse_completeness:.6g}"
        )
    if grounding_score < float(fixture.expected_min_document_grounding_score):
        reasons.append(
            "task.document_grounding_score "
            + f"{grounding_score:.6g} < expected minimum {fixture.expected_min_document_grounding_score:.6g}"
        )
    if runtime_status != fixture.expected_runtime_status:
        reasons.append(f"runtime_status mismatch: expected {fixture.expected_runtime_status!r}, got {runtime_status!r}")
    if adjudication_status != fixture.expected_adjudication_status:
        reasons.append(
            "adjudication_status mismatch: "
            + f"expected {fixture.expected_adjudication_status!r}, got {adjudication_status!r}"
        )
    result = {
        "concept_id": fixture.fixture_id,
        "fixture_id": fixture.fixture_id,
        "item_kind": "document",
        "eval_partition": fixture.eval_partition,
        "source_format": document_bundle.source.source_format,
        "source_path": str(fixture.resolved_document_path()),
        "sidecar_path": str(fixture.resolved_sidecar_path()) if fixture.resolved_sidecar_path() else None,
        "pair_group_id": fixture.paired_group_id,
        "pair_variant_id": fixture.pair_variant_id,
        "projection_kind": document_bundle.projection.projection_kind,
        "expected_slot_targets": list(fixture.expected_slot_targets),
        "current_slot_targets": list(current_slot_targets),
        "anchor_count": anchor_count,
        "parser_provenance_id": document_bundle.record.parser_provenance_id,
        "parser_provenance_refs": dict(document_bundle.record.parser_provenance_refs),
        "provenance.parser_coverage": parser_coverage,
        "rep.document.parse_completeness": parse_completeness,
        "task.document_grounding_score": grounding_score,
        "proof.evidence_coverage": 1.0 if len(tuple(current_state.VS)) > 0 else 0.0,
        "concept.isolation_score": grounding_score,
        "concept.leakage_score": max(0.0, 1.0 - grounding_score),
        "runtime_status": runtime_status,
        "adjudication_status": adjudication_status,
        "task_family": str(l6.get("task_family", "")),
        "task_family_resolution_source": str(l6.get("task_family_resolution_source", "")),
        "task_adjudication_policy_id": str(l6.get("task_adjudication_policy_id", "")),
        "task_adjudication_policy_resolution_source": str(
            l6.get("task_adjudication_policy_resolution_source", "")
        ),
        "benchmark_family_override_ref": (
            str(l6.get("benchmark_family_override_ref", "")).strip() or None
        ),
        "status": "PASS" if not reasons else "FAIL",
        "failure_reasons": reasons,
        "technical_debt": (
            {
                "label": "TEMPORARY TECHNICAL DEBT",
                "note": SIDECAR_TECH_DEBT_NOTE,
                "removal_criterion": "Replace sidecar-backed local fixture replay with mounted parser/verifier backends.",
                "intended_learned_replacement": REPLACEMENT_TARGET,
            }
            if document_bundle.source.source_format in {"IMAGE", "SCANNED_NOTE", "DIAGRAM", "PDF"}
            else None
        ),
    }
    return result, document_bundle


def proof_emitted_evidence_classes(state: StateIR, l6_diagnostics: Mapping[str, Any]) -> Tuple[str, ...]:
    emitted = {
        str(getattr(entry, "evidence_class", "")).strip()
        for entry in state.VS
        if str(getattr(entry, "evidence_class", "")).strip()
    }
    verifier_head = dict(dict(l6_diagnostics.get("internal_heads", {})).get("verifier_aggregator", {}))
    formal_bridge_status = str(verifier_head.get("formal_bridge_status", "")).strip()
    if formal_bridge_status and formal_bridge_status != "disabled":
        emitted.add("formal_bridge")
    return tuple(sorted(emitted))


def proof_evidence_coverage(required: Sequence[str], emitted: Sequence[str]) -> float:
    required_set = {str(item).strip() for item in required if str(item).strip()}
    if not required_set:
        return 1.0
    emitted_set = {str(item).strip() for item in emitted if str(item).strip()}
    return float(len(required_set & emitted_set)) / float(len(required_set))


def _mean_truthy_fraction(items: Sequence[Mapping[str, Any]], field_name: str) -> float:
    if not items:
        return 0.0
    truthy = 0
    for item in items:
        value = item.get(field_name)
        if isinstance(value, (int, float)):
            if float(value) > 0.0:
                truthy += 1
            continue
        if str(value or "").strip():
            truthy += 1
    return float(truthy) / float(len(items))


def _value_counts(items: Sequence[Mapping[str, Any]], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item.get(field_name, "")).strip() or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_truthy(items: Sequence[Mapping[str, Any]], field_name: str) -> int:
    return sum(1 for item in items if item.get(field_name))


def evaluate_proof_fixture(
    fixture: ProofEvalFixture,
    *,
    policy_bundle: Any,
    document_bundles: Mapping[str, Any],
    hidden_dim: int,
    seed: int,
) -> Dict[str, Any]:
    linked_bundle = document_bundles.get(str(fixture.document_fixture_id or ""))
    reasons: List[str] = []
    if fixture.document_fixture_id and linked_bundle is None:
        reasons.append(f"linked document_fixture_id is unresolved: {fixture.document_fixture_id}")
    current_state, diagnostics = run_mounted_stack(
        state=initial_state(
            hidden_dim=hidden_dim,
            task_type=fixture.task_type,
            target_spec=fixture.target_spec,
            required_output=fixture.required_output,
        ),
        context_in={
            "document_bundle": linked_bundle,
            "benchmark_family_policy": benchmark_policy(policy_bundle, fixture.benchmark_family_policy_ref),
            "task_family_override": fixture.task_family,
            "task_adjudication_policy_id": fixture.item_policy_id,
        },
        hidden_dim=hidden_dim,
        seed=seed,
    )
    l6 = diagnostics["L6"]
    emitted = proof_emitted_evidence_classes(current_state, l6)
    evidence_coverage = proof_evidence_coverage(fixture.required_evidence_classes, emitted)
    runtime_status = str(l6.get("runtime_status", current_state.CS.runtime_status))
    adjudication_state = current_state.CS.adjudication_state
    adjudication_status = str(
        l6.get(
            "adjudication_status",
            adjudication_state.adjudication_status if adjudication_state is not None else "pending",
        )
    )
    task_family = str(l6.get("task_family", ""))
    policy_id = str(l6.get("task_adjudication_policy_id", ""))
    validity_score = float(l6.get("task.validity_score", 0.0))
    false_accept_rate = float(l6.get("eval.false_accept_rate", 1.0))
    calibration_error = float(l6.get("eval.calibration_error", 1.0))
    verifier_coverage = float(l6.get("provenance.verifier_coverage", 0.0))
    formalizer_coverage = float(l6.get("provenance.formalizer_coverage", 0.0))
    verifier_head = dict(dict(l6.get("internal_heads", {})).get("verifier_aggregator", {}))
    formal_bridge_status = str(verifier_head.get("formal_bridge_status", "")).strip() or None
    verifier_manifest = dict(getattr(policy_bundle, "provenance_manifests", {})).get("verifier-stack-v1")
    formalizer_manifest = dict(getattr(policy_bundle, "provenance_manifests", {})).get("formalizer-skeleton-v1")
    if task_family != fixture.task_family:
        reasons.append(f"task_family mismatch: expected {fixture.task_family!r}, got {task_family!r}")
    if policy_id != fixture.expected_task_adjudication_policy_id:
        reasons.append(
            "task_adjudication_policy_id mismatch: "
            + f"expected {fixture.expected_task_adjudication_policy_id!r}, got {policy_id!r}"
        )
    if runtime_status != fixture.expected_runtime_status:
        reasons.append(f"runtime_status mismatch: expected {fixture.expected_runtime_status!r}, got {runtime_status!r}")
    if adjudication_status != fixture.expected_adjudication_status:
        reasons.append(
            "adjudication_status mismatch: "
            + f"expected {fixture.expected_adjudication_status!r}, got {adjudication_status!r}"
        )
    if evidence_coverage < float(fixture.min_verifier_coverage):
        reasons.append(
            f"proof.evidence_coverage {evidence_coverage:.6g} < expected minimum {fixture.min_verifier_coverage:.6g}"
        )
    if validity_score < float(fixture.min_validity_score):
        reasons.append(
            f"task.validity_score {validity_score:.6g} < expected minimum {fixture.min_validity_score:.6g}"
        )
    if false_accept_rate > float(fixture.max_false_accept_rate):
        reasons.append(
            f"eval.false_accept_rate {false_accept_rate:.6g} > allowed maximum {fixture.max_false_accept_rate:.6g}"
        )
    if calibration_error > float(fixture.max_calibration_error):
        reasons.append(
            f"eval.calibration_error {calibration_error:.6g} > allowed maximum {fixture.max_calibration_error:.6g}"
        )
    if verifier_coverage < float(fixture.min_verifier_coverage):
        reasons.append(
            f"provenance.verifier_coverage {verifier_coverage:.6g} < expected minimum {fixture.min_verifier_coverage:.6g}"
        )
    if fixture.expected_formal_bridge_status and formal_bridge_status != fixture.expected_formal_bridge_status:
        reasons.append(
            "formal_bridge_status mismatch: "
            + f"expected {fixture.expected_formal_bridge_status!r}, got {formal_bridge_status!r}"
        )
    return {
        "concept_id": fixture.fixture_id,
        "fixture_id": fixture.fixture_id,
        "item_kind": "proof",
        "eval_partition": fixture.eval_partition,
        "document_fixture_id": fixture.document_fixture_id,
        "proof_seed": fixture.proof_seed,
        "task_type": fixture.task_type,
        "target_spec": fixture.target_spec,
        "task_family": task_family,
        "task_adjudication_policy_id": policy_id,
        "required_evidence_classes": list(fixture.required_evidence_classes),
        "emitted_evidence_classes": list(emitted),
        "proof.evidence_coverage": evidence_coverage,
        "verifier_provenance_id": (
            str(getattr(verifier_manifest, "manifest_id", "")).strip() or None
        ),
        "formalizer_provenance_id": (
            str(getattr(formalizer_manifest, "manifest_id", "")).strip() or None
        ),
        "rep.document.parse_completeness": float(diagnostics["L0"].get("rep.document.parse_completeness", 0.0)),
        "task.document_grounding_score": float(l6.get("task.document_grounding_score", 0.0)),
        "task.validity_score": validity_score,
        "eval.false_accept_rate": false_accept_rate,
        "eval.calibration_error": calibration_error,
        "provenance.verifier_coverage": verifier_coverage,
        "provenance.formalizer_coverage": formalizer_coverage,
        "concept.isolation_score": evidence_coverage,
        "concept.leakage_score": max(0.0, 1.0 - evidence_coverage),
        "runtime_status": runtime_status,
        "adjudication_status": adjudication_status,
        "task_family_resolution_source": str(l6.get("task_family_resolution_source", "")),
        "task_adjudication_policy_resolution_source": str(
            l6.get("task_adjudication_policy_resolution_source", "")
        ),
        "benchmark_family_override_ref": (
            str(l6.get("benchmark_family_override_ref", "")).strip() or None
        ),
        "formal_bridge_status": formal_bridge_status,
        "status": "PASS" if not reasons else "FAIL",
        "failure_reasons": reasons,
    }


def mean_metric(items: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    if not items:
        return 0.0
    return float(sum(_safe_float(item.get(metric_name), 0.0) for item in items) / float(len(items)))


def document_eval_packet(
    *,
    document_results: Sequence[Mapping[str, Any]],
    context: GateContext,
    eval_partition: str,
) -> Dict[str, Any]:
    payload = {
        "schema": "iris.regression.document_eval_packet/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "eval_partition": eval_partition,
        "fixture_count": len(document_results),
        "status": "PASS" if document_results else "FAIL",
        "aggregate": {
            "rep.document.parse_completeness.mean": mean_metric(document_results, "rep.document.parse_completeness"),
            "task.document_grounding_score.mean": mean_metric(document_results, "task.document_grounding_score"),
            "provenance.parser_coverage.mean": mean_metric(document_results, "provenance.parser_coverage"),
        },
        "coverage_summary": {
            "task_family_resolution_coverage": _mean_truthy_fraction(
                document_results, "task_family_resolution_source"
            ),
            "task_adjudication_policy_resolution_coverage": _mean_truthy_fraction(
                document_results, "task_adjudication_policy_resolution_source"
            ),
            "runtime_status_counts": _value_counts(document_results, "runtime_status"),
            "adjudication_status_counts": _value_counts(document_results, "adjudication_status"),
            "task_family_resolution_source_counts": _value_counts(
                document_results, "task_family_resolution_source"
            ),
            "task_adjudication_policy_resolution_source_counts": _value_counts(
                document_results, "task_adjudication_policy_resolution_source"
            ),
            "sidecar_fixture_count": _count_truthy(document_results, "sidecar_path"),
            "technical_debt_fixture_count": _count_truthy(document_results, "technical_debt"),
        },
        "fixtures": list(document_results),
    }
    payload["artifact_hash"] = hash_payload(payload)
    return payload


def proof_eval_packet(
    *,
    proof_results: Sequence[Mapping[str, Any]],
    context: GateContext,
    eval_partition: str,
) -> Dict[str, Any]:
    payload = {
        "schema": "iris.regression.proof_eval_packet/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "eval_partition": eval_partition,
        "fixture_count": len(proof_results),
        "status": "PASS" if proof_results else "FAIL",
        "aggregate": {
            "proof.evidence_coverage.mean": mean_metric(proof_results, "proof.evidence_coverage"),
            "task.validity_score.mean": mean_metric(proof_results, "task.validity_score"),
            "eval.false_accept_rate.mean": mean_metric(proof_results, "eval.false_accept_rate"),
            "eval.calibration_error.mean": mean_metric(proof_results, "eval.calibration_error"),
            "provenance.verifier_coverage.mean": mean_metric(
                proof_results, "provenance.verifier_coverage"
            ),
            "provenance.formalizer_coverage.mean": mean_metric(
                proof_results, "provenance.formalizer_coverage"
            ),
        },
        "coverage_summary": {
            "task_family_resolution_coverage": _mean_truthy_fraction(
                proof_results, "task_family_resolution_source"
            ),
            "task_adjudication_policy_resolution_coverage": _mean_truthy_fraction(
                proof_results, "task_adjudication_policy_resolution_source"
            ),
            "runtime_status_counts": _value_counts(proof_results, "runtime_status"),
            "adjudication_status_counts": _value_counts(proof_results, "adjudication_status"),
            "task_family_resolution_source_counts": _value_counts(
                proof_results, "task_family_resolution_source"
            ),
            "task_adjudication_policy_resolution_source_counts": _value_counts(
                proof_results, "task_adjudication_policy_resolution_source"
            ),
            "formal_bridge_status_counts": _value_counts(proof_results, "formal_bridge_status"),
        },
        "fixtures": list(proof_results),
    }
    payload["artifact_hash"] = hash_payload(payload)
    return payload
