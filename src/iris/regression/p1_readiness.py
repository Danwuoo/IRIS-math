from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..train import load_policy_bundle_for_profile_phase, resolve_learning_objective_bundle
from .phase_c_gate import GateContext, _safe_float, utc_now_iso

_P1_REQUIRED_SURFACES: Tuple[str, ...] = (
    "rep.document.parse_completeness",
    "task.document_grounding_score",
    "failure.credit.collapse_rate",
    "eval.false_accept_rate",
    "eval.calibration_error",
    "contam.strict_holdout_leakage_score",
    "provenance.parser_coverage",
    "provenance.verifier_coverage",
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"data": payload}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _metric_from_mapping(payload: Mapping[str, Any], *paths: str) -> float | None:
    for dotted_path in paths:
        if dotted_path in payload:
            try:
                return float(payload[dotted_path])
            except (TypeError, ValueError):
                pass
        current: Any = payload
        remaining = dotted_path.split(".")
        found = True
        while remaining:
            if not isinstance(current, Mapping):
                found = False
                break
            direct_key = ".".join(remaining)
            if direct_key in current:
                current = current[direct_key]
                remaining = []
                break
            part = remaining.pop(0)
            if part not in current:
                found = False
                break
            current = current[part]
        if found and not remaining:
            try:
                return float(current)
            except (TypeError, ValueError):
                continue
    return None


def _collapse_rate(credit_vector: Mapping[str, Any]) -> float:
    if not isinstance(credit_vector, Mapping):
        return 0.0
    values = [_safe_float(value, 0.0) for value in credit_vector.values()]
    return max(values, default=0.0)


def _value_counts(items: Sequence[Mapping[str, Any]], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item.get(field_name, "")).strip() or "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _truthy_count(items: Sequence[Mapping[str, Any]], field_name: str) -> int:
    return sum(1 for item in items if item.get(field_name))


def _resolve_checkpoint_path(checkpoint_ref: Any, model_run_dir: Path) -> Path | None:
    if not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip():
        return None
    candidate = Path(checkpoint_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    candidate = model_run_dir / checkpoint_ref
    if candidate.exists():
        return candidate.resolve()
    return None


def inspect_model_run(model_run_dir: Path | None) -> Dict[str, Any]:
    if model_run_dir is None:
        return {
            "status": "FAIL",
            "governed_run": False,
            "reason": "model_run_dir was not provided.",
        }
    model_root = Path(model_run_dir)
    if not model_root.exists():
        return {
            "status": "FAIL",
            "governed_run": False,
            "reason": f"model_run_dir does not exist: {model_root}",
        }
    journal_path = model_root / "segment_journal.jsonl"
    events = _read_jsonl(journal_path)
    applied_events = [event for event in events if str(event.get("status", "")).upper() == "APPLIED"]
    if not applied_events:
        return {
            "status": "FAIL",
            "governed_run": False,
            "reason": f"No APPLIED segment was found under {journal_path}.",
        }
    last_applied = applied_events[-1]
    checkpoint_path = _resolve_checkpoint_path(last_applied.get("checkpoint_ref"), model_root)
    if checkpoint_path is None:
        return {
            "status": "FAIL",
            "governed_run": False,
            "reason": "APPLIED segment does not resolve a checkpoint payload.",
            "checkpoint_ref": last_applied.get("checkpoint_ref"),
        }
    checkpoint_payload = _read_json(checkpoint_path)
    schema = str(checkpoint_payload.get("schema", "")).strip()
    governed = schema == "iris.training_checkpoint/v2"
    return {
        "status": "PASS" if governed else "FAIL",
        "governed_run": governed,
        "checkpoint_schema": schema or "missing",
        "checkpoint_path": str(checkpoint_path),
        "journal_path": str(journal_path),
        "reason": (
            ""
            if governed
            else "Checkpoint schema is not iris.training_checkpoint/v2; readiness evidence remains synthetic or partial."
        ),
    }


def _load_leakage_audit(leakage_audit_path: Path | None) -> Dict[str, Any]:
    if leakage_audit_path is None:
        return {
            "available": False,
            "reproducible": False,
            "source_path": None,
            "reason": "No strict held-out leakage audit artifact was provided.",
            "value": None,
            "uncertainty_band_pp": None,
        }
    path = Path(leakage_audit_path)
    if not path.exists():
        return {
            "available": False,
            "reproducible": False,
            "source_path": str(path),
            "reason": "Leakage audit artifact path does not exist.",
            "value": None,
            "uncertainty_band_pp": None,
        }
    payload = _read_json(path)
    value = _metric_from_mapping(
        payload,
        "contam.strict_holdout_leakage_score",
        "aggregate.contam.strict_holdout_leakage_score",
        "strict_holdout_leakage_score",
    )
    uncertainty_band_pp = _metric_from_mapping(
        payload,
        "audit_uncertainty_band_pp",
        "aggregate.audit_uncertainty_band_pp",
        "uncertainty_band_pp",
    )
    return {
        "available": value is not None,
        "reproducible": value is not None,
        "source_path": str(path),
        "reason": "" if value is not None else "Artifact does not expose contam.strict_holdout_leakage_score.",
        "schema": payload.get("schema"),
        "value": value,
        "uncertainty_band_pp": uncertainty_band_pp,
    }


def _load_phase_packet(root: Path, name: str) -> Dict[str, Any]:
    packet_path = Path(root) / name
    if not packet_path.exists():
        return {"status": "FAIL", "reason": f"Missing required artifact: {packet_path}", "_path": str(packet_path)}
    payload = _read_json(packet_path)
    payload["_path"] = str(packet_path)
    return payload


def _phase_surface_values(phase_e_root: Path) -> Dict[str, Any]:
    phase_e_root = Path(phase_e_root)
    heldout_packet = _load_phase_packet(phase_e_root, "phase_e_heldout_packet.json")
    heldout_document_packet = dict(heldout_packet.get("document_packet", {}))
    heldout_proof_packet = dict(heldout_packet.get("proof_packet", {}))
    resume_packet = _load_phase_packet(phase_e_root, "resume_consistency_packet.json")
    uninterrupted = {}
    for path_entry in resume_packet.get("paths", []) if isinstance(resume_packet.get("paths"), list) else []:
        if str(path_entry.get("resume_path_id", "")) == "uninterrupted":
            uninterrupted = dict(path_entry)
            break
    return {
        "heldout_packet": heldout_packet,
        "heldout_document_packet": heldout_document_packet,
        "heldout_proof_packet": heldout_proof_packet,
        "resume_packet": resume_packet,
        "uninterrupted_path": uninterrupted,
    }


def build_p1_readiness_snapshot(
    *,
    phase_d_root: Path,
    phase_e_root: Path,
    model_run_dir: Path | None,
    leakage_audit_path: Path | None,
    baseline_id: str,
    tolerance_profile_id: str,
) -> Dict[str, Any]:
    phase_d_root = Path(phase_d_root)
    phase_e_root = Path(phase_e_root)
    phase_d_summary = _load_phase_packet(phase_d_root, "summary_report.json")
    phase_e_summary = _load_phase_packet(phase_e_root, "summary_report.json")
    phase_d_provenance = _load_phase_packet(phase_d_root, "provenance_contamination_summary.json")
    phase_data = _phase_surface_values(phase_e_root)
    document_packet = dict(phase_data.get("heldout_document_packet", {}))
    proof_packet = dict(phase_data.get("heldout_proof_packet", {}))
    heldout_packet = dict(phase_data.get("heldout_packet", {}))
    uninterrupted_path = dict(phase_data.get("uninterrupted_path", {}))
    leakage_audit = _load_leakage_audit(leakage_audit_path)
    model_run_summary = inspect_model_run(model_run_dir)
    policy_bundle = load_policy_bundle_for_profile_phase("P1", "E")
    learning_bundle, learning_bundle_source = resolve_learning_objective_bundle(profile_id="P1", phase="E")

    document_fixtures = list(document_packet.get("fixtures", [])) if isinstance(document_packet.get("fixtures"), list) else []
    proof_fixtures = list(proof_packet.get("fixtures", [])) if isinstance(proof_packet.get("fixtures"), list) else []
    document_coverage = dict(document_packet.get("coverage_summary", {}))
    proof_coverage = dict(proof_packet.get("coverage_summary", {}))

    surface_values = {
        "rep.document.parse_completeness": {
            "value": _metric_from_mapping(
                document_packet, "aggregate.rep.document.parse_completeness.mean"
            ),
            "source_artifact": str(heldout_packet.get("_path", "")),
        },
        "task.document_grounding_score": {
            "value": _metric_from_mapping(
                document_packet, "aggregate.task.document_grounding_score.mean"
            ),
            "source_artifact": str(heldout_packet.get("_path", "")),
        },
        "failure.credit.collapse_rate": {
            "value": _collapse_rate(dict(uninterrupted_path.get("credit_vector", {}))),
            "source_artifact": str(phase_data.get("resume_packet", {}).get("_path", "")),
        },
        "eval.false_accept_rate": {
            "value": _metric_from_mapping(
                proof_packet, "aggregate.eval.false_accept_rate.mean"
            ),
            "source_artifact": str(heldout_packet.get("_path", "")),
        },
        "eval.calibration_error": {
            "value": _metric_from_mapping(
                proof_packet, "aggregate.eval.calibration_error.mean"
            ),
            "source_artifact": str(heldout_packet.get("_path", "")),
        },
        "contam.strict_holdout_leakage_score": {
            "value": leakage_audit.get("value"),
            "source_artifact": str(leakage_audit.get("source_path") or ""),
        },
        "provenance.parser_coverage": {
            "value": _metric_from_mapping(
                document_packet, "aggregate.provenance.parser_coverage.mean"
            ),
            "source_artifact": str(heldout_packet.get("_path", "")),
        },
        "provenance.verifier_coverage": {
            "value": _metric_from_mapping(
                proof_packet, "aggregate.provenance.verifier_coverage.mean"
            ),
            "source_artifact": str(heldout_packet.get("_path", "")),
        },
    }

    blockers: List[Dict[str, Any]] = []
    if str(phase_d_summary.get("regression.status", "FAIL")).upper() != "PASS":
        blockers.append(
            {
                "blocker": "phase_d_regression",
                "reason": "Phase D regression artifacts are not PASS.",
                "source_artifact": str(phase_d_summary.get("_path", "")),
            }
        )
    if str(phase_e_summary.get("regression.status", "FAIL")).upper() != "PASS":
        blockers.append(
            {
                "blocker": "phase_e_regression",
                "reason": "Phase E regression artifacts are not PASS.",
                "source_artifact": str(phase_e_summary.get("_path", "")),
            }
        )
    if not bool(model_run_summary.get("governed_run", False)):
        blockers.append(
            {
                "blocker": "governed_training_run",
                "reason": str(model_run_summary.get("reason", "Governed training checkpoint is missing.")),
                "source_artifact": str(model_run_summary.get("checkpoint_path") or model_run_summary.get("journal_path") or ""),
            }
        )
    if not bool(leakage_audit.get("available", False)):
        blockers.append(
            {
                "blocker": "strict_holdout_leakage_audit",
                "reason": str(
                    leakage_audit.get("reason", "contam.strict_holdout_leakage_score is unavailable.")
                ),
                "source_artifact": str(leakage_audit.get("source_path") or ""),
            }
        )
    sidecar_heldout_count = int(document_coverage.get("technical_debt_fixture_count", 0))
    if sidecar_heldout_count > 0:
        blockers.append(
            {
                "blocker": "sidecar_document_pipeline_debt",
                "reason": (
                    f"{sidecar_heldout_count} strict held-out document fixtures still depend on sidecar-backed "
                    "normalization."
                ),
                "source_artifact": str(heldout_packet.get("_path", "")),
            }
        )
    if _safe_float(document_coverage.get("task_family_resolution_coverage"), 0.0) < 1.0:
        blockers.append(
            {
                "blocker": "document_task_family_coverage",
                "reason": "Document packet does not retain full task-family resolution coverage.",
                "source_artifact": str(heldout_packet.get("_path", "")),
            }
        )
    if _safe_float(proof_coverage.get("task_adjudication_policy_resolution_coverage"), 0.0) < 1.0:
        blockers.append(
            {
                "blocker": "proof_adjudication_coverage",
                "reason": "Proof packet does not retain full task adjudication policy coverage.",
                "source_artifact": str(heldout_packet.get("_path", "")),
            }
        )

    snapshot = {
        "schema": "iris.readiness.p1_snapshot/v1",
        "generated_at_utc": utc_now_iso(),
        "profile_id": "P1",
        "baseline_id": str(baseline_id),
        "tolerance_profile_id": str(tolerance_profile_id),
        "change_class": GateContext().change_class,
        "phase_artifact_metadata": {
            "phase_d_artifact_baseline_id": phase_d_summary.get("baseline_id"),
            "phase_e_artifact_baseline_id": phase_e_summary.get("baseline_id"),
            "phase_d_regression_status": phase_d_summary.get("regression.status"),
            "phase_e_regression_status": phase_e_summary.get("regression.status"),
        },
        "artifact_roots": {
            "phase_d_root": str(phase_d_root.resolve()),
            "phase_e_root": str(phase_e_root.resolve()),
            "model_run_dir": str(Path(model_run_dir).resolve()) if model_run_dir is not None else None,
            "leakage_audit_path": str(Path(leakage_audit_path).resolve()) if leakage_audit_path is not None else None,
        },
        "governance_summary": {
            "data_realization_policy_id": policy_bundle.data_realization_policy.data_realization_policy_id,
            "decontam_policy_id": policy_bundle.decontam_policy.decontam_policy_id,
            "benchmark_family_policy_refs": list(
                policy_bundle.data_realization_policy.benchmark_family_policy_refs
            ),
            "learning_objective_bundle_id": learning_bundle.learning_objective_bundle_id,
            "learning_objective_bundle_resolution_source": learning_bundle_source,
            "model_run": model_run_summary,
        },
        "coverage_summary": {
            "document_packet": document_coverage,
            "proof_packet": proof_coverage,
            "runtime_status_counts": {
                "document": _value_counts(document_fixtures, "runtime_status"),
                "proof": _value_counts(proof_fixtures, "runtime_status"),
            },
            "adjudication_status_counts": {
                "document": _value_counts(document_fixtures, "adjudication_status"),
                "proof": _value_counts(proof_fixtures, "adjudication_status"),
            },
        },
        "evidence_posture": {
            "heldout_document_fixture_count": len(document_fixtures),
            "heldout_proof_fixture_count": len(proof_fixtures),
            "sidecar_heldout_fixture_count": sidecar_heldout_count,
            "strict_holdout_leakage_audit": leakage_audit,
            "phase_d_provenance_summary": phase_d_provenance,
        },
        "hard_gate_surface_values": surface_values,
        "residual_blockers": blockers,
    }
    return snapshot


def _evaluate_surface(metric_name: str, current_value: float | None, baseline_value: float | None) -> Dict[str, Any]:
    status = "PASS"
    reasons: List[str] = []
    delta = None if current_value is None or baseline_value is None else float(current_value - baseline_value)

    if current_value is None:
        reasons.append("current value is unavailable")
    if baseline_value is None:
        reasons.append("baseline value is unavailable")
    if reasons:
        return {
            "metric": metric_name,
            "current_value": current_value,
            "baseline_value": baseline_value,
            "delta_vs_baseline": delta,
            "status": "FAIL",
            "reasons": reasons,
        }

    assert current_value is not None
    assert baseline_value is not None

    if metric_name == "rep.document.parse_completeness":
        if current_value < 0.97:
            reasons.append(f"current value {current_value:.6g} is below the >= 0.97 floor")
        if delta < -0.005:
            reasons.append(f"delta {delta:.6g} is below the allowed -0.005 floor")
    elif metric_name == "task.document_grounding_score":
        if delta < 0.01:
            reasons.append(f"delta {delta:.6g} is below the required +0.01 improvement")
    elif metric_name == "failure.credit.collapse_rate":
        if current_value > 0.02:
            reasons.append(f"current value {current_value:.6g} exceeds the <= 0.02 ceiling")
        if delta > 0.005:
            reasons.append(f"delta {delta:.6g} exceeds the allowed +0.005 worsening")
    elif metric_name == "eval.false_accept_rate":
        if delta > 0.0025:
            reasons.append(f"delta {delta:.6g} exceeds the allowed +0.0025 worsening")
    elif metric_name == "eval.calibration_error":
        if delta > 0.005:
            reasons.append(f"delta {delta:.6g} exceeds the allowed +0.005 worsening")
    elif metric_name == "contam.strict_holdout_leakage_score":
        if delta > 0.001:
            reasons.append(f"delta {delta:.6g} exceeds the allowed +0.001 worsening")
    elif metric_name == "provenance.parser_coverage":
        if current_value < 0.95:
            reasons.append(f"current value {current_value:.6g} is below the >= 0.95 floor")
        if delta < -0.01:
            reasons.append(f"delta {delta:.6g} is below the allowed -0.01 floor")
    elif metric_name == "provenance.verifier_coverage":
        if current_value < 0.90:
            reasons.append(f"current value {current_value:.6g} is below the >= 0.90 floor")
        if delta < -0.01:
            reasons.append(f"delta {delta:.6g} is below the allowed -0.01 floor")

    if reasons:
        status = "FAIL"
    return {
        "metric": metric_name,
        "current_value": current_value,
        "baseline_value": baseline_value,
        "delta_vs_baseline": delta,
        "status": status,
        "reasons": reasons,
    }


def _count_trailing_gate_passes(
    history_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_id: str,
    tolerance_profile_id: str,
) -> int:
    count = 0
    for row in reversed(history_rows):
        if str(row.get("baseline_id", "")) != baseline_id:
            break
        if str(row.get("tolerance_profile_id", "")) != tolerance_profile_id:
            break
        if str(row.get("run_gate_status", "")).upper() != "PASS":
            break
        count += 1
    return count


def evaluate_p1_readiness(
    *,
    snapshot: Mapping[str, Any],
    baseline_snapshot: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    current_surfaces = dict(snapshot.get("hard_gate_surface_values", {}))
    baseline_surfaces = dict(baseline_snapshot.get("hard_gate_surface_values", {}))
    surface_evaluations: Dict[str, Any] = {}
    for metric_name in _P1_REQUIRED_SURFACES:
        current_value = _metric_from_mapping(dict(current_surfaces.get(metric_name, {})), "value")
        baseline_value = _metric_from_mapping(dict(baseline_surfaces.get(metric_name, {})), "value")
        surface_evaluations[metric_name] = _evaluate_surface(metric_name, current_value, baseline_value)

    residual_blockers = [dict(item) for item in snapshot.get("residual_blockers", [])]
    if str(baseline_snapshot.get("baseline_id", "")) != str(snapshot.get("baseline_id", "")):
        residual_blockers.append(
            {
                "blocker": "baseline_id_mismatch",
                "reason": "baseline snapshot uses a different readiness baseline_id.",
            }
        )
    if str(baseline_snapshot.get("tolerance_profile_id", "")) != str(snapshot.get("tolerance_profile_id", "")):
        residual_blockers.append(
            {
                "blocker": "tolerance_profile_id_mismatch",
                "reason": "baseline snapshot uses a different tolerance_profile_id.",
            }
        )

    surface_failures = [
        metric_name
        for metric_name, evaluation in surface_evaluations.items()
        if str(evaluation.get("status", "FAIL")).upper() != "PASS"
    ]
    run_gate_status = "PASS" if not surface_failures and not residual_blockers else "FAIL"
    history = list(history_rows or [])
    current_history_row = {
        "generated_at_utc": snapshot.get("generated_at_utc"),
        "profile_id": snapshot.get("profile_id"),
        "baseline_id": snapshot.get("baseline_id"),
        "tolerance_profile_id": snapshot.get("tolerance_profile_id"),
        "run_gate_status": run_gate_status,
    }
    consecutive_gate_passed_runs = _count_trailing_gate_passes(
        history + [current_history_row],
        baseline_id=str(snapshot.get("baseline_id", "")),
        tolerance_profile_id=str(snapshot.get("tolerance_profile_id", "")),
    )
    promotion_status = "PASS" if run_gate_status == "PASS" and consecutive_gate_passed_runs >= 3 else "BLOCKED"
    return {
        "schema": "iris.readiness.p1_packet/v1",
        "generated_at_utc": utc_now_iso(),
        "profile_id": snapshot.get("profile_id", "P1"),
        "baseline_id": snapshot.get("baseline_id"),
        "tolerance_profile_id": snapshot.get("tolerance_profile_id"),
        "run_gate_status": run_gate_status,
        "promotion_status": promotion_status,
        "required_consecutive_gate_passed_runs": 3,
        "consecutive_gate_passed_runs": consecutive_gate_passed_runs,
        "hard_gate_surface_evaluations": surface_evaluations,
        "surface_failures": surface_failures,
        "residual_blockers": residual_blockers,
        "snapshot": dict(snapshot),
    }


def append_readiness_history(history_path: Path, packet: Mapping[str, Any]) -> None:
    snapshot = dict(packet.get("snapshot", {}))
    _append_jsonl(
        Path(history_path),
        {
            "generated_at_utc": packet.get("generated_at_utc"),
            "profile_id": packet.get("profile_id"),
            "baseline_id": packet.get("baseline_id"),
            "tolerance_profile_id": packet.get("tolerance_profile_id"),
            "run_gate_status": packet.get("run_gate_status"),
            "promotion_status": packet.get("promotion_status"),
            "consecutive_gate_passed_runs": packet.get("consecutive_gate_passed_runs"),
            "surface_failures": list(packet.get("surface_failures", [])),
            "artifact_roots": dict(snapshot.get("artifact_roots", {})),
        },
    )


def build_p1_readiness_report(packet: Mapping[str, Any]) -> str:
    lines = [
        "# P1 Readiness Packet",
        "",
        f"- Profile: {packet.get('profile_id', 'P1')}",
        f"- Baseline ID: {packet.get('baseline_id', '')}",
        f"- Tolerance Profile ID: {packet.get('tolerance_profile_id', '')}",
        f"- Run Gate Status: **{packet.get('run_gate_status', 'FAIL')}**",
        f"- Promotion Status: **{packet.get('promotion_status', 'BLOCKED')}**",
        f"- Consecutive Gate-Passed Runs: {packet.get('consecutive_gate_passed_runs', 0)}/3",
        "",
        "## Hard-Gate Surfaces",
    ]
    for metric_name, evaluation in sorted(
        dict(packet.get("hard_gate_surface_evaluations", {})).items()
    ):
        reasons = list(evaluation.get("reasons", []))
        reason_text = "; ".join(reasons) if reasons else "within tolerance"
        lines.append(
            f"- {metric_name}: {evaluation.get('status', 'FAIL')} "
            f"(current={evaluation.get('current_value')}, baseline={evaluation.get('baseline_value')}, "
            f"delta={evaluation.get('delta_vs_baseline')}) -> {reason_text}"
        )
    lines.append("")
    lines.append("## Residual Blockers")
    blockers = list(packet.get("residual_blockers", []))
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker.get('blocker', 'blocker')}: {blocker.get('reason', '')}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"
