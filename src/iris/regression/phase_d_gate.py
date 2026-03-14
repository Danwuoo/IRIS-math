from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..train import load_policy_bundle_for_profile_phase
from .math_native_fixtures import load_document_eval_fixtures, load_proof_eval_fixtures
from .math_native_phase_eval import (
    SIDECAR_TECH_DEBT_NOTE,
    default_document_fixture_root,
    default_proof_fixture_root,
    document_eval_packet,
    evaluate_document_fixture,
    evaluate_proof_fixture,
    hash_payload,
    phase_root_paths,
    proof_eval_packet,
    read_json,
    repo_root,
    write_json,
)
from .phase_c_gate import (
    GateContext,
    Tolerances,
    _is_phase_c_or_later,
    _safe_float,
    build_concept_breakdown as build_arc_compat_concept_breakdown,
    build_credit_routing_diff,
    build_failure_profile_diff,
    build_h100_packet_summary,
    build_paired_representation_diff as build_arc_compat_paired_representation_diff,
    build_resume_consistency_packet,
    build_summary_report,
    evaluate_s6_status,
    evaluate_s7_status,
    evaluate_s8_status,
    utc_now_iso,
)

_EXPECTED_FAILURE_IMPACT = (
    "Improves visibility and contract compliance for F_REP, F_PROC, F_SEARCH, F_EVAL while keeping "
    "failure.credit informative."
)


def build_concept_breakdown_v2(
    *,
    document_results: Sequence[Mapping[str, Any]],
    proof_results: Sequence[Mapping[str, Any]],
    context: GateContext,
) -> Dict[str, Any]:
    concepts = [dict(item) for item in document_results] + [dict(item) for item in proof_results]
    payload = {
        "schema": "iris.regression.concept_breakdown/v2",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "concept_count": len(concepts),
        "document_fixture_count": len(document_results),
        "proof_fixture_count": len(proof_results),
        "status": "PASS" if concepts else "FAIL",
        "aggregate": {
            "rep.document.parse_completeness.mean": _mean_metric(concepts, "rep.document.parse_completeness"),
            "task.document_grounding_score.mean": _mean_metric(concepts, "task.document_grounding_score"),
            "proof.evidence_coverage.mean": _mean_metric(concepts, "proof.evidence_coverage"),
            "concept.isolation_score.mean": _mean_metric(concepts, "concept.isolation_score"),
            "concept.leakage_score.mean": _mean_metric(concepts, "concept.leakage_score"),
        },
        "documents": [dict(item) for item in document_results],
        "proofs": [dict(item) for item in proof_results],
        "concepts": concepts,
    }
    if not concepts:
        payload["reason"] = "No math-native document/proof fixtures were available."
    payload["artifact_hash"] = hash_payload(payload)
    return payload


def build_paired_representation_diff_v2(
    *,
    document_results: Sequence[Mapping[str, Any]],
    context: GateContext,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in document_results:
        pair_group_id = str(row.get("pair_group_id", "")).strip()
        if pair_group_id:
            grouped.setdefault(pair_group_id, []).append(row)

    pairs: List[Dict[str, Any]] = []
    for pair_group_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: str(item.get("pair_variant_id", "")))
        if len(ordered) < 2:
            continue
        left = ordered[0]
        right = ordered[1]
        grounding_gap = abs(
            _safe_float(left.get("task.document_grounding_score"), 0.0)
            - _safe_float(right.get("task.document_grounding_score"), 0.0)
        )
        parse_gap = abs(
            _safe_float(left.get("rep.document.parse_completeness"), 0.0)
            - _safe_float(right.get("rep.document.parse_completeness"), 0.0)
        )
        asymmetry = str(left.get("status", "FAIL")) != str(right.get("status", "FAIL"))
        pairs.append(
            {
                "pair_id": pair_group_id,
                "left_fixture_id": str(left.get("fixture_id", "")),
                "right_fixture_id": str(right.get("fixture_id", "")),
                "pair.invariance.gap": float(max(grounding_gap, parse_gap)),
                "pair.asymmetry": bool(asymmetry),
                "status": "PASS" if not asymmetry else "FAIL",
                "failure_reasons": [] if not asymmetry else ["paired variants diverged in pass/fail status"],
            }
        )

    pair_count = len(pairs)
    payload = {
        "schema": "iris.regression.paired_representation_diff/v2",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "pair_count": pair_count,
        "paired.asymmetry_rate": (
            sum(1.0 for row in pairs if bool(row.get("pair.asymmetry", False))) / float(pair_count)
            if pair_count
            else 1.0
        ),
        "paired.invariance.gap": (
            sum(_safe_float(row.get("pair.invariance.gap"), 0.0) for row in pairs) / float(pair_count)
            if pair_count
            else 1.0
        ),
        "status": "PASS" if pair_count > 0 else "FAIL",
        "pairs": pairs,
    }
    if pair_count <= 0:
        payload["reason"] = "No paired math-native reformulation fixtures were available."
    payload["artifact_hash"] = hash_payload(payload)
    return payload


def build_failure_profile_diff_v2(*, resume_packet: Mapping[str, Any], context: GateContext) -> Dict[str, Any]:
    base = dict(build_failure_profile_diff(resume_packet, context))
    payload = {
        **base,
        "schema": "iris.regression.failure_profile_diff/v2",
        "failure_profile.kl_divergence": float(base.get("max_failure_profile_kl", 0.0)),
        "failure_profile.l1_distance": float(base.get("max_failure_profile_kl", 0.0)),
    }
    payload["artifact_hash"] = hash_payload(payload)
    return payload


def evaluate_s3_status_v2(
    *,
    failure_profile_diff: Mapping[str, Any],
    tolerances: Tolerances,
    phase: str,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if str(failure_profile_diff.get("status")) != "PASS":
        reasons.append("failure_profile_diff artifact is unavailable")
    kl_value = _safe_float(
        failure_profile_diff.get("failure_profile.kl_divergence", failure_profile_diff.get("max_failure_profile_kl")),
        float("inf"),
    )
    if kl_value > tolerances.failure_profile_kl_epsilon:
        reasons.append(
            f"failure profile KL drift {kl_value:.6g} exceeds tolerance {tolerances.failure_profile_kl_epsilon:.6g}"
        )
    if _is_phase_c_or_later(phase) and _safe_float(failure_profile_diff.get("failure_profile.l1_distance"), 0.0) < 0.0:
        reasons.append("failure profile L1 distance must be non-negative")
    return ("PASS" if not reasons else "FAIL", reasons)


def evaluate_s4_status_v2(
    *,
    current_concept_breakdown: Mapping[str, Any],
    baseline_concept_breakdown: Mapping[str, Any] | None,
    tolerances: Tolerances,
    phase: str,
) -> tuple[str, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    details: List[Dict[str, Any]] = []
    current_concepts = current_concept_breakdown.get("concepts", [])
    if not isinstance(current_concepts, list) or not current_concepts:
        reasons.append("current concept_breakdown has no concept entries")
    if _is_phase_c_or_later(phase) and not isinstance(baseline_concept_breakdown, Mapping):
        reasons.append("baseline concept_breakdown is required in Phase C+")
        return "FAIL", reasons, details
    baseline_by_id = {
        str(item.get("concept_id", "")): item
        for item in (baseline_concept_breakdown or {}).get("concepts", [])
        if isinstance(item, Mapping)
    }
    for current in current_concepts if isinstance(current_concepts, list) else []:
        concept_id = str(current.get("concept_id", ""))
        if str(current.get("status", "FAIL")) != "PASS":
            details.append(
                {
                    "metric": "document_or_proof_fixture.status",
                    "concept_id": concept_id,
                    "reason": "; ".join(current.get("failure_reasons", [])) or "fixture failed absolute checks",
                }
            )
        baseline = baseline_by_id.get(concept_id)
        if baseline is None:
            details.append({"metric": "concept.baseline_presence", "concept_id": concept_id, "reason": "concept missing in baseline"})
            continue
        for metric_name in ("rep.document.parse_completeness", "task.document_grounding_score", "proof.evidence_coverage"):
            current_value = _safe_float(current.get(metric_name), 0.0)
            baseline_value = _safe_float(baseline.get(metric_name), 0.0)
            delta = current_value - baseline_value
            if delta < -tolerances.metric_epsilon:
                details.append(
                    {
                        "metric": metric_name,
                        "concept_id": concept_id,
                        "baseline": baseline_value,
                        "current": current_value,
                        "delta": delta,
                        "threshold": -tolerances.metric_epsilon,
                        "reason": "metric decreased beyond tolerance",
                    }
                )
    if details:
        reasons.append(f"{len(details)} document/proof regression violations detected")
    return ("PASS" if not reasons else "FAIL", reasons, details)


def evaluate_s5_status_v2(
    *,
    current_paired_diff: Mapping[str, Any],
    baseline_paired_diff: Mapping[str, Any] | None,
    tolerances: Tolerances,
    phase: str,
) -> tuple[str, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    details: List[Dict[str, Any]] = []
    if _is_phase_c_or_later(phase) and not isinstance(baseline_paired_diff, Mapping):
        reasons.append("baseline paired_representation_diff is required in Phase C+")
        return "FAIL", reasons, details
    current_pair_count = int(_safe_float(current_paired_diff.get("pair_count"), 0.0))
    baseline_pair_count = int(_safe_float((baseline_paired_diff or {}).get("pair_count"), 0.0))
    if current_pair_count <= 0:
        reasons.append("current paired_representation_diff has no pairs")
        return "FAIL", reasons, details
    if baseline_pair_count <= 0:
        reasons.append("baseline paired_representation_diff has no pairs")
        return "FAIL", reasons, details
    baseline_by_id = {
        str(item.get("pair_id", "")): item for item in (baseline_paired_diff or {}).get("pairs", []) if isinstance(item, Mapping)
    }
    for current in current_paired_diff.get("pairs", []) if isinstance(current_paired_diff.get("pairs", []), list) else []:
        pair_id = str(current.get("pair_id", ""))
        if str(current.get("status", "FAIL")) != "PASS":
            details.append({"metric": "paired_variant.status", "pair_id": pair_id, "reason": "; ".join(current.get("failure_reasons", []))})
        baseline = baseline_by_id.get(pair_id)
        if baseline is None:
            details.append({"metric": "pair.baseline_presence", "pair_id": pair_id, "reason": "pair missing in baseline"})
            continue
        gap_delta = _safe_float(current.get("pair.invariance.gap"), 0.0) - _safe_float(baseline.get("pair.invariance.gap"), 0.0)
        if gap_delta > tolerances.paired_invariance_gap_delta_epsilon:
            details.append(
                {
                    "metric": "paired.invariance.gap",
                    "pair_id": pair_id,
                    "baseline": _safe_float(baseline.get("pair.invariance.gap"), 0.0),
                    "current": _safe_float(current.get("pair.invariance.gap"), 0.0),
                    "delta": gap_delta,
                    "threshold": tolerances.paired_invariance_gap_delta_epsilon,
                    "reason": "invariance gap increased beyond tolerance",
                }
            )
    aggregate_gap_delta = _safe_float(current_paired_diff.get("paired.invariance.gap"), 0.0) - _safe_float((baseline_paired_diff or {}).get("paired.invariance.gap"), 0.0)
    aggregate_asymmetry_delta = _safe_float(current_paired_diff.get("paired.asymmetry_rate"), 0.0) - _safe_float((baseline_paired_diff or {}).get("paired.asymmetry_rate"), 0.0)
    if aggregate_gap_delta > tolerances.paired_invariance_gap_delta_epsilon:
        details.append({"metric": "paired.invariance.gap", "delta": aggregate_gap_delta, "reason": "aggregate invariance gap increased beyond tolerance"})
    if aggregate_asymmetry_delta > tolerances.paired_asymmetry_delta_epsilon:
        details.append({"metric": "paired.asymmetry_rate", "delta": aggregate_asymmetry_delta, "reason": "aggregate asymmetry rate increased beyond tolerance"})
    if details:
        reasons.append(f"{len(details)} paired reformulation regression violations detected")
    return ("PASS" if not reasons else "FAIL", reasons, details)


def run_phase_d_gate(
    *,
    context: GateContext,
    tolerances: Tolerances,
    output_dir: Path,
    model_run_dir: Path,
    conceptarc_corpus: Path | None,
    rearc_tasks: Path | None,
    baseline_report_dir: Path | None,
    phase_root: Path,
    h100_path_map: Mapping[str, Path],
    s1_status: str,
    s1_reasons: Sequence[str],
    s2_status: str,
    s2_reasons: Sequence[str],
    s1_output: Mapping[str, Any],
    s2_output: Mapping[str, Any],
    s2_mounted_output: Mapping[str, Any],
    max_reasoning_cycles: int,
    termination_threshold: float,
    seed: int,
    pairing_policy: str = "adjacent",
    freeze_baseline: bool = False,
    document_fixture_root: Path | None = None,
    proof_fixture_root: Path | None = None,
) -> Dict[str, Any]:
    del max_reasoning_cycles, termination_threshold, pairing_policy
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_bundle = load_policy_bundle_for_profile_phase("P1", context.phase)
    document_fixtures = load_document_eval_fixtures(document_fixture_root or default_document_fixture_root(), eval_partition="diagnostic")
    proof_fixtures = load_proof_eval_fixtures(proof_fixture_root or default_proof_fixture_root(), eval_partition="diagnostic")
    document_results = []
    document_bundles: Dict[str, Any] = {}
    for index, fixture in enumerate(document_fixtures):
        result, document_bundle = evaluate_document_fixture(fixture, policy_bundle=policy_bundle, hidden_dim=16, seed=seed + index)
        document_results.append(result)
        document_bundles[fixture.fixture_id] = document_bundle
    proof_results = [
        evaluate_proof_fixture(fixture, policy_bundle=policy_bundle, document_bundles=document_bundles, hidden_dim=16, seed=seed + 100 + index)
        for index, fixture in enumerate(proof_fixtures)
    ]
    document_packet = document_eval_packet(document_results=document_results, context=context, eval_partition="diagnostic")
    proof_packet = proof_eval_packet(proof_results=proof_results, context=context, eval_partition="diagnostic")
    concept_breakdown = build_concept_breakdown_v2(document_results=document_results, proof_results=proof_results, context=context)
    paired_diff = build_paired_representation_diff_v2(document_results=document_results, context=context)
    resume_packet = build_resume_consistency_packet(phase_root_paths(Path(phase_root)), context)
    failure_profile_diff = build_failure_profile_diff_v2(resume_packet=resume_packet, context=context)
    credit_routing_diff = build_credit_routing_diff(resume_packet, context)
    h100_packet = build_h100_packet_summary(h100_path_map, context, tolerances=tolerances)
    baseline_concept = None
    baseline_paired = None
    baseline_errors: List[str] = []
    if baseline_report_dir is not None:
        baseline_dir = Path(baseline_report_dir)
        if freeze_baseline:
            write_json(baseline_dir / "concept_breakdown.json", concept_breakdown)
            write_json(baseline_dir / "paired_representation_diff.json", paired_diff)
        concept_path = baseline_dir / "concept_breakdown.json"
        paired_path = baseline_dir / "paired_representation_diff.json"
        baseline_concept = read_json(concept_path) if concept_path.exists() else None
        baseline_paired = read_json(paired_path) if paired_path.exists() else None
        if baseline_concept is None:
            baseline_errors.append(f"missing baseline artifact: {concept_path}")
        if baseline_paired is None:
            baseline_errors.append(f"missing baseline artifact: {paired_path}")
    s3_status, s3_reasons = evaluate_s3_status_v2(failure_profile_diff=failure_profile_diff, tolerances=tolerances, phase=context.phase)
    s4_status, s4_reasons, s4_details = evaluate_s4_status_v2(current_concept_breakdown=concept_breakdown, baseline_concept_breakdown=baseline_concept, tolerances=tolerances, phase=context.phase)
    s5_status, s5_reasons, s5_details = evaluate_s5_status_v2(current_paired_diff=paired_diff, baseline_paired_diff=baseline_paired, tolerances=tolerances, phase=context.phase)
    s6_status, s6_reasons = evaluate_s6_status(resume_packet, credit_routing_diff, tolerances)
    s7_status, s7_reasons = evaluate_s7_status(resume_packet, tolerances)
    s8_status, s8_reasons = evaluate_s8_status(resume_packet, tolerances)
    if baseline_report_dir is None and _is_phase_c_or_later(context.phase):
        s4_status = "FAIL"
        s5_status = "FAIL"
        s4_reasons.append("baseline_report_dir is required for S4 in Phase C+.")
        s5_reasons.append("baseline_report_dir is required for S5 in Phase C+.")
    if baseline_errors:
        s4_status = "FAIL"
        s5_status = "FAIL"
        s4_reasons.extend(baseline_errors)
        s5_reasons.extend(baseline_errors)
    suite_status = {"S1": s1_status, "S2": s2_status, "S3": s3_status, "S4": s4_status, "S5": s5_status, "S6": s6_status, "S7": s7_status, "S8": s8_status, "S8_h100_packet": str(h100_packet.get("s8_status_for_h100_packet", "FAIL"))}
    violations: List[Dict[str, Any]] = []
    for suite_name, reasons, metric, details in (
        ("S1", s1_reasons if str(s1_output.get("status", "")).upper() == "PASS" else list(s1_reasons) + ["S1 output status is not PASS"], "smoke runtime checks", None),
        ("S2", list(s2_reasons) + ([] if str(s2_output.get("status", "")).upper() == "PASS" and str(s2_mounted_output.get("status", "")).upper() == "PASS" else ["S2/S2M output status is not PASS"]), "State IR + L0-L6 interface layer", None),
        ("S3", s3_reasons, "failure taxonomy histogram drift", None),
        ("S4", s4_reasons, "rep.document.parse_completeness / task.document_grounding_score / proof.evidence_coverage", s4_details),
        ("S5", s5_reasons, "paired.asymmetry_rate / paired.invariance.gap", s5_details),
        ("S6", s6_reasons, "failure.credit stability", None),
        ("S7", s7_reasons, "pretraining diagnostics deltas", None),
        ("S8", s8_reasons, "resume consistency drift / coverage", None),
    ):
        if suite_status.get(suite_name) != "PASS":
            violations.append({"suite": suite_name, "reason": "; ".join(dict.fromkeys(reasons)), "metric": metric, "phase": context.phase, "details": details})
    if suite_status["S8_h100_packet"] != "PASS":
        violations.append({"suite": "S8", "reason": str(h100_packet.get("block_reason", "H100 packet coverage incomplete.")), "metric": "crash-class coverage (H100 packet)", "phase": context.phase})
    compatibility_arc_appendix = {
        "schema": "iris.regression.compatibility_arc_appendix/v1",
        "phase": context.phase,
        "status": "AVAILABLE" if (conceptarc_corpus and Path(conceptarc_corpus).exists()) or (rearc_tasks and Path(rearc_tasks).exists()) else "SKIPPED",
        "notes": ["Conflict resolved in favor of active v2 docs: ARC/ConceptARC is compatibility-only."],
        "concept_breakdown": build_arc_compat_concept_breakdown(Path(conceptarc_corpus), context) if conceptarc_corpus and Path(conceptarc_corpus).exists() else {"status": "SKIPPED"},
        "paired_representation_diff": build_arc_compat_paired_representation_diff(Path(rearc_tasks), context) if rearc_tasks and Path(rearc_tasks).exists() else {"status": "SKIPPED"},
    }
    provenance_summary = {
        "schema": "iris.regression.provenance_contamination_summary/v1",
        "profile_id": "P1",
        "phase": context.phase,
        "data_realization_policy_id": policy_bundle.data_realization_policy.data_realization_policy_id,
        "decontam_policy_id": policy_bundle.decontam_policy.decontam_policy_id,
        "benchmark_family_policy_refs": list(policy_bundle.data_realization_policy.benchmark_family_policy_refs),
        "document_fixture_count": len(document_fixtures),
        "proof_fixture_count": len(proof_fixtures),
        "heldout_posture": "Tier 2/3 remain eval-only; no new train-visible benchmark exposure introduced.",
        "frontiermath_posture": "original FrontierMath remains untouched; local held-out fixtures are synthetic or fixture-backed only.",
        "provenance_guardrails": [SIDECAR_TECH_DEBT_NOTE],
    }
    notes = [
        "Conflict resolved: prior ARC-centric Phase D implementation was demoted to compatibility-only reporting.",
        f"math_native.document_fixture_count={len(document_fixtures)}",
        f"math_native.proof_fixture_count={len(proof_fixtures)}",
        f"model_run_dir_exists={Path(model_run_dir).exists()}",
        f"s8_local_packet_drift_clear={resume_packet.get('all_drift_labels_clear')}",
        f"s8_h100_packet_status={h100_packet.get('s8_status_for_h100_packet')}",
        SIDECAR_TECH_DEBT_NOTE,
    ]
    summary_report = build_summary_report(context=context, suite_status=suite_status, violations=violations, generated_at_utc=utc_now_iso(), notes=notes, evidence_paths=[str(output_dir / "summary_report.json"), str(output_dir / "concept_breakdown.json"), str(output_dir / "paired_representation_diff.json"), str(output_dir / "document_eval_packet.json"), str(output_dir / "proof_eval_packet.json")])
    checklist = dict(summary_report.get("completion_checklist", {}))
    checklist["expected_failure_category_impact"] = _EXPECTED_FAILURE_IMPACT
    checklist["technical_debt_guardrails_introduced"] = SIDECAR_TECH_DEBT_NOTE
    checklist["benchmark_contamination_provenance_implications"] = "Tier 2/3 remain eval-only; held-out packets are fixture-backed only."
    checklist["termination"] = "Done" if summary_report["regression.status"] == "PASS" else "Blocked"
    summary_report["completion_checklist"] = checklist
    artifact_paths = {name: str(output_dir / name) for name in ("summary_report.json", "concept_breakdown.json", "paired_representation_diff.json", "failure_profile_diff.json", "credit_routing_diff.json", "resume_consistency_packet.json", "h100_packet_summary.json", "document_eval_packet.json", "proof_eval_packet.json", "compatibility_arc_appendix.json", "provenance_contamination_summary.json", "s1_output.json", "s2_output.json", "s2_mounted_output.json", "phase_d_gate_report.md")}
    write_json(Path(artifact_paths["summary_report.json"]), summary_report)
    write_json(Path(artifact_paths["concept_breakdown.json"]), concept_breakdown)
    write_json(Path(artifact_paths["paired_representation_diff.json"]), paired_diff)
    write_json(Path(artifact_paths["failure_profile_diff.json"]), failure_profile_diff)
    write_json(Path(artifact_paths["credit_routing_diff.json"]), credit_routing_diff)
    write_json(Path(artifact_paths["resume_consistency_packet.json"]), resume_packet)
    write_json(Path(artifact_paths["h100_packet_summary.json"]), h100_packet)
    write_json(Path(artifact_paths["document_eval_packet.json"]), document_packet)
    write_json(Path(artifact_paths["proof_eval_packet.json"]), proof_packet)
    write_json(Path(artifact_paths["compatibility_arc_appendix.json"]), compatibility_arc_appendix)
    write_json(Path(artifact_paths["provenance_contamination_summary.json"]), provenance_summary)
    write_json(Path(artifact_paths["s1_output.json"]), dict(s1_output))
    write_json(Path(artifact_paths["s2_output.json"]), dict(s2_output))
    write_json(Path(artifact_paths["s2_mounted_output.json"]), dict(s2_mounted_output))
    Path(artifact_paths["phase_d_gate_report.md"]).write_text(
        "\n".join(
            [
                "# Phase D Gate Report",
                "",
                "- Primary Surface: math-native document-grounded diagnostics",
                "- ARC Posture: compatibility-only appendix, non-blocking",
                f"- Regression Status: **{summary_report.get('regression.status', 'FAIL')}**",
                "",
                "## Suite Status",
                *[f"- {suite}: {status}" for suite, status in sorted(suite_status.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"summary_report": summary_report, "suite_status": suite_status, "artifact_paths": artifact_paths, "document_eval_packet": document_packet, "proof_eval_packet": proof_packet, "compatibility_arc_appendix": compatibility_arc_appendix}


def _mean_metric(items: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    if not items:
        return 0.0
    return float(sum(_safe_float(item.get(metric_name), 0.0) for item in items) / float(len(items)))
