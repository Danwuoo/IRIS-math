from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from iris.regression import (
    GateContext,
    Tolerances,
    build_concept_breakdown,
    build_credit_routing_diff,
    build_failure_profile_diff,
    build_h100_packet_summary,
    build_paired_representation_diff,
    build_resume_consistency_packet,
    build_summary_report,
    evaluate_s3_status,
    evaluate_s4_status,
    evaluate_s5_status,
    evaluate_s6_status,
    evaluate_s7_status,
    evaluate_s8_status,
    utc_now_iso,
    write_phase_c_gate_artifacts,
)


def _run_json_suite(command: List[str]) -> Dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    parsed: Dict[str, Any] = {}
    stdout_lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if stdout_lines:
        try:
            parsed = json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            parsed = {}
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "payload": parsed,
    }


def _suite_status_from_command(record: Dict[str, Any], suite_name: str) -> tuple[str, str]:
    if record["status"] == "PASS":
        return "PASS", ""
    stderr = (record.get("stderr") or "").strip()
    reason = stderr if stderr else f"{suite_name} command failed (rc={record.get('returncode')})"
    return "FAIL", reason


def _load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_suite_payload(
    *,
    record: Dict[str, Any],
    fallback_path: Path,
    suite_key: str,
    allow_fallback: bool,
) -> Dict[str, Any]:
    payload = dict(record.get("payload") or {})
    if record.get("status") == "PASS" and payload:
        return payload
    if allow_fallback and fallback_path.exists():
        fallback_payload = _load_json_file(fallback_path)
        if isinstance(fallback_payload, dict):
            fallback_payload.setdefault("suite", suite_key)
            fallback_payload["fallback_used"] = True
            return fallback_payload
    if payload:
        return payload
    return {"status": "FAIL", "suite": suite_key, "returncode": record.get("returncode")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase C gate artifacts and report.")
    parser.add_argument(
        "--baseline-report-dir",
        type=Path,
        default=None,
        help="Directory containing frozen baseline artifacts (concept_breakdown/paired_representation_diff).",
    )
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("artifacts/phase_c_gate_20260301"),
        help="Root directory for local S8 packet artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase_c_gate_20260301/report"),
        help="Directory where gate artifacts and markdown report are written.",
    )
    parser.add_argument(
        "--conceptarc-corpus",
        type=Path,
        default=Path("tools/ConceptARC/corpus"),
        help="ConceptARC corpus root directory.",
    )
    parser.add_argument(
        "--rearc-tasks",
        type=Path,
        default=Path("data/arc/re_arc/tasks"),
        help="re_arc task directory.",
    )
    parser.add_argument(
        "--rearc-max-tasks",
        type=int,
        default=128,
        help="Max number of re_arc task files sampled for S5 (<=0 means all).",
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        default="toy-baseline",
    )
    parser.add_argument(
        "--tolerance-profile-id",
        type=str,
        default="toy-default",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="C",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device used for S1/S2 suite command checks.",
    )
    parser.add_argument(
        "--metric-epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--kl-epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--credit-epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--h100-uninterrupted",
        type=Path,
        default=Path("artifacts/toy_train_gpu_h100"),
    )
    parser.add_argument(
        "--h100-execute",
        type=Path,
        default=Path("artifacts/toy_resume_h100"),
    )
    parser.add_argument(
        "--h100-pre-commit",
        type=Path,
        default=Path("artifacts/toy_resume_h100_pre_commit"),
    )
    parser.add_argument(
        "--h100-post-commit",
        type=Path,
        default=Path("artifacts/toy_resume_h100_post_commit"),
    )
    parser.add_argument(
        "--strict-suite-exec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, suite command failures always fail S1/S2 regardless of payload parsing.",
    )
    parser.add_argument(
        "--reuse-existing-suite-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load existing s1/s2 JSON outputs for diagnostics when suite command fails (still blocks gate).",
    )
    args = parser.parse_args()

    context = GateContext(
        phase=args.phase,
        baseline_id=args.baseline_id,
        tolerance_profile_id=args.tolerance_profile_id,
    )
    tolerances = Tolerances(
        metric_epsilon=args.metric_epsilon,
        failure_profile_kl_epsilon=args.kl_epsilon,
        failure_credit_delta_epsilon=args.credit_epsilon,
        concept_isolation_delta_epsilon=args.metric_epsilon,
        concept_leakage_delta_epsilon=args.metric_epsilon,
        paired_asymmetry_delta_epsilon=args.metric_epsilon,
        paired_invariance_gap_delta_epsilon=args.metric_epsilon,
        s8_metric_delta_epsilon=args.metric_epsilon,
    )

    s1_record = _run_json_suite(
        [sys.executable, "scripts/s1_smoke.py", "--device", args.device]
    )
    s2_record = _run_json_suite([sys.executable, "scripts/s2_structural.py"])
    s2_mounted_record = _run_json_suite(
        [sys.executable, "scripts/s2_mounted.py", "--device", args.device]
    )

    s1_output = _resolve_suite_payload(
        record=s1_record,
        fallback_path=args.output_dir / "s1_output.json",
        suite_key="S1",
        allow_fallback=args.reuse_existing_suite_artifacts,
    )
    s2_output = _resolve_suite_payload(
        record=s2_record,
        fallback_path=args.output_dir / "s2_output.json",
        suite_key="S2",
        allow_fallback=args.reuse_existing_suite_artifacts,
    )
    s2_mounted_output = _resolve_suite_payload(
        record=s2_mounted_record,
        fallback_path=args.output_dir / "s2_mounted_output.json",
        suite_key="S2M",
        allow_fallback=args.reuse_existing_suite_artifacts,
    )

    fallback_suites = [
        name
        for name, payload in (
            ("S1", s1_output),
            ("S2", s2_output),
            ("S2M", s2_mounted_output),
        )
        if bool(payload.get("fallback_used"))
    ]

    s1_status = "PASS"
    s1_reasons: List[str] = []
    if args.strict_suite_exec and s1_record.get("status") != "PASS":
        _, reason = _suite_status_from_command(s1_record, "S1")
        s1_reasons.append(reason)
        s1_status = "FAIL"
    if str(s1_output.get("status", "")).upper() != "PASS":
        _, reason = _suite_status_from_command(s1_record, "S1")
        s1_reasons.append(reason)
        s1_status = "FAIL"
    if "S1" in fallback_suites:
        s1_reasons.append("Fallback artifact used for S1; strict gate forbids fallback pass.")
        s1_status = "FAIL"

    s2_status = "PASS"
    s2_reasons: List[str] = []
    if args.strict_suite_exec and s2_record.get("status") != "PASS":
        _, reason = _suite_status_from_command(s2_record, "S2")
        s2_reasons.append(reason)
        s2_status = "FAIL"
    if args.strict_suite_exec and s2_mounted_record.get("status") != "PASS":
        _, reason = _suite_status_from_command(s2_mounted_record, "S2M")
        s2_reasons.append(reason)
        s2_status = "FAIL"
    if str(s2_output.get("status", "")).upper() != "PASS":
        _, reason = _suite_status_from_command(s2_record, "S2")
        s2_reasons.append(reason)
        s2_status = "FAIL"
    if str(s2_mounted_output.get("status", "")).upper() != "PASS":
        _, reason = _suite_status_from_command(s2_mounted_record, "S2M")
        s2_reasons.append(reason)
        s2_status = "FAIL"
    if "S2" in fallback_suites or "S2M" in fallback_suites:
        s2_reasons.append("Fallback artifact used for S2/S2M; strict gate forbids fallback pass.")
        s2_status = "FAIL"

    concept_breakdown = build_concept_breakdown(args.conceptarc_corpus, context)
    paired_diff = build_paired_representation_diff(
        args.rearc_tasks,
        context,
        max_tasks=args.rearc_max_tasks,
    )

    local_s8_paths = {
        "uninterrupted": args.phase_root / "s8_uninterrupted",
        "execute_crash": args.phase_root / "s8_execute",
        "pre_commit_crash": args.phase_root / "s8_pre_commit",
        "post_commit_crash": args.phase_root / "s8_post_commit",
    }
    resume_packet = build_resume_consistency_packet(local_s8_paths, context)
    failure_profile_diff = build_failure_profile_diff(resume_packet, context)
    credit_routing_diff = build_credit_routing_diff(resume_packet, context)

    h100_packet = build_h100_packet_summary(
        {
            "uninterrupted": args.h100_uninterrupted,
            "execute_crash": args.h100_execute,
            "pre_commit_crash": args.h100_pre_commit,
            "post_commit_crash": args.h100_post_commit,
        },
        context,
        tolerances=tolerances,
    )

    baseline_concept_breakdown: Dict[str, Any] | None = None
    baseline_paired_diff: Dict[str, Any] | None = None
    baseline_lookup_errors: List[str] = []
    if args.baseline_report_dir is not None:
        baseline_concept_path = args.baseline_report_dir / "concept_breakdown.json"
        baseline_paired_path = args.baseline_report_dir / "paired_representation_diff.json"
        if baseline_concept_path.exists():
            baseline_concept_breakdown = _load_json_file(baseline_concept_path)
        else:
            baseline_lookup_errors.append(f"missing baseline artifact: {baseline_concept_path}")
        if baseline_paired_path.exists():
            baseline_paired_diff = _load_json_file(baseline_paired_path)
        else:
            baseline_lookup_errors.append(f"missing baseline artifact: {baseline_paired_path}")

    s3_status, s3_reasons = evaluate_s3_status(failure_profile_diff, tolerances)
    s4_status, s4_reasons, s4_details = evaluate_s4_status(
        current_concept_breakdown=concept_breakdown,
        baseline_concept_breakdown=baseline_concept_breakdown,
        tolerances=tolerances,
        phase=context.phase,
    )
    s5_status, s5_reasons, s5_details = evaluate_s5_status(
        current_paired_diff=paired_diff,
        baseline_paired_diff=baseline_paired_diff,
        tolerances=tolerances,
        phase=context.phase,
    )
    s6_status, s6_reasons = evaluate_s6_status(resume_packet, credit_routing_diff, tolerances)
    s7_status, s7_reasons = evaluate_s7_status(resume_packet, tolerances)
    s8_status, s8_reasons = evaluate_s8_status(resume_packet, tolerances)

    if args.baseline_report_dir is None and str(context.phase).upper() in {"C", "D", "E"}:
        s4_status = "FAIL"
        s5_status = "FAIL"
        s4_reasons.append("baseline_report_dir is required for S4 in Phase C+.")
        s5_reasons.append("baseline_report_dir is required for S5 in Phase C+.")

    if baseline_lookup_errors:
        if str(context.phase).upper() in {"C", "D", "E"}:
            s4_status = "FAIL"
            s5_status = "FAIL"
        s4_reasons.extend(baseline_lookup_errors)
        s5_reasons.extend(baseline_lookup_errors)

    suite_status = {
        "S1": s1_status,
        "S2": s2_status,
        "S3": s3_status,
        "S4": s4_status,
        "S5": s5_status,
        "S6": s6_status,
        "S7": s7_status,
        "S8": s8_status,
        "S8_h100_packet": str(h100_packet.get("s8_status_for_h100_packet", "FAIL")),
    }

    violations: List[Dict[str, Any]] = []
    if s1_status != "PASS":
        violations.append(
            {
                "suite": "S1",
                "reason": "; ".join(dict.fromkeys(s1_reasons)) or "S1 failed.",
                "metric": "smoke runtime checks",
                "phase": context.phase,
                "suspected_level": "L0-L6",
            }
        )
    if s2_status != "PASS":
        violations.append(
            {
                "suite": "S2",
                "reason": "; ".join(s2_reasons),
                "metric": "structural contract checks",
                "phase": context.phase,
                "suspected_level": "State IR + L0-L6 interface layer",
            }
        )
    if s3_status != "PASS":
        violations.append(
            {
                "suite": "S3",
                "reason": "; ".join(s3_reasons),
                "metric": "failure profile KL divergence",
                "phase": context.phase,
                "suspected_level": "L6 routing / regression harness",
            }
        )
    if s4_status != "PASS":
        violations.append(
            {
                "suite": "S4",
                "reason": "; ".join(dict.fromkeys(s4_reasons))
                or str(concept_breakdown.get("reason", "ConceptARC breakdown unavailable.")),
                "metric": "concept.isolation_score / concept.leakage_score (baseline+tolerance)",
                "phase": context.phase,
                "suspected_level": "L5",
                "details": s4_details,
            }
        )
    if s5_status != "PASS":
        violations.append(
            {
                "suite": "S5",
                "reason": "; ".join(dict.fromkeys(s5_reasons))
                or str(paired_diff.get("reason", "Paired-task regression artifact unavailable.")),
                "metric": "paired.invariance.gap / asymmetry rate (baseline+tolerance)",
                "phase": context.phase,
                "suspected_level": "L0/L1/L2",
                "details": s5_details,
            }
        )
    if s6_status != "PASS":
        violations.append(
            {
                "suite": "S6",
                "reason": "; ".join(s6_reasons),
                "metric": "failure.credit stability",
                "phase": context.phase,
                "suspected_level": "L6",
            }
        )
    if s7_status != "PASS":
        violations.append(
            {
                "suite": "S7",
                "reason": "; ".join(s7_reasons),
                "metric": "pretraining diagnostics deltas",
                "phase": context.phase,
                "suspected_level": "L3/L6",
            }
        )
    if s8_status != "PASS":
        violations.append(
            {
                "suite": "S8",
                "reason": "; ".join(s8_reasons),
                "metric": "resume consistency drift / coverage",
                "phase": context.phase,
                "suspected_level": "L3/L6 + training transaction path",
            }
        )
    if suite_status["S8_h100_packet"] != "PASS":
        violations.append(
            {
                "suite": "S8",
                "reason": str(h100_packet.get("block_reason", "H100 packet coverage incomplete.")),
                "metric": "crash-class coverage (H100 packet)",
                "phase": context.phase,
                "suspected_level": "L3/L6 + training transaction path",
            }
        )
    if fallback_suites:
        violations.append(
            {
                "suite": "S1/S2",
                "reason": "Fallback artifacts used for suites: " + ", ".join(fallback_suites),
                "metric": "regression.harness.fallback_used",
                "phase": context.phase,
                "suspected_level": "L6 / regression harness",
            }
        )

    expected_artifacts = [
        str(args.output_dir / "s1_output.json"),
        str(args.output_dir / "s2_output.json"),
        str(args.output_dir / "s2_mounted_output.json"),
        str(args.output_dir / "concept_breakdown.json"),
        str(args.output_dir / "paired_representation_diff.json"),
        str(args.output_dir / "failure_profile_diff.json"),
        str(args.output_dir / "credit_routing_diff.json"),
        str(args.output_dir / "resume_consistency_packet.json"),
        str(args.output_dir / "h100_packet_summary.json"),
    ]
    notes = [
        f"S8 local packet drift_clear={resume_packet.get('all_drift_labels_clear')}",
        f"S8 h100 packet status={h100_packet.get('s8_status_for_h100_packet')}",
        f"strict_suite_exec={args.strict_suite_exec}",
        f"reuse_existing_suite_artifacts={args.reuse_existing_suite_artifacts}",
    ]
    if fallback_suites:
        notes.append("Fallback artifacts used for suites: " + ", ".join(fallback_suites))

    summary_report = build_summary_report(
        context=context,
        suite_status=suite_status,
        violations=violations,
        generated_at_utc=utc_now_iso(),
        notes=notes,
        evidence_paths=expected_artifacts,
    )

    artifact_paths = write_phase_c_gate_artifacts(
        output_dir=args.output_dir,
        summary_report=summary_report,
        concept_breakdown=concept_breakdown,
        paired_representation_diff=paired_diff,
        failure_profile_diff=failure_profile_diff,
        credit_routing_diff=credit_routing_diff,
        resume_packet=resume_packet,
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2_mounted_output,
        h100_packet=h100_packet,
    )

    print(
        json.dumps(
            {
                "status": summary_report["regression.status"],
                "suite_status": suite_status,
                "report_path": artifact_paths["phase_c_gate_report.md"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary_report["regression.status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
