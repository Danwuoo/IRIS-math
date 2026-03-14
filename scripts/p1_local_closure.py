from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
    run_phase_d_gate,
    run_phase_e_gate,
    utc_now_iso,
    write_phase_c_gate_artifacts,
)
from iris.regression.local_closure_bootstrap import seed_local_model_run, seed_resume_packet_root


def _run_json_suite(command: List[str]) -> Dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    payload: Dict[str, Any] = {}
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = {}
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "payload": payload,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")


def _phase_c_local_check(
    *,
    output_dir: Path,
    phase_root: Path,
    h100_path_map: Mapping[str, Path],
    s1_output: Mapping[str, Any],
    s2_output: Mapping[str, Any],
    s2_mounted_output: Mapping[str, Any],
) -> Dict[str, Any]:
    context = GateContext(phase="C", baseline_id="p1-local-closure", tolerance_profile_id="p1-local-closure")
    tolerances = Tolerances(
        metric_epsilon=1e-6,
        failure_profile_kl_epsilon=1e-6,
        failure_credit_delta_epsilon=1e-6,
        concept_isolation_delta_epsilon=1e-6,
        concept_leakage_delta_epsilon=1e-6,
        paired_asymmetry_delta_epsilon=1e-6,
        paired_invariance_gap_delta_epsilon=1e-6,
        s8_metric_delta_epsilon=1e-6,
    )
    concept_breakdown = build_concept_breakdown(Path("tests/fixtures/p1_phase_c/regression/concepts"), context)
    paired_diff = build_paired_representation_diff(Path("tests/fixtures/p1_phase_c/regression/pairs"), context)
    resume_packet = build_resume_consistency_packet(seed_resume_packet_root(phase_root, phase="C"), context)
    failure_profile_diff = build_failure_profile_diff(resume_packet, context)
    credit_routing_diff = build_credit_routing_diff(resume_packet, context)
    h100_packet = build_h100_packet_summary(h100_path_map, context, tolerances=tolerances)
    s3_status, s3_reasons = evaluate_s3_status(failure_profile_diff, tolerances)
    s4_status, s4_reasons, s4_details = evaluate_s4_status(
        current_concept_breakdown=concept_breakdown,
        baseline_concept_breakdown=concept_breakdown,
        tolerances=tolerances,
        phase="C",
    )
    s5_status, s5_reasons, s5_details = evaluate_s5_status(
        current_paired_diff=paired_diff,
        baseline_paired_diff=paired_diff,
        tolerances=tolerances,
        phase="C",
    )
    s6_status, s6_reasons = evaluate_s6_status(resume_packet, credit_routing_diff, tolerances)
    s7_status, s7_reasons = evaluate_s7_status(resume_packet, tolerances)
    s8_status, s8_reasons = evaluate_s8_status(resume_packet, tolerances)
    suite_status = {
        "S1": str(s1_output.get("status", "FAIL")),
        "S2": "PASS" if str(s2_output.get("status", "FAIL")) == "PASS" and str(s2_mounted_output.get("status", "FAIL")) == "PASS" else "FAIL",
        "S3": s3_status,
        "S4": s4_status,
        "S5": s5_status,
        "S6": s6_status,
        "S7": s7_status,
        "S8": s8_status,
        "S8_h100_packet": str(h100_packet.get("s8_status_for_h100_packet", "FAIL")),
    }
    violations: List[Dict[str, Any]] = []
    for suite_name, reasons, metric, details in (
        ("S3", s3_reasons, "failure profile KL divergence", None),
        ("S4", s4_reasons, "concept.isolation_score / concept.leakage_score", s4_details),
        ("S5", s5_reasons, "paired.asymmetry_rate / paired.invariance.gap", s5_details),
        ("S6", s6_reasons, "failure.credit stability", None),
        ("S7", s7_reasons, "pretraining diagnostics deltas", None),
        ("S8", s8_reasons, "resume consistency drift / coverage", None),
    ):
        if suite_status[suite_name] != "PASS":
            violations.append({"suite": suite_name, "reason": "; ".join(reasons), "metric": metric, "phase": "C", "details": details})
    summary_report = build_summary_report(
        context=context,
        suite_status=suite_status,
        violations=violations,
        generated_at_utc=utc_now_iso(),
        notes=["Local Phase C closure uses self-baselined checked-in fixture packet."],
        evidence_paths=[str(output_dir / "summary_report.json")],
    )
    artifacts = write_phase_c_gate_artifacts(
        output_dir=output_dir,
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
    return {"summary_report": summary_report, "artifact_paths": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local P1 A-E closure using checked-in fixtures.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/p1_local_closure"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_run_dir = seed_local_model_run(output_dir / "work" / "model_run")
    phase_root = output_dir / "work" / "phase_root"
    h100_path_map = seed_resume_packet_root(output_dir / "work" / "h100_packet", phase="E")

    s1_record = _run_json_suite([sys.executable, "scripts/s1_smoke.py", "--device", args.device])
    s2_record = _run_json_suite([sys.executable, "scripts/s2_structural.py"])
    s2m_record = _run_json_suite([sys.executable, "scripts/s2_mounted.py", "--device", args.device])
    s1_output = dict(s1_record.get("payload") or {"status": "FAIL"})
    s2_output = dict(s2_record.get("payload") or {"status": "FAIL"})
    s2m_output = dict(s2m_record.get("payload") or {"status": "FAIL"})

    phase_c = _phase_c_local_check(
        output_dir=output_dir / "phase_c",
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2m_output,
    )

    baseline_phase_d = output_dir / "baselines" / "phase_d"
    context_d = GateContext(phase="D", baseline_id="p1-local-phase-d", tolerance_profile_id="p1-local-closure")
    tolerances = Tolerances(
        metric_epsilon=1e-6,
        failure_profile_kl_epsilon=1e-6,
        failure_credit_delta_epsilon=1e-6,
        concept_isolation_delta_epsilon=1e-6,
        concept_leakage_delta_epsilon=1e-6,
        paired_asymmetry_delta_epsilon=1e-6,
        paired_invariance_gap_delta_epsilon=1e-6,
        s8_metric_delta_epsilon=1e-6,
    )
    run_phase_d_gate(
        context=context_d,
        tolerances=tolerances,
        output_dir=output_dir / "phase_d_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_phase_d,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status=str(s1_output.get("status", "FAIL")),
        s1_reasons=[],
        s2_status="PASS" if str(s2_output.get("status", "FAIL")) == "PASS" and str(s2m_output.get("status", "FAIL")) == "PASS" else "FAIL",
        s2_reasons=[],
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2m_output,
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=args.seed,
        freeze_baseline=True,
    )
    phase_d = run_phase_d_gate(
        context=context_d,
        tolerances=tolerances,
        output_dir=output_dir / "phase_d",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_phase_d,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status=str(s1_output.get("status", "FAIL")),
        s1_reasons=[],
        s2_status="PASS" if str(s2_output.get("status", "FAIL")) == "PASS" and str(s2m_output.get("status", "FAIL")) == "PASS" else "FAIL",
        s2_reasons=[],
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2m_output,
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=args.seed,
    )

    baseline_phase_e = output_dir / "baselines" / "phase_e"
    context_e = GateContext(phase="E", baseline_id="p1-local-phase-e", tolerance_profile_id="p1-local-closure")
    run_phase_e_gate(
        context=context_e,
        tolerances=tolerances,
        output_dir=output_dir / "phase_e_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_phase_e,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status=str(s1_output.get("status", "FAIL")),
        s1_reasons=[],
        s2_status="PASS" if str(s2_output.get("status", "FAIL")) == "PASS" and str(s2m_output.get("status", "FAIL")) == "PASS" else "FAIL",
        s2_reasons=[],
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2m_output,
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=args.seed,
        arc_benchmark_probe_path=None,
        freeze_baseline=True,
    )
    phase_e = run_phase_e_gate(
        context=context_e,
        tolerances=tolerances,
        output_dir=output_dir / "phase_e",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_phase_e,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status=str(s1_output.get("status", "FAIL")),
        s1_reasons=[],
        s2_status="PASS" if str(s2_output.get("status", "FAIL")) == "PASS" and str(s2m_output.get("status", "FAIL")) == "PASS" else "FAIL",
        s2_reasons=[],
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2m_output,
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=args.seed,
        arc_benchmark_probe_path=None,
    )

    closure_status = "Done" if all(
        str(result.get("summary_report", {}).get("regression.status", "FAIL")).upper() == "PASS"
        for result in (phase_c, phase_d, phase_e)
    ) else "Blocked"
    summary_payload = {
        "schema": "iris.local_closure_summary/v1",
        "change_class": "Capability expansion",
        "phase_status": {
            "C": phase_c["summary_report"]["regression.status"],
            "D": phase_d["summary_report"]["regression.status"],
            "E": phase_e["summary_report"]["regression.status"],
        },
        "artifact_root": str(output_dir.resolve()),
        "termination": closure_status,
    }
    _write_json(output_dir / "closure_summary.json", summary_payload)
    (output_dir / "closure_summary.md").write_text(
        "\n".join(
            [
                "# P1 Local Closure",
                "",
                f"- Phase C: {phase_c['summary_report']['regression.status']}",
                f"- Phase D: {phase_d['summary_report']['regression.status']}",
                f"- Phase E: {phase_e['summary_report']['regression.status']}",
                f"- Termination: {closure_status}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": closure_status, "output_dir": str(output_dir.resolve())}, sort_keys=True))
    return 0 if closure_status == "Done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
