from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import _bootstrap  # noqa: F401
from iris.regression import GateContext, Tolerances, run_phase_d_gate


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


def _suite_status_from_command(record: Mapping[str, Any], suite_name: str) -> tuple[str, str]:
    if record.get("status") == "PASS":
        return "PASS", ""
    stderr = str(record.get("stderr", "")).strip()
    reason = stderr if stderr else f"{suite_name} command failed (rc={record.get('returncode')})"
    return "FAIL", reason


def _load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_suite_payload(
    *,
    record: Mapping[str, Any],
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
    parser = argparse.ArgumentParser(description="Generate Phase D gate artifacts and report.")
    parser.add_argument("--phase", type=str, default="D")
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-report-dir",
        type=Path,
        default=Path("artifacts/baselines/phase-d-v1"),
        help="Directory containing frozen phase-d-v1 baseline artifacts.",
    )
    parser.add_argument(
        "--freeze-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write current S3/S4/S5 artifacts into baseline-report-dir.",
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
        default=Path("artifacts/phase_d_gate_20260302/report"),
        help="Directory where gate artifacts and markdown report are written.",
    )
    parser.add_argument(
        "--document-fixtures",
        type=Path,
        default=Path("tests/fixtures/p1_phase_de/document_eval"),
        help="Math-native document eval fixture root.",
    )
    parser.add_argument(
        "--proof-fixtures",
        type=Path,
        default=Path("tests/fixtures/p1_phase_de/proof_eval"),
        help="Math-native proof eval fixture root.",
    )
    parser.add_argument(
        "--conceptarc-corpus",
        type=Path,
        default=None,
        help="Optional ConceptARC compatibility appendix path.",
    )
    parser.add_argument(
        "--rearc-tasks",
        type=Path,
        default=None,
        help="Optional re_arc compatibility appendix path.",
    )
    parser.add_argument(
        "--pairing-policy",
        type=str,
        default="adjacent",
        choices=["adjacent"],
    )
    parser.add_argument("--baseline-id", type=str, default="phase-d-v1")
    parser.add_argument("--tolerance-profile-id", type=str, default="phase-d-default")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--metric-epsilon", type=float, default=1e-6)
    parser.add_argument("--kl-epsilon", type=float, default=1e-6)
    parser.add_argument("--credit-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-reasoning-cycles", type=int, default=3)
    parser.add_argument("--termination-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--h100-uninterrupted", type=Path, default=Path("artifacts/toy_train_gpu_h100"))
    parser.add_argument("--h100-execute", type=Path, default=Path("artifacts/toy_resume_h100"))
    parser.add_argument("--h100-pre-commit", type=Path, default=Path("artifacts/toy_resume_h100_pre_commit"))
    parser.add_argument("--h100-post-commit", type=Path, default=Path("artifacts/toy_resume_h100_post_commit"))
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

    if str(args.phase).strip().upper() != "D":
        raise SystemExit("Phase D gate accepts only --phase D.")

    context = GateContext(
        phase="D",
        baseline_id=args.baseline_id,
        tolerance_profile_id=args.tolerance_profile_id,
        change_class="Capability expansion (Phase D math-native document-grounded diagnostics)",
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

    s1_record = _run_json_suite([sys.executable, "scripts/s1_smoke.py", "--device", args.device])
    s2_record = _run_json_suite([sys.executable, "scripts/s2_structural.py"])
    s2_mounted_record = _run_json_suite([sys.executable, "scripts/s2_mounted.py", "--device", args.device])

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
        suite_name
        for suite_name, payload in (
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

    result = run_phase_d_gate(
        context=context,
        tolerances=tolerances,
        output_dir=args.output_dir,
        model_run_dir=args.model_run_dir,
        conceptarc_corpus=args.conceptarc_corpus,
        rearc_tasks=args.rearc_tasks,
        baseline_report_dir=args.baseline_report_dir,
        phase_root=args.phase_root,
        h100_path_map={
            "uninterrupted": args.h100_uninterrupted,
            "execute_crash": args.h100_execute,
            "pre_commit_crash": args.h100_pre_commit,
            "post_commit_crash": args.h100_post_commit,
        },
        s1_status=s1_status,
        s1_reasons=s1_reasons,
        s2_status=s2_status,
        s2_reasons=s2_reasons,
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2_mounted_output,
        max_reasoning_cycles=args.max_reasoning_cycles,
        termination_threshold=args.termination_threshold,
        seed=args.seed,
        pairing_policy=args.pairing_policy,
        freeze_baseline=args.freeze_baseline,
        document_fixture_root=args.document_fixtures,
        proof_fixture_root=args.proof_fixtures,
    )

    summary = dict(result.get("summary_report", {}))
    suite_status = dict(result.get("suite_status", {}))
    report_path = str((args.output_dir / "phase_d_gate_report.md").resolve())
    print(
        json.dumps(
            {
                "status": summary.get("regression.status", "FAIL"),
                "suite_status": suite_status,
                "report_path": report_path,
            },
            sort_keys=True,
        )
    )
    return 0 if str(summary.get("regression.status", "FAIL")).upper() == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
