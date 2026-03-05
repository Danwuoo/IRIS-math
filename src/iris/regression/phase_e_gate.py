from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .phase_c_gate import GateContext, Tolerances, utc_now_iso
from .phase_d_gate import run_phase_d_gate


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return dict(payload)
    return {"data": payload}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")


def _phase_e_probe_check(probe: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if str(probe.get("status", "FAIL")).upper() != "PASS":
        reasons.append("arc_benchmark_probe.status != PASS")
    if not bool(probe.get("baseline_non_regression", False)):
        reasons.append("arc benchmark baseline_non_regression is false")

    baseline_results_path = Path(
        str(
            ((probe.get("probe_a_baseline") or {}).get("scoring") or {}).get("results_json_path", "")
        )
    )
    iris_results_path = Path(
        str(
            ((probe.get("probe_b_iris") or {}).get("scoring") or {}).get("results_json_path", "")
        )
    )
    if not baseline_results_path.exists():
        reasons.append("baseline scoring artifact is missing")
    if not iris_results_path.exists():
        reasons.append("iris scoring artifact is missing")

    return ("PASS" if not reasons else "FAIL", reasons)


def _build_phase_e_report(
    *,
    summary_report: Mapping[str, Any],
    phase_e_probe_status: str,
    phase_e_probe_reasons: Sequence[str],
) -> str:
    lines = []
    lines.append("# Phase E Gate Report")
    lines.append("")
    lines.append("- Document Type: Design Note (Non-normative)")
    lines.append(f"- Generated At (UTC): {summary_report.get('generated_at_utc', utc_now_iso())}")
    lines.append(f"- Phase: {summary_report.get('phase', '')}")
    lines.append(f"- Baseline ID: {summary_report.get('baseline_id', '')}")
    lines.append(f"- Tolerance Profile ID: {summary_report.get('tolerance_profile_id', '')}")
    lines.append(f"- Change Class: {summary_report.get('change_class', '')}")
    lines.append(f"- Regression Status: **{summary_report.get('regression.status', 'FAIL')}**")
    lines.append("")
    lines.append("## 1) Suite Status")
    for suite_name, status in sorted(dict(summary_report.get("suite_status", {})).items()):
        lines.append(f"- {suite_name}: {status}")
    lines.append("")
    lines.append("## 2) Phase E Probe")
    lines.append(f"- PhaseEProbe: {phase_e_probe_status}")
    if phase_e_probe_reasons:
        for reason in phase_e_probe_reasons:
            lines.append(f"- Block reason: {reason}")
    lines.append("")
    lines.append("## 3) Violations")
    violations = summary_report.get("regression.violations", [])
    if isinstance(violations, list) and violations:
        for row in violations:
            lines.append(
                f"- [{row.get('suite', 'unknown')}] {row.get('metric', '')}: {row.get('reason', '')}"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## 4) Notes")
    for note in summary_report.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def run_phase_e_gate(
    *,
    context: GateContext,
    tolerances: Tolerances,
    output_dir: Path,
    model_run_dir: Path,
    conceptarc_corpus: Path,
    rearc_tasks: Path,
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
    arc_benchmark_probe_path: Path,
    pairing_policy: str = "adjacent",
    freeze_baseline: bool = False,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_result = run_phase_d_gate(
        context=context,
        tolerances=tolerances,
        output_dir=output_dir,
        model_run_dir=model_run_dir,
        conceptarc_corpus=conceptarc_corpus,
        rearc_tasks=rearc_tasks,
        baseline_report_dir=baseline_report_dir,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status=s1_status,
        s1_reasons=s1_reasons,
        s2_status=s2_status,
        s2_reasons=s2_reasons,
        s1_output=s1_output,
        s2_output=s2_output,
        s2_mounted_output=s2_mounted_output,
        max_reasoning_cycles=max_reasoning_cycles,
        termination_threshold=termination_threshold,
        seed=seed,
        pairing_policy=pairing_policy,
        freeze_baseline=freeze_baseline,
    )

    summary_report = dict(base_result.get("summary_report", {}))
    suite_status = dict(base_result.get("suite_status", {}))
    violations = list(summary_report.get("regression.violations", []))

    probe_path = Path(arc_benchmark_probe_path)
    if not probe_path.exists():
        phase_e_probe = {
            "status": "FAIL",
            "baseline_non_regression": False,
            "block_reasons": [f"arc_benchmark_probe artifact missing: {probe_path}"],
        }
    else:
        phase_e_probe = _read_json(probe_path)

    phase_e_probe_status, phase_e_probe_reasons = _phase_e_probe_check(phase_e_probe)
    suite_status["PhaseEProbe"] = phase_e_probe_status

    notes = list(summary_report.get("notes", []))
    notes.append(f"phase_e_probe.status={phase_e_probe_status}")

    if phase_e_probe_status != "PASS":
        violations.append(
            {
                "suite": "PhaseEProbe",
                "metric": "arc benchmark probe runtime/scoring/baseline checks",
                "phase": context.phase,
                "suspected_level": "Regression harness / verifier bridge",
                "reason": "; ".join(phase_e_probe_reasons),
            }
        )

    all_suites_pass = all(str(status).upper() == "PASS" for status in suite_status.values())
    summary_report["suite_status"] = suite_status
    summary_report["regression.violations"] = violations
    summary_report["notes"] = notes
    summary_report["regression.status"] = "PASS" if all_suites_pass and not violations else "FAIL"

    checklist = dict(summary_report.get("completion_checklist", {}))
    checklist["expected_failure_category_impact"] = (
        "F_REP, F_PROC, F_SEARCH, F_EVAL visibility uplift plus benchmark verifier bridge coverage."
    )
    checklist["termination"] = "Done" if summary_report["regression.status"] == "PASS" else "Blocked"
    summary_report["completion_checklist"] = checklist

    _write_json(output_dir / "summary_report.json", summary_report)
    _write_json(output_dir / "phase_e_probe.json", phase_e_probe)
    (output_dir / "phase_e_gate_report.md").write_text(
        _build_phase_e_report(
            summary_report=summary_report,
            phase_e_probe_status=phase_e_probe_status,
            phase_e_probe_reasons=phase_e_probe_reasons,
        ),
        encoding="utf-8",
    )

    if hard_fail and summary_report["regression.status"] != "PASS":
        summary_report["termination"] = "Blocked"

    return {
        "summary_report": summary_report,
        "suite_status": suite_status,
        "artifact_paths": {
            **dict(base_result.get("artifact_paths", {})),
            "phase_e_probe.json": str(output_dir / "phase_e_probe.json"),
            "phase_e_gate_report.md": str(output_dir / "phase_e_gate_report.md"),
        },
    }
