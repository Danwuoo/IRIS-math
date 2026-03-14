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
    proof_eval_packet,
    read_json,
    write_json,
)
from .phase_c_gate import GateContext, Tolerances
from .phase_d_gate import run_phase_d_gate


def _build_phase_e_report(
    *,
    summary_report: Mapping[str, Any],
    heldout_packet: Mapping[str, Any],
    compatibility_arc_appendix: Mapping[str, Any],
) -> str:
    lines = [
        "# Phase E Gate Report",
        "",
        "- Primary Surface: strict held-out document + proof packet",
        "- ARC Posture: compatibility-only appendix, non-blocking",
        f"- Regression Status: **{summary_report.get('regression.status', 'FAIL')}**",
        "",
        "## Suite Status",
    ]
    for suite_name, status in sorted(dict(summary_report.get("suite_status", {})).items()):
        lines.append(f"- {suite_name}: {status}")
    lines.extend(
        [
            "",
            "## Held-out Packet",
            f"- Document fixtures: {heldout_packet.get('document_fixture_count', 0)}",
            f"- Proof fixtures: {heldout_packet.get('proof_fixture_count', 0)}",
            f"- Status: {heldout_packet.get('status', 'FAIL')}",
            "",
            "## Compatibility ARC Appendix",
            f"- Status: {compatibility_arc_appendix.get('status', 'SKIPPED')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _heldout_packet(
    *,
    context: GateContext,
    document_results: Sequence[Mapping[str, Any]],
    proof_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    failures: List[str] = []
    for row in list(document_results) + list(proof_results):
        if str(row.get("status", "FAIL")) != "PASS":
            failures.append(str(row.get("fixture_id", "unknown")))
    return {
        "schema": "iris.regression.phase_e_heldout_packet/v1",
        "phase": context.phase,
        "document_fixture_count": len(document_results),
        "proof_fixture_count": len(proof_results),
        "status": "PASS" if not failures and document_results and proof_results else "FAIL",
        "document_packet": document_eval_packet(
            document_results=document_results,
            context=context,
            eval_partition="strict_holdout",
        ),
        "proof_packet": proof_eval_packet(
            proof_results=proof_results,
            context=context,
            eval_partition="strict_holdout",
        ),
        "failures": failures,
    }


def run_phase_e_gate(
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
    arc_benchmark_probe_path: Path | None,
    pairing_policy: str = "adjacent",
    freeze_baseline: bool = False,
    hard_fail: bool = True,
    document_fixture_root: Path | None = None,
    proof_fixture_root: Path | None = None,
) -> Dict[str, Any]:
    del hard_fail
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
        document_fixture_root=document_fixture_root,
        proof_fixture_root=proof_fixture_root,
    )
    policy_bundle = load_policy_bundle_for_profile_phase("P1", context.phase)
    heldout_document_fixtures = load_document_eval_fixtures(document_fixture_root or default_document_fixture_root(), eval_partition="strict_holdout")
    heldout_proof_fixtures = load_proof_eval_fixtures(proof_fixture_root or default_proof_fixture_root(), eval_partition="strict_holdout")
    document_results = []
    document_bundles: Dict[str, Any] = {}
    for index, fixture in enumerate(heldout_document_fixtures):
        result, document_bundle = evaluate_document_fixture(fixture, policy_bundle=policy_bundle, hidden_dim=16, seed=seed + 500 + index)
        document_results.append(result)
        document_bundles[fixture.fixture_id] = document_bundle
    proof_results = [
        evaluate_proof_fixture(fixture, policy_bundle=policy_bundle, document_bundles=document_bundles, hidden_dim=16, seed=seed + 700 + index)
        for index, fixture in enumerate(heldout_proof_fixtures)
    ]
    heldout_packet = _heldout_packet(context=context, document_results=document_results, proof_results=proof_results)
    compatibility_arc_appendix = dict(base_result.get("compatibility_arc_appendix", {}))
    if arc_benchmark_probe_path is not None and Path(arc_benchmark_probe_path).exists():
        compatibility_arc_appendix["arc_benchmark_probe"] = read_json(Path(arc_benchmark_probe_path))
        compatibility_arc_appendix["status"] = "AVAILABLE"
    summary_report = dict(base_result.get("summary_report", {}))
    suite_status = dict(base_result.get("suite_status", {}))
    suite_status["PhaseEHeldout"] = str(heldout_packet.get("status", "FAIL"))
    violations = list(summary_report.get("regression.violations", []))
    if heldout_packet.get("status") != "PASS":
        violations.append(
            {
                "suite": "PhaseEHeldout",
                "reason": "held-out packet contains document/proof fixture failures",
                "metric": "task family / adjudication / verifier calibration packet",
                "phase": context.phase,
                "details": list(heldout_packet.get("failures", [])),
            }
        )
    summary_report["suite_status"] = suite_status
    summary_report["regression.violations"] = violations
    summary_report["notes"] = list(summary_report.get("notes", [])) + [
        f"phase_e_heldout_document_fixture_count={len(heldout_document_fixtures)}",
        f"phase_e_heldout_proof_fixture_count={len(heldout_proof_fixtures)}",
        "formalization registry coverage retained under P1 partial_mount.",
        SIDECAR_TECH_DEBT_NOTE,
    ]
    summary_report["regression.status"] = "PASS" if all(str(status).upper() == "PASS" for status in suite_status.values()) and not violations else "FAIL"
    checklist = dict(summary_report.get("completion_checklist", {}))
    checklist["expected_failure_category_impact"] = (
        "Improves strict held-out visibility for F_REP, F_PROC, F_SEARCH, F_EVAL with verifier-calibration emphasis."
    )
    checklist["technical_debt_guardrails_introduced"] = SIDECAR_TECH_DEBT_NOTE
    checklist["termination"] = "Done" if summary_report["regression.status"] == "PASS" else "Blocked"
    summary_report["completion_checklist"] = checklist
    write_json(output_dir / "summary_report.json", summary_report)
    write_json(output_dir / "phase_e_heldout_packet.json", heldout_packet)
    write_json(output_dir / "compatibility_arc_appendix.json", compatibility_arc_appendix)
    (output_dir / "phase_e_gate_report.md").write_text(
        _build_phase_e_report(
            summary_report=summary_report,
            heldout_packet=heldout_packet,
            compatibility_arc_appendix=compatibility_arc_appendix,
        ),
        encoding="utf-8",
    )
    return {
        "summary_report": summary_report,
        "suite_status": suite_status,
        "artifact_paths": {
            **dict(base_result.get("artifact_paths", {})),
            "phase_e_heldout_packet.json": str(output_dir / "phase_e_heldout_packet.json"),
            "phase_e_gate_report.md": str(output_dir / "phase_e_gate_report.md"),
            "compatibility_arc_appendix.json": str(output_dir / "compatibility_arc_appendix.json"),
        },
    }
