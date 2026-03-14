from __future__ import annotations

import json
from pathlib import Path

from iris.regression import GateContext, Tolerances, run_phase_d_gate, run_phase_e_gate
from iris.regression.local_closure_bootstrap import seed_local_model_run, seed_resume_packet_root
from iris.regression.p1_readiness import build_p1_readiness_snapshot, evaluate_p1_readiness


def _suite_payload() -> dict[str, str]:
    return {"status": "PASS"}


def _tolerances() -> Tolerances:
    return Tolerances(
        metric_epsilon=1e-6,
        failure_profile_kl_epsilon=1e-6,
        failure_credit_delta_epsilon=1e-6,
        concept_isolation_delta_epsilon=1e-6,
        concept_leakage_delta_epsilon=1e-6,
        paired_asymmetry_delta_epsilon=1e-6,
        paired_invariance_gap_delta_epsilon=1e-6,
        s8_metric_delta_epsilon=1e-6,
    )


def _build_phase_de_reports(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_run_dir = seed_local_model_run(tmp_path / "model_run")
    phase_root = tmp_path / "phase_root"
    seed_resume_packet_root(phase_root, phase="E")
    h100_path_map = seed_resume_packet_root(tmp_path / "h100", phase="E")
    baseline_d = tmp_path / "baseline_d"
    baseline_e = tmp_path / "baseline_e"

    context_d = GateContext(
        phase="D",
        baseline_id="p1-readiness-phase-d",
        tolerance_profile_id="tp_p1_bootstrap",
    )
    context_e = GateContext(
        phase="E",
        baseline_id="p1-readiness-phase-e",
        tolerance_profile_id="tp_p1_bootstrap",
    )

    run_phase_d_gate(
        context=context_d,
        tolerances=_tolerances(),
        output_dir=tmp_path / "phase_d_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_d,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status="PASS",
        s1_reasons=[],
        s2_status="PASS",
        s2_reasons=[],
        s1_output=_suite_payload(),
        s2_output=_suite_payload(),
        s2_mounted_output=_suite_payload(),
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
        freeze_baseline=True,
    )
    run_phase_d_gate(
        context=context_d,
        tolerances=_tolerances(),
        output_dir=tmp_path / "phase_d",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_d,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status="PASS",
        s1_reasons=[],
        s2_status="PASS",
        s2_reasons=[],
        s1_output=_suite_payload(),
        s2_output=_suite_payload(),
        s2_mounted_output=_suite_payload(),
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
    )

    run_phase_e_gate(
        context=context_e,
        tolerances=_tolerances(),
        output_dir=tmp_path / "phase_e_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_e,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status="PASS",
        s1_reasons=[],
        s2_status="PASS",
        s2_reasons=[],
        s1_output=_suite_payload(),
        s2_output=_suite_payload(),
        s2_mounted_output=_suite_payload(),
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
        arc_benchmark_probe_path=None,
        freeze_baseline=True,
    )
    run_phase_e_gate(
        context=context_e,
        tolerances=_tolerances(),
        output_dir=tmp_path / "phase_e",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_e,
        phase_root=phase_root,
        h100_path_map=h100_path_map,
        s1_status="PASS",
        s1_reasons=[],
        s2_status="PASS",
        s2_reasons=[],
        s1_output=_suite_payload(),
        s2_output=_suite_payload(),
        s2_mounted_output=_suite_payload(),
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
        arc_benchmark_probe_path=None,
    )
    return tmp_path / "phase_d", tmp_path / "phase_e", model_run_dir


def _make_snapshot(
    *,
    baseline_id: str,
    tolerance_profile_id: str,
    grounding: float = 0.952,
) -> dict[str, object]:
    values = {
        "rep.document.parse_completeness": 0.976,
        "task.document_grounding_score": grounding,
        "failure.credit.collapse_rate": 0.018,
        "eval.false_accept_rate": 0.031,
        "eval.calibration_error": 0.071,
        "contam.strict_holdout_leakage_score": 0.0055,
        "provenance.parser_coverage": 0.97,
        "provenance.verifier_coverage": 0.95,
    }
    return {
        "schema": "iris.readiness.p1_snapshot/v1",
        "generated_at_utc": "2026-03-14T00:00:00Z",
        "profile_id": "P1",
        "baseline_id": baseline_id,
        "tolerance_profile_id": tolerance_profile_id,
        "hard_gate_surface_values": {
            metric_name: {"value": metric_value, "source_artifact": f"{metric_name}.json"}
            for metric_name, metric_value in values.items()
        },
        "residual_blockers": [],
    }


def test_phase_e_heldout_packet_retains_readiness_coverage_summary(tmp_path: Path) -> None:
    _, phase_e_dir, _ = _build_phase_de_reports(tmp_path)
    heldout_packet = json.loads((phase_e_dir / "phase_e_heldout_packet.json").read_text(encoding="utf-8"))

    assert heldout_packet["document_packet"]["aggregate"]["provenance.parser_coverage.mean"] > 0.0
    assert heldout_packet["proof_packet"]["aggregate"]["provenance.verifier_coverage.mean"] == 1.0
    assert (
        heldout_packet["proof_packet"]["coverage_summary"]["task_adjudication_policy_resolution_coverage"]
        == 1.0
    )


def test_p1_readiness_packet_blocks_fixture_backed_local_closure(tmp_path: Path) -> None:
    phase_d_dir, phase_e_dir, model_run_dir = _build_phase_de_reports(tmp_path)
    snapshot = build_p1_readiness_snapshot(
        phase_d_root=phase_d_dir,
        phase_e_root=phase_e_dir,
        model_run_dir=model_run_dir,
        leakage_audit_path=None,
        baseline_id="p1-fixed-baseline",
        tolerance_profile_id="tp_p1_bootstrap",
    )
    packet = evaluate_p1_readiness(
        snapshot=snapshot,
        baseline_snapshot=snapshot,
        history_rows=[],
    )

    assert packet["run_gate_status"] == "FAIL"
    assert "rep.document.parse_completeness" in packet["surface_failures"]
    assert "task.document_grounding_score" in packet["surface_failures"]
    assert "failure.credit.collapse_rate" in packet["surface_failures"]
    assert "contam.strict_holdout_leakage_score" in packet["surface_failures"]
    blockers = {item["blocker"] for item in packet["residual_blockers"]}
    assert "governed_training_run" in blockers
    assert "sidecar_document_pipeline_debt" in blockers


def test_p1_readiness_promotion_requires_three_consecutive_gate_passes() -> None:
    baseline_id = "p1-fixed-baseline"
    tolerance_profile_id = "tp_p1_bootstrap"
    baseline_snapshot = _make_snapshot(
        baseline_id=baseline_id,
        tolerance_profile_id=tolerance_profile_id,
        grounding=0.94,
    )
    current_snapshot = _make_snapshot(
        baseline_id=baseline_id,
        tolerance_profile_id=tolerance_profile_id,
        grounding=0.952,
    )

    packet1 = evaluate_p1_readiness(
        snapshot=current_snapshot,
        baseline_snapshot=baseline_snapshot,
        history_rows=[],
    )
    history = [
        {
            "baseline_id": packet1["baseline_id"],
            "tolerance_profile_id": packet1["tolerance_profile_id"],
            "run_gate_status": packet1["run_gate_status"],
        }
    ]
    packet2 = evaluate_p1_readiness(
        snapshot=current_snapshot,
        baseline_snapshot=baseline_snapshot,
        history_rows=history,
    )
    history.append(
        {
            "baseline_id": packet2["baseline_id"],
            "tolerance_profile_id": packet2["tolerance_profile_id"],
            "run_gate_status": packet2["run_gate_status"],
        }
    )
    packet3 = evaluate_p1_readiness(
        snapshot=current_snapshot,
        baseline_snapshot=baseline_snapshot,
        history_rows=history,
    )

    assert packet1["run_gate_status"] == "PASS"
    assert packet1["promotion_status"] == "BLOCKED"
    assert packet1["consecutive_gate_passed_runs"] == 1
    assert packet2["promotion_status"] == "BLOCKED"
    assert packet2["consecutive_gate_passed_runs"] == 2
    assert packet3["promotion_status"] == "PASS"
    assert packet3["consecutive_gate_passed_runs"] == 3
