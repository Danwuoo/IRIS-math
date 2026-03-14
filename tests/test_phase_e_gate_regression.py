from __future__ import annotations

import json
import shutil
from pathlib import Path

from iris.regression import GateContext, Tolerances, run_phase_e_gate
from iris.regression.local_closure_bootstrap import seed_local_model_run, seed_resume_packet_root


def _suite_payload() -> dict[str, str]:
    return {"status": "PASS"}


def _context() -> GateContext:
    return GateContext(
        phase="E",
        baseline_id="phase-e-math-native",
        tolerance_profile_id="phase-e-math-native",
        change_class="Capability expansion (Phase E strict held-out proof/verifier packet)",
    )


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


def test_phase_e_gate_passes_with_strict_heldout_packet(tmp_path: Path) -> None:
    model_run_dir = seed_local_model_run(tmp_path / "model_run")
    phase_root = tmp_path / "phase_root"
    seed_resume_packet_root(phase_root, phase="E")
    h100_path_map = seed_resume_packet_root(tmp_path / "h100", phase="E")
    baseline_dir = tmp_path / "baseline"

    run_phase_e_gate(
        context=_context(),
        tolerances=_tolerances(),
        output_dir=tmp_path / "report_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_dir,
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
    result = run_phase_e_gate(
        context=_context(),
        tolerances=_tolerances(),
        output_dir=tmp_path / "report",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_dir,
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

    assert result["summary_report"]["regression.status"] == "PASS"
    heldout_packet = json.loads((tmp_path / "report" / "phase_e_heldout_packet.json").read_text(encoding="utf-8"))
    assert heldout_packet["status"] == "PASS"


def test_phase_e_gate_fails_when_formalization_policy_expectation_breaks(tmp_path: Path) -> None:
    fixture_root = tmp_path / "tests" / "fixtures"
    shutil.copytree(Path("tests/fixtures/p1_phase_de"), fixture_root / "p1_phase_de")
    shutil.copytree(Path("tests/fixtures/p1_phase_c"), fixture_root / "p1_phase_c")
    proof_path = fixture_root / "p1_phase_de" / "proof_eval" / "heldout_formalization_bridge.json"
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    payload["expected_task_adjudication_policy_id"] = "wrong-policy-id"
    proof_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    model_run_dir = seed_local_model_run(tmp_path / "model_run")
    phase_root = tmp_path / "phase_root"
    seed_resume_packet_root(phase_root, phase="E")
    h100_path_map = seed_resume_packet_root(tmp_path / "h100", phase="E")
    baseline_dir = tmp_path / "baseline"

    run_phase_e_gate(
        context=_context(),
        tolerances=_tolerances(),
        output_dir=tmp_path / "report_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_dir,
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
    result = run_phase_e_gate(
        context=_context(),
        tolerances=_tolerances(),
        output_dir=tmp_path / "report",
        model_run_dir=model_run_dir,
        conceptarc_corpus=None,
        rearc_tasks=None,
        baseline_report_dir=baseline_dir,
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
        document_fixture_root=fixture_root / "p1_phase_de" / "document_eval",
        proof_fixture_root=fixture_root / "p1_phase_de" / "proof_eval",
    )

    assert result["summary_report"]["regression.status"] == "FAIL"
    assert result["suite_status"]["PhaseEHeldout"] == "FAIL"
