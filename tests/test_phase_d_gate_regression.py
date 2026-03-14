from __future__ import annotations

import json
import shutil
from pathlib import Path

from iris.regression import GateContext, Tolerances, run_phase_d_gate
from iris.regression.local_closure_bootstrap import seed_local_model_run, seed_resume_packet_root


def _suite_payload() -> dict[str, str]:
    return {"status": "PASS"}


def _context() -> GateContext:
    return GateContext(
        phase="D",
        baseline_id="phase-d-math-native",
        tolerance_profile_id="phase-d-math-native",
        change_class="Capability expansion (Phase D math-native document-grounded diagnostics)",
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


def test_phase_d_gate_passes_with_math_native_fixtures(tmp_path: Path) -> None:
    model_run_dir = seed_local_model_run(tmp_path / "model_run")
    phase_root = tmp_path / "phase_root"
    seed_resume_packet_root(phase_root, phase="D")
    h100_path_map = seed_resume_packet_root(tmp_path / "h100", phase="D")
    baseline_dir = tmp_path / "baseline"

    run_phase_d_gate(
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
        freeze_baseline=True,
    )
    result = run_phase_d_gate(
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
    )

    assert result["summary_report"]["regression.status"] == "PASS"
    concept_breakdown = json.loads((tmp_path / "report" / "concept_breakdown.json").read_text(encoding="utf-8"))
    assert concept_breakdown["document_fixture_count"] >= 4
    compatibility = json.loads((tmp_path / "report" / "compatibility_arc_appendix.json").read_text(encoding="utf-8"))
    assert compatibility["status"] == "SKIPPED"


def test_phase_d_gate_fails_on_document_grounding_regression(tmp_path: Path) -> None:
    fixture_root = tmp_path / "tests" / "fixtures"
    shutil.copytree(Path("tests/fixtures/p1_phase_de"), fixture_root / "p1_phase_de")
    shutil.copytree(Path("tests/fixtures/p1_phase_c"), fixture_root / "p1_phase_c")
    fixture_path = fixture_root / "p1_phase_de" / "document_eval" / "scanned_identity_note.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload["expected_min_document_grounding_score"] = 0.99
    fixture_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    model_run_dir = seed_local_model_run(tmp_path / "model_run")
    phase_root = tmp_path / "phase_root"
    seed_resume_packet_root(phase_root, phase="D")
    h100_path_map = seed_resume_packet_root(tmp_path / "h100", phase="D")
    baseline_dir = tmp_path / "baseline"

    run_phase_d_gate(
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
        freeze_baseline=True,
    )
    result = run_phase_d_gate(
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
        document_fixture_root=fixture_root / "p1_phase_de" / "document_eval",
        proof_fixture_root=fixture_root / "p1_phase_de" / "proof_eval",
    )

    assert result["summary_report"]["regression.status"] == "FAIL"
    assert result["suite_status"]["S4"] == "FAIL"
