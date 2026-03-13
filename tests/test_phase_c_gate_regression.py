from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "iris" / "regression" / "phase_c_gate.py"
SPEC = importlib.util.spec_from_file_location("iris_phase_c_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["iris_phase_c_gate"] = MODULE
SPEC.loader.exec_module(MODULE)

GateContext = MODULE.GateContext
Tolerances = MODULE.Tolerances
build_concept_breakdown = MODULE.build_concept_breakdown
build_h100_packet_summary = MODULE.build_h100_packet_summary
build_paired_representation_diff = MODULE.build_paired_representation_diff
build_resume_consistency_packet = MODULE.build_resume_consistency_packet
evaluate_s4_status = MODULE.evaluate_s4_status
evaluate_s5_status = MODULE.evaluate_s5_status
evaluate_s8_status = MODULE.evaluate_s8_status


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _runtime_manifest_payload() -> dict:
    return {
        "schema": "iris.runtime_lock_manifest/v1",
        "created_at": "2026-03-01T00:00:00Z",
        "phase": "C",
        "host": {"os": "test", "kernel": "test", "gpu": "none", "nvidia_driver": "n/a", "cuda_runtime": "n/a", "cudnn": "n/a"},
        "python": {"version": "3.12.0", "packages": []},
        "jax": {"jax": "0.0", "jaxlib": "0.0", "jaxlib_build": "n/a", "xla_flags": "", "env": {}},
    }


def test_build_concept_breakdown_from_minimal_conceptarc(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus" / "Copy"
    _write_json(
        corpus / "Copy1.json",
        {
            "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 2]], "output": [[2, 2]]}],
        },
    )
    report = build_concept_breakdown(tmp_path / "corpus", GateContext())
    assert report["status"] == "PASS"
    assert report["concept_count"] == 1
    assert report["concepts"][0]["concept_id"] == "Copy"


def test_gate_context_uses_v2_mandatory_docs_sequence() -> None:
    docs = GateContext().mandatory_docs_consulted
    assert docs == (
        "docs/00_INDEX.md",
        "docs/10_Glossary_and_Normative_Status.md",
        "docs/13_Goals_and_Success_Criteria.md",
        "docs/07_Data_Constitution.md",
        "docs/01_Architecture_Constitution.md",
        "docs/02_State_IR_Spec.md",
        "docs/03_Level_Contracts_L0-L6.md",
        "docs/04_Credit_Assignment_and_Recovery.md",
        "docs/18_Optimization_and_Learning_Contract.md",
        "docs/19_Runtime_and_Task_Adjudication_Semantics.md",
        "docs/05_Eval_Metrics_Spec.md",
        "docs/06_Regression_and_Phase_Gates.md",
        "docs/08_Training_Run_Governance.md",
        "docs/09_Training_Profiles_and_Scaling.md",
        "docs/14_Multimodal_Document_Pipeline.md",
        "docs/15_Benchmark_Registry_and_Tiering_Playbook.md",
        "docs/16_Verifier_and_Formalization_Stack.md",
        "docs/17_Scaling_Promotion_and_Readiness.md",
    )


def test_build_paired_representation_diff_from_minimal_rearc(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _write_json(
        tasks / "a.json",
        [
            {"input": [[1, 1]], "output": [[1, 1]]},
            {"input": [[2, 2]], "output": [[2, 0]]},
            {"input": [[3, 3]], "output": [[3, 3]]},
            {"input": [[4, 4]], "output": [[4, 0]]},
        ],
    )
    report = build_paired_representation_diff(tasks, GateContext(), max_tasks=8, max_pairs_per_task=4)
    assert report["status"] == "PASS"
    assert report["pair_count"] == 2
    assert report["paired.invariance.gap"] >= 0.0


def test_resume_consistency_packet_passes_when_four_paths_exist(tmp_path: Path) -> None:
    context = GateContext()
    run_ids = ["uninterrupted", "execute_crash", "pre_commit_crash", "post_commit_crash"]
    path_map = {}
    for run_id in run_ids:
        run_dir = tmp_path / run_id
        path_map[run_id] = run_dir
        _write_json(
            run_dir / "runtime_lock_manifest.json",
            _runtime_manifest_payload(),
        )
        checkpoint_rel = "checkpoints/segment_000000.json"
        _write_json(run_dir / checkpoint_rel, {"schema": "iris.checkpoint.test/v1"})
        _write_jsonl(
            run_dir / "segment_journal.jsonl",
            [
                {"status": "PENDING", "segment_id": 0},
                {"status": "APPLIED", "segment_id": 0, "checkpoint_ref": checkpoint_rel},
            ],
        )
        _write_jsonl(
            run_dir / "metrics.jsonl",
            [
                {
                    "segment_id": 0,
                    "optimizer_step_id": 1,
                    "dataset_slice_id": "slice-000000",
                    "data_seed": 17,
                    "rng_hash_pre": "a",
                    "rng_hash_post": "b",
                    "runtime_lock_manifest_id": "lock",
                    "runtime_lock_manifest_sha256": "locksha",
                    "task.validity_score": 0.5,
                    "task.confidence": 0.5,
                    "failure.credit.collapse_rate": 1.0 / 7.0,
                    "eval.calibration_error": 0.0,
                    "rep.tokenizer.ir_fragmentation_rate": 0.0,
                    "paired.invariance.gap": 0.0,
                    "concept.leakage_score": 0.0,
                    "process.failure_distribution_entropy": 1.0,
                    "failure.credit": {
                        "L0": 1.0 / 7.0,
                        "L1": 1.0 / 7.0,
                        "L2": 1.0 / 7.0,
                        "L3": 1.0 / 7.0,
                        "L4": 1.0 / 7.0,
                        "L5": 1.0 / 7.0,
                        "L6": 1.0 / 7.0,
                    },
                }
            ],
        )

    packet = build_resume_consistency_packet(path_map, context)
    status, violations = evaluate_s8_status(packet)
    assert status == "PASS"
    assert not violations


def test_s4_fails_when_baseline_missing_in_phase_c() -> None:
    status, reasons, details = evaluate_s4_status(
        current_concept_breakdown={
            "concepts": [{"concept_id": "Copy", "concept.isolation_score": 0.8, "concept.leakage_score": 0.2}]
        },
        baseline_concept_breakdown=None,
        tolerances=Tolerances(),
        phase="C",
    )
    assert status == "FAIL"
    assert reasons
    assert not details


def test_s5_detects_invariance_gap_regression() -> None:
    status, reasons, details = evaluate_s5_status(
        current_paired_diff={"pair_count": 4, "paired.asymmetry_rate": 0.1, "paired.invariance.gap": 0.2},
        baseline_paired_diff={"pair_count": 4, "paired.asymmetry_rate": 0.0, "paired.invariance.gap": 0.0},
        tolerances=Tolerances(paired_asymmetry_delta_epsilon=0.01, paired_invariance_gap_delta_epsilon=0.01),
        phase="C",
    )
    assert status == "FAIL"
    assert reasons
    assert details
    assert any(item["metric"] == "paired.invariance.gap" for item in details)


def test_h100_packet_fails_when_checkpoint_is_not_resolvable(tmp_path: Path) -> None:
    context = GateContext()
    path_map = {}
    for run_id in ["uninterrupted", "execute_crash", "pre_commit_crash", "post_commit_crash"]:
        run_dir = tmp_path / run_id
        path_map[run_id] = run_dir
        _write_json(run_dir / "runtime_lock_manifest.json", _runtime_manifest_payload())
        _write_jsonl(
            run_dir / "segment_journal.jsonl",
            [
                {"status": "PENDING", "segment_id": 0},
                {"status": "APPLIED", "segment_id": 0, "checkpoint_ref": "/not/found/checkpoint.json"},
            ],
        )
        _write_jsonl(
            run_dir / "metrics.jsonl",
            [
                {
                    "segment_id": 0,
                    "optimizer_step_id": 1,
                    "dataset_slice_id": "slice-000000",
                    "data_seed": 17,
                    "rng_hash_pre": "a",
                    "rng_hash_post": "b",
                    "runtime_lock_manifest_id": "lock",
                    "runtime_lock_manifest_sha256": "locksha",
                    "task.validity_score": 0.5,
                    "task.confidence": 0.5,
                    "failure.credit.collapse_rate": 1.0 / 7.0,
                    "eval.calibration_error": 0.0,
                    "rep.tokenizer.ir_fragmentation_rate": 0.0,
                    "paired.invariance.gap": 0.0,
                    "concept.leakage_score": 0.0,
                    "process.failure_distribution_entropy": 1.0,
                    "failure.credit": {f"L{i}": 1.0 / 7.0 for i in range(7)},
                }
            ],
        )

    summary = build_h100_packet_summary(path_map, context)
    assert summary["s8_status_for_h100_packet"] == "FAIL"
    assert "checkpoint_ref" in str(summary.get("block_reason", ""))
