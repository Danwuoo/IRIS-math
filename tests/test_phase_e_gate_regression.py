from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.regression import GateContext, Tolerances, run_phase_e_gate
from iris.schema import STATE_IR_TOKEN_ORDER
from iris.train.checkpoint import save_checkpoint_atomic


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
        "created_at": "2026-03-03T00:00:00Z",
        "phase": "E",
        "host": {
            "os": "test",
            "kernel": "test",
            "gpu": "none",
            "nvidia_driver": "n/a",
            "cuda_runtime": "n/a",
            "cudnn": "n/a",
        },
        "python": {"version": "3.12.0", "packages": []},
        "jax": {"jax": "0.0", "jaxlib": "0.0", "jaxlib_build": "n/a", "xla_flags": "", "env": {}},
    }


def _build_model_run(tmp_path: Path, hidden_dim: int = 12) -> Path:
    trunk = {
        "type_embeddings": np.zeros((len(STATE_IR_TOKEN_ORDER), hidden_dim), dtype=np.float32),
        "seq_w": np.zeros((hidden_dim, hidden_dim), dtype=np.float32),
        "seq_b": np.zeros((hidden_dim,), dtype=np.float32),
        "ctrl_w": np.zeros((hidden_dim, 8), dtype=np.float32),
        "ctrl_b": np.zeros((8,), dtype=np.float32),
    }
    levels = {}
    for idx in range(7):
        level_id = f"L{idx}"
        payload = {
            "res_w": np.zeros((hidden_dim, hidden_dim), dtype=np.float32),
            "res_b": np.zeros((hidden_dim,), dtype=np.float32),
            "gate_w": np.zeros((hidden_dim, hidden_dim), dtype=np.float32),
            "gate_b": np.zeros((hidden_dim,), dtype=np.float32),
            "ctrl_w": np.zeros((hidden_dim, 8), dtype=np.float32),
            "ctrl_b": np.zeros((8,), dtype=np.float32),
        }
        if level_id == "L6":
            payload["credit_w"] = np.zeros((hidden_dim, 7), dtype=np.float32)
            payload["credit_b"] = np.zeros((7,), dtype=np.float32)
        levels[level_id] = payload

    run_dir = tmp_path / "model_run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_path = save_checkpoint_atomic(
        checkpoint_dir=checkpoints_dir,
        segment_id=0,
        payload={
            "model_state": {
                "schema": "iris.model_state/v2",
                "backend": "jax",
                "hidden_dim": hidden_dim,
                "trunk": trunk,
                "levels": levels,
            }
        },
    )
    _write_jsonl(
        run_dir / "segment_journal.jsonl",
        [
            {"status": "PENDING", "segment_id": 0},
            {"status": "APPLIED", "segment_id": 0, "checkpoint_ref": str(checkpoint_path)},
        ],
    )
    return run_dir


def _build_resume_packet_paths(base_dir: Path) -> dict[str, Path]:
    directory_name_by_run = {
        "uninterrupted": "s8_uninterrupted",
        "execute_crash": "s8_execute",
        "pre_commit_crash": "s8_pre_commit",
        "post_commit_crash": "s8_post_commit",
    }
    path_map: dict[str, Path] = {}
    for run_id in ("uninterrupted", "execute_crash", "pre_commit_crash", "post_commit_crash"):
        run_dir = base_dir / directory_name_by_run[run_id]
        path_map[run_id] = run_dir
        _write_json(run_dir / "runtime_lock_manifest.json", _runtime_manifest_payload())
        _write_json(run_dir / "checkpoints" / "segment_000000.json", {"schema": "iris.checkpoint.test/v1"})
        _write_jsonl(
            run_dir / "segment_journal.jsonl",
            [
                {"status": "PENDING", "segment_id": 0},
                {"status": "APPLIED", "segment_id": 0, "checkpoint_ref": "checkpoints/segment_000000.json"},
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
    return path_map


def _build_probe(path: Path, *, status: str, baseline_non_regression: bool) -> None:
    _write_json(
        path,
        {
            "status": status,
            "baseline_non_regression": baseline_non_regression,
            "probe_a_baseline": {"scoring": {"results_json_path": str(path.parent / "baseline_results.json")}},
            "probe_b_iris": {"scoring": {"results_json_path": str(path.parent / "iris_results.json")}},
        },
    )
    _write_json(path.parent / "baseline_results.json", {"ok": True})
    _write_json(path.parent / "iris_results.json", {"ok": True})


def _run_gate(tmp_path: Path, probe_status: str, baseline_non_regression: bool) -> str:
    model_run_dir = _build_model_run(tmp_path)
    _write_json(
        tmp_path / "conceptarc" / "Copy" / "copy_a.json",
        {
            "train": [{"input": [[1, 1]], "output": [[1, 1]]}],
            "test": [{"input": [[1, 1]], "output": [[1, 1]]}],
        },
    )
    _write_json(
        tmp_path / "rearc" / "pair_a.json",
        [
            {"input": [[1]], "output": [[1]]},
            {"input": [[2]], "output": [[2]]},
        ],
    )

    phase_paths = _build_resume_packet_paths(tmp_path / "phase_root")
    context = GateContext(
        phase="E",
        baseline_id="phase-e-v1",
        tolerance_profile_id="phase-e-default",
        change_class="Capability expansion (Phase E streaming pretrain + benchmark bridge)",
    )
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

    baseline_dir = tmp_path / "baseline_phase_e_v1"

    # First run initializes baseline artifacts inherited from Phase D suites.
    probe_path = tmp_path / "probe" / "arc_benchmark_probe.json"
    _build_probe(probe_path, status="PASS", baseline_non_regression=True)
    run_phase_e_gate(
        context=context,
        tolerances=tolerances,
        output_dir=tmp_path / "report_init",
        model_run_dir=model_run_dir,
        conceptarc_corpus=tmp_path / "conceptarc",
        rearc_tasks=tmp_path / "rearc",
        baseline_report_dir=baseline_dir,
        phase_root=tmp_path / "phase_root",
        h100_path_map=phase_paths,
        s1_status="PASS",
        s1_reasons=[],
        s2_status="PASS",
        s2_reasons=[],
        s1_output={"status": "PASS"},
        s2_output={"status": "PASS"},
        s2_mounted_output={"status": "PASS"},
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
        arc_benchmark_probe_path=probe_path,
        pairing_policy="adjacent",
        freeze_baseline=True,
        hard_fail=True,
    )

    _build_probe(probe_path, status=probe_status, baseline_non_regression=baseline_non_regression)
    result = run_phase_e_gate(
        context=context,
        tolerances=tolerances,
        output_dir=tmp_path / "report",
        model_run_dir=model_run_dir,
        conceptarc_corpus=tmp_path / "conceptarc",
        rearc_tasks=tmp_path / "rearc",
        baseline_report_dir=baseline_dir,
        phase_root=tmp_path / "phase_root",
        h100_path_map=phase_paths,
        s1_status="PASS",
        s1_reasons=[],
        s2_status="PASS",
        s2_reasons=[],
        s1_output={"status": "PASS"},
        s2_output={"status": "PASS"},
        s2_mounted_output={"status": "PASS"},
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
        arc_benchmark_probe_path=probe_path,
        pairing_policy="adjacent",
        freeze_baseline=False,
        hard_fail=True,
    )
    return str(result["summary_report"]["regression.status"])


def test_phase_e_gate_passes_with_passing_probe(tmp_path: Path) -> None:
    assert _run_gate(tmp_path, probe_status="PASS", baseline_non_regression=True) == "PASS"


def test_phase_e_gate_fails_when_probe_regresses(tmp_path: Path) -> None:
    assert _run_gate(tmp_path, probe_status="FAIL", baseline_non_regression=False) == "FAIL"
