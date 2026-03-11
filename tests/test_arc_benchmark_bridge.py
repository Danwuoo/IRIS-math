from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.arc.benchmark_bridge import export_benchmark_submission
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


def _install_vendor_src() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor_src = root / "tools" / "arc-agi-benchmarking" / "src"
    if str(vendor_src) not in sys.path:
        sys.path.insert(0, str(vendor_src))


def test_export_submission_is_schema_valid_and_scorable(tmp_path: Path) -> None:
    model_run_dir = _build_model_run(tmp_path)
    tasks_dir = tmp_path / "tasks"
    _write_json(
        tasks_dir / "task_a.json",
        {
            "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 2]], "output": [[2, 2]]}],
        },
    )

    submission_dir = tmp_path / "submission"
    summary = export_benchmark_submission(
        model_run_dir=model_run_dir,
        tasks_dir=tasks_dir,
        submission_dir=submission_dir,
        max_reasoning_cycles=1,
        termination_threshold=0.5,
        seed=17,
        device="cpu",
    )
    assert summary["status"] == "PASS"
    assert summary["tasks_exported"] == 1

    submission_file = submission_dir / "task_a.json"
    assert submission_file.exists()
    payload = json.loads(submission_file.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    assert "attempt_1" in payload[0]

    _install_vendor_src()
    from arc_agi_benchmarking.scoring.scoring import ARCScorer
    from arc_agi_benchmarking.schemas import BenchmarkedTaskResults

    BenchmarkedTaskResults(test_pairs=payload)

    scorer = ARCScorer(
        task_dir=str(tasks_dir),
        submission_dir=str(submission_dir),
        print_logs=False,
        results_dir=str(tmp_path / "results"),
    )
    total_score, total_tasks = scorer.score_submission()
    assert total_tasks == 1
    assert 0.0 <= float(total_score) <= float(total_tasks)
