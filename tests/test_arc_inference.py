from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from iris.arc import ArcEvalConfig, ArcExample, ArcTask, encode_arc_case_to_state, run_arc_diagnostic_eval
from iris.schema import STATE_IR_TOKEN_ORDER
from iris.train.checkpoint import save_checkpoint_atomic


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_minimal_model_state(hidden_dim: int) -> dict:
    trunk = {
        "type_embeddings": np.zeros((len(STATE_IR_TOKEN_ORDER), hidden_dim), dtype=np.float32),
        "seq_w": np.zeros((hidden_dim, hidden_dim), dtype=np.float32),
        "seq_b": np.zeros((hidden_dim,), dtype=np.float32),
        "ctrl_w": np.zeros((hidden_dim, 8), dtype=np.float32),
        "ctrl_b": np.zeros((8,), dtype=np.float32),
    }
    levels = {}
    for level_idx in range(7):
        level_id = f"L{level_idx}"
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
    return {
        "schema": "iris.model_state/v2",
        "backend": "jax",
        "hidden_dim": hidden_dim,
        "trunk": trunk,
        "levels": levels,
    }


def _build_checkpoint_run(tmp_path: Path, hidden_dim: int = 16) -> Path:
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_path = save_checkpoint_atomic(
        checkpoint_dir=checkpoints_dir,
        segment_id=0,
        payload={
            "model_state": _build_minimal_model_state(hidden_dim),
            "optimizer_state": {"step": 0},
            "rng_state": {"rng.model.train": 0, "rng.control.train": 0, "rng.data.train": 0},
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


def test_encode_arc_case_to_state_uses_canonical_token_order() -> None:
    task = ArcTask(
        task_id="toy",
        train_examples=(ArcExample(input_grid=[[1]], output_grid=[[1]]),),
        test_examples=(ArcExample(input_grid=[[2]], output_grid=[[2]]),),
        source_path="toy.json",
        source_name="ConceptARC",
        concept_id="Copy",
    )
    state = encode_arc_case_to_state(task=task, test_example=task.test_examples[0], hidden_dim=8)
    lengths = state.section_lengths()
    assert tuple(lengths.keys()) == STATE_IR_TOKEN_ORDER
    assert state.T.shape[0] == 1
    assert state.G.shape[0] == 1
    assert state.to_canonical_sequence().shape[1] == 8


def test_run_arc_diagnostic_eval_is_deterministic_for_same_checkpoint(tmp_path: Path) -> None:
    run_dir = _build_checkpoint_run(tmp_path, hidden_dim=12)
    task = ArcTask(
        task_id="toy_task",
        train_examples=(ArcExample(input_grid=[[1, 1]], output_grid=[[1, 1]]),),
        test_examples=(ArcExample(input_grid=[[2, 2]], output_grid=[[2, 2]]),),
        source_path="toy_task.json",
        source_name="ConceptARC",
        concept_id="Copy",
    )
    config = ArcEvalConfig(
        model_run_dir=run_dir,
        max_reasoning_cycles=2,
        termination_threshold=0.5,
        seed=17,
        device="cpu",
    )
    records_a = run_arc_diagnostic_eval(tasks=[task], config=config, mode="concept.isolation")
    records_b = run_arc_diagnostic_eval(tasks=[task], config=config, mode="concept.isolation")

    assert len(records_a) == 1
    assert len(records_b) == 1
    rec_a = records_a[0]
    rec_b = records_b[0]
    assert rec_a.predicted_grid == rec_b.predicted_grid
    assert rec_a.failure_code == rec_b.failure_code
    assert abs(rec_a.confidence - rec_b.confidence) < 1e-12
    assert abs(sum(rec_a.failure_code_distribution.values()) - 1.0) < 1e-6
