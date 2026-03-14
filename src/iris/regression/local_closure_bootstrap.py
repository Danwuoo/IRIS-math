from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def seed_local_model_run(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoints" / "segment_000000.json"
    _write_json(checkpoint_path, {"schema": "iris.local_checkpoint/v1", "status": "synthetic"})
    _write_jsonl(
        output_dir / "segment_journal.jsonl",
        [
            {"status": "PENDING", "segment_id": 0},
            {"status": "APPLIED", "segment_id": 0, "checkpoint_ref": "checkpoints/segment_000000.json"},
        ],
    )
    return output_dir


def seed_resume_packet_root(output_dir: Path, *, phase: str) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    path_map: Dict[str, Path] = {}
    for run_id, directory_name in {
        "uninterrupted": "s8_uninterrupted",
        "execute_crash": "s8_execute",
        "pre_commit_crash": "s8_pre_commit",
        "post_commit_crash": "s8_post_commit",
    }.items():
        run_dir = output_dir / directory_name
        path_map[run_id] = run_dir
        _write_json(
            run_dir / "runtime_lock_manifest.json",
            {
                "schema": "iris.runtime_lock_manifest/v1",
                "created_at": "2026-03-14T00:00:00Z",
                "phase": phase,
                "host": {"os": "synthetic", "kernel": "synthetic", "gpu": "none", "nvidia_driver": "n/a", "cuda_runtime": "n/a", "cudnn": "n/a"},
                "python": {"version": "3.12.0", "packages": []},
                "jax": {"jax": "0.0", "jaxlib": "0.0", "jaxlib_build": "n/a", "xla_flags": "", "env": {}},
            },
        )
        _write_json(run_dir / "checkpoints" / "segment_000000.json", {"schema": "iris.local_checkpoint/v1"})
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
