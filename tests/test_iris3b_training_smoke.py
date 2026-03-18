from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("sentencepiece")
pytest.importorskip("transformers")

from iris.train.iris3b_checkpoint import load_iris3b_checkpoint
from iris.train.iris3b_model import small_test_config
from iris.train.journal import load_journal


def _write_synthetic_manifest(path: Path) -> None:
    payload = {
        "schema": "p1_streaming_manifest/v1",
        "manifest_id": "p1-synth-test",
        "profile_id": "P1",
        "phase": "E",
        "commit_posture": "committed",
        "default_streaming_mode": "auto",
        "sources": [
            {"source_id": "a", "source_kind": "synthetic", "pool_id": "A", "pool_role": "core", "token_weight": 0.20, "record_weight": 0.25, "revision": "synthetic-v1"},
            {"source_id": "b", "source_kind": "synthetic", "pool_id": "B", "pool_role": "core", "token_weight": 0.10, "record_weight": 0.10, "revision": "synthetic-v1"},
            {"source_id": "c", "source_kind": "synthetic", "pool_id": "C", "pool_role": "core", "token_weight": 0.25, "record_weight": 0.25, "revision": "synthetic-v1"},
            {"source_id": "d", "source_kind": "synthetic", "pool_id": "D", "pool_role": "core", "token_weight": 0.35, "record_weight": 0.20, "revision": "synthetic-v1"},
            {"source_id": "e", "source_kind": "synthetic", "pool_id": "E", "pool_role": "core", "token_weight": 0.10, "record_weight": 0.20, "revision": "synthetic-v1"},
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def test_p1_training_cycle_writes_governed_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_synthetic_manifest(manifest_path)
    model_config = replace(
        small_test_config(),
        segment_steps=1,
        gradient_accumulation_steps=2,
        micro_batch_size=1,
        warmup_steps=1,
    ).validate()
    model_config_path = tmp_path / "model_config.json"
    model_config_path.write_text(
        json.dumps(model_config.to_payload(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_DISABLE_JIT"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_p1_3b.py"),
            "cycle",
            "--output-dir",
            str(run_dir),
            "--run-id",
            "smoke",
            "--manifest-path",
            str(manifest_path),
            "--model-config-json",
            str(model_config_path),
            "--streaming-mode",
            "auto",
            "--device",
            "cpu",
            "--no-strict-jax",
            "--max-cycle-minutes",
            "1",
            "--max-segments",
            "1",
            "--tokenizer-workdir",
            str(tmp_path / "tok_work"),
            "--dataset-cache-limit-gib",
            "1",
            "--no-download-latest",
            "--no-sync-checkpoint",
        ],
        cwd=str(repo_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])

    assert summary["status"] == "Done"
    assert summary["segments_completed"] == 1

    events = load_journal(run_dir / "segment_journal.jsonl")
    applied = [event for event in events if event.get("status") == "APPLIED"]
    assert len(applied) == 1
    assert applied[0]["data.source_manifest_sha256"]
    assert applied[0]["checkpoint_payload_ref"]

    checkpoint = load_iris3b_checkpoint(run_dir / applied[0]["checkpoint_ref"])
    assert checkpoint["schema"] == "iris.training_checkpoint/v2"
    assert checkpoint["checkpoint_kind"] == "orbax_sidecar"
    assert checkpoint["data_provenance"]["streaming_manifest_sha256"]

    metrics_rows = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = json.loads(metrics_rows[-1])
    assert metrics["runtime_lock_manifest_id"]
    assert metrics["runtime_lock_manifest_sha256"]
    assert metrics["data.source_manifest_sha256"]
    assert metrics["checkpoint_payload_ref"]


def test_p1_training_cycle_prunes_old_checkpoint_payloads(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_synthetic_manifest(manifest_path)
    model_config = replace(
        small_test_config(),
        segment_steps=1,
        gradient_accumulation_steps=1,
        micro_batch_size=1,
        warmup_steps=1,
    ).validate()
    model_config_path = tmp_path / "model_config.json"
    model_config_path.write_text(
        json.dumps(model_config.to_payload(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run_prune"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_DISABLE_JIT"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_p1_3b.py"),
            "cycle",
            "--output-dir",
            str(run_dir),
            "--run-id",
            "smoke-prune",
            "--manifest-path",
            str(manifest_path),
            "--model-config-json",
            str(model_config_path),
            "--streaming-mode",
            "auto",
            "--device",
            "cpu",
            "--no-strict-jax",
            "--max-cycle-minutes",
            "1",
            "--max-segments",
            "2",
            "--checkpoint-retention-limit",
            "1",
            "--tokenizer-workdir",
            str(tmp_path / "tok_work_prune"),
            "--dataset-cache-limit-gib",
            "1",
            "--no-download-latest",
            "--no-sync-checkpoint",
        ],
        cwd=str(repo_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])

    assert summary["status"] == "Done"
    assert summary["segments_completed"] == 2
    assert summary["checkpoint_retention_limit"] == 1
    assert summary["checkpoint_retention"]["pruned_segment_ids"] == [0]

    payload_root = run_dir / "checkpoints" / "payloads"
    remaining_payloads = sorted(path.name for path in payload_root.iterdir())
    assert remaining_payloads == ["segment_000001"]

    events = load_journal(run_dir / "segment_journal.jsonl")
    applied = [event for event in events if event.get("status") == "APPLIED"]
    assert len(applied) == 2

    with pytest.raises(RuntimeError, match="pruned by retention policy"):
        load_iris3b_checkpoint(run_dir / applied[0]["checkpoint_ref"])

    checkpoint = load_iris3b_checkpoint(run_dir / applied[-1]["checkpoint_ref"])
    assert checkpoint["segment_id_last_applied"] == 1
