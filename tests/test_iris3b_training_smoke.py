from __future__ import annotations

import importlib
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

import iris.train.iris3b_training as iris3b_training
from iris.train.iris3b_checkpoint import load_iris3b_checkpoint
from iris.train.iris3b_model import small_test_config
from iris.train.iris3b_training import P1TrainConfig, run_p1_training_cycle
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
    assert summary["sync_reserve_minutes"] == 0.0
    assert summary["segment_guard_minutes"] == 0.0
    assert summary["cycle_budget_met"] is True
    assert Path(summary["tokenizer_manifest_ref"]).resolve() == (
        run_dir / "tokenizer" / "iris_p1_tokenizer" / "tokenizer_build_manifest.json"
    ).resolve()
    assert not (tmp_path / "tok_work" / "iris_p1_tokenizer").exists()
    assert not (tmp_path / "tok_work" / "tokenizer_corpus.txt").exists()

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


def test_p1_training_cycle_can_stop_before_next_segment_and_reuse_latest_checkpoint(tmp_path: Path) -> None:
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
    run_dir = tmp_path / "run_reuse"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_DISABLE_JIT"] = "1"

    first_proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_p1_3b.py"),
            "cycle",
            "--output-dir",
            str(run_dir),
            "--run-id",
            "smoke-reuse",
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
            str(tmp_path / "tok_work_reuse"),
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
    first_summary = json.loads(first_proc.stdout.strip().splitlines()[-1])

    second_proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_p1_3b.py"),
            "cycle",
            "--output-dir",
            str(run_dir),
            "--run-id",
            "smoke-reuse",
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
            "--sync-reserve-minutes",
            "0.9",
            "--segment-guard-minutes",
            "0.1",
            "--max-segments",
            "1",
            "--tokenizer-workdir",
            str(tmp_path / "tok_work_reuse"),
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
    second_summary = json.loads(second_proc.stdout.strip().splitlines()[-1])

    assert first_summary["checkpoint_manifest_path"] == second_summary["checkpoint_manifest_path"]
    assert second_summary["status"] == "Done"
    assert second_summary["segments_completed"] == 0
    assert second_summary["termination_reason"] == "sync_reserve_reached"

    events = load_journal(run_dir / "segment_journal.jsonl")
    applied = [event for event in events if event.get("status") == "APPLIED"]
    assert len(applied) == 1


def test_p1_training_cycle_stops_before_segment_when_checkpoint_volume_is_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    run_dir = tmp_path / "run_space_guard"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["JAX_DISABLE_JIT"] = "1"

    first_proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_p1_3b.py"),
            "cycle",
            "--output-dir",
            str(run_dir),
            "--run-id",
            "smoke-space-guard",
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
            str(tmp_path / "tok_work_space_guard"),
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
    first_summary = json.loads(first_proc.stdout.strip().splitlines()[-1])
    checkpoint_manifest_path = first_summary["checkpoint_manifest_path"]
    checkpoint_payload_root = run_dir / "checkpoints" / "payloads"

    monkeypatch.setenv("JAX_DISABLE_JIT", "1")

    def fake_free_bytes(path: Path | None) -> int:
        candidate = Path(path) if path is not None else None
        if candidate is not None and candidate.resolve(strict=False) == checkpoint_payload_root.resolve(strict=False):
            return 0
        return 1 << 40

    monkeypatch.setattr(iris3b_training, "_free_bytes", fake_free_bytes)

    summary = run_p1_training_cycle(
        P1TrainConfig(
            output_dir=run_dir,
            run_id="smoke-space-guard",
            manifest_path=manifest_path,
            streaming_mode="auto",
            tokenizer_workdir=tmp_path / "tok_work_space_guard",
            device="cpu",
            strict_jax=False,
            max_cycle_minutes=1,
            max_segments=1,
            dataset_cache_limit_gib=1,
            model_config=model_config,
        )
    )

    assert summary["status"] == "Done"
    assert summary["segments_completed"] == 0
    assert summary["termination_reason"] == "checkpoint_space_guard_stop"
    assert summary["checkpoint_manifest_path"] == checkpoint_manifest_path

    events = load_journal(run_dir / "segment_journal.jsonl")
    applied = [event for event in events if event.get("status") == "APPLIED"]
    pending = [event for event in events if event.get("status") == "PENDING"]
    assert len(applied) == 1
    assert len(pending) == 1


def test_p1_training_cycle_rejects_first_run_when_initial_checkpoint_cannot_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_synthetic_manifest(manifest_path)
    model_config = replace(
        small_test_config(),
        segment_steps=1,
        gradient_accumulation_steps=1,
        micro_batch_size=1,
        warmup_steps=1,
    ).validate()
    run_dir = tmp_path / "run_initial_space"
    checkpoint_payload_root = run_dir / "checkpoints" / "payloads"

    monkeypatch.setenv("JAX_DISABLE_JIT", "1")

    def fake_free_bytes(path: Path | None) -> int:
        candidate = Path(path) if path is not None else None
        if candidate is not None and candidate.resolve(strict=False) == checkpoint_payload_root.resolve(strict=False):
            return 0
        return 1 << 40

    monkeypatch.setattr(iris3b_training, "_free_bytes", fake_free_bytes)

    with pytest.raises(RuntimeError, match="commit the initial checkpoint payload"):
        run_p1_training_cycle(
            P1TrainConfig(
                output_dir=run_dir,
                run_id="smoke-initial-space",
                manifest_path=manifest_path,
                streaming_mode="auto",
                tokenizer_workdir=tmp_path / "tok_work_initial_space",
                device="cpu",
                strict_jax=False,
                max_cycle_minutes=1,
                max_segments=1,
                dataset_cache_limit_gib=1,
                model_config=model_config,
            )
        )

    assert not (run_dir / "segment_journal.jsonl").exists()


def test_cycle_retries_smaller_profile_when_initial_checkpoint_space_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "scripts"))
    train_script = importlib.import_module("train_p1_3b")
    train_script = importlib.reload(train_script)

    import iris.train.iris3b_training as iris3b_training_module

    attempts: list[str] = []
    output_dir = tmp_path / "retry_run"

    def fake_run_p1_training_cycle(config: object) -> dict[str, object]:
        profile = str(config.model_config["profile"])
        attempts.append(profile)
        if len(attempts) == 1:
            raise RuntimeError(
                "Insufficient free space on the checkpoint payload volume to commit the initial checkpoint payload: "
                "checkpoint_payload_root=/tmp/payloads, free_bytes=1, required_free_bytes=2."
            )
        return {
            "status": "Done",
            "checkpoint_manifest_path": str(output_dir / "checkpoints" / "retry" / "segment_000000.json"),
            "streaming_manifest_sha256": "manifest-sha",
            "tokenizer_manifest_ref": "tokenizer/manifest.json",
            "requested_streaming_mode": "auto",
            "effective_streaming_mode": "auto",
            "local_snapshot_manifest_ref": "",
            "last_segment_id": 0,
        }

    monkeypatch.setattr(train_script, "_kaggle_runtime_present", lambda: True)
    monkeypatch.setattr(train_script, "_maybe_bootstrap_from_hf", lambda args: None)
    monkeypatch.setattr(train_script, "_load_model_config", lambda path, memory_profile, kaggle_runtime: {"profile": memory_profile})
    monkeypatch.setattr(train_script, "cycle_memory_profile_candidates", lambda memory_profile, kaggle_runtime: ("kaggle_safe", "kaggle_emergency"))
    monkeypatch.setattr(iris3b_training_module, "run_p1_training_cycle", fake_run_p1_training_cycle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_p1_3b.py",
            "cycle",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "retry",
            "--max-cycle-minutes",
            "1",
            "--no-download-latest",
            "--no-sync-checkpoint",
        ],
    )

    assert train_script.main() == 0

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert attempts == ["kaggle_safe", "kaggle_emergency"]
    assert summary["attempted_memory_profiles"] == ["kaggle_safe", "kaggle_emergency"]
    assert summary["effective_memory_profile"] == "kaggle_emergency"
    assert summary["memory_profile_fallback_used"] is True


def test_p1_training_cycle_supports_external_checkpoint_payload_root(tmp_path: Path) -> None:
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
    run_dir = tmp_path / "run_external"
    payload_root = tmp_path / "spill" / "payloads"
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
            "smoke-external",
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
            "--checkpoint-payload-root",
            str(payload_root),
            "--tokenizer-workdir",
            str(tmp_path / "tok_work_external"),
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
    assert summary["checkpoint_payload_root"] == str(payload_root)

    events = load_journal(run_dir / "segment_journal.jsonl")
    applied = [event for event in events if event.get("status") == "APPLIED"]
    assert len(applied) == 1
    assert not (run_dir / "checkpoints" / "payloads" / "segment_000000").exists()
    assert (payload_root / "segment_000000").exists()

    checkpoint = load_iris3b_checkpoint(
        run_dir / applied[0]["checkpoint_ref"],
        payload_roots=(payload_root,),
    )
    assert checkpoint["segment_id_last_applied"] == 0
