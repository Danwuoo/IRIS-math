from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.train import ToyTrainConfig, run_toy_training
from iris.train.checkpoint import load_checkpoint
from iris.train.journal import load_journal


def _pinned_manifest_payload() -> dict:
    return {
        "schema": "iris.runtime_lock_manifest/v1",
        "created_at": "2026-03-01T00:00:00Z",
        "phase": "C",
        "host": {
            "os": "test-os",
            "kernel": "test-kernel",
            "gpu": "unknown",
            "nvidia_driver": "unknown",
            "cuda_runtime": "unknown",
            "cudnn": "unknown",
        },
        "python": {"version": "3.12.0", "packages": []},
        "jax": {"jax": "0.0", "jaxlib": "0.0", "jaxlib_build": "unknown", "xla_flags": "", "env": {}},
    }


def test_pre_commit_crash_resume_keeps_single_applied_event(tmp_path: Path) -> None:
    output_dir = tmp_path / "toy_resume_jax"

    crash_config = ToyTrainConfig(
        output_dir=output_dir,
        segments=1,
        micro_steps=2,
        device="cpu",
        backend="jax",
        strict_jax=True,
        level_impl="jax_transition",
        crash_point="pre_commit",
        crash_segment=0,
    )
    with pytest.raises(RuntimeError):
        run_toy_training(crash_config)

    resume_config = ToyTrainConfig(
        output_dir=output_dir,
        segments=1,
        micro_steps=2,
        device="cpu",
        backend="jax",
        strict_jax=True,
        level_impl="jax_transition",
    )
    summary = run_toy_training(resume_config)
    assert summary["status"] == "Done"

    events = load_journal(output_dir / "segment_journal.jsonl")
    applied_segment0 = [
        event
        for event in events
        if int(event.get("segment_id", -1)) == 0 and event.get("status") == "APPLIED"
    ]
    assert len(applied_segment0) == 1

    runtime_manifest = json.loads((output_dir / "runtime_lock_manifest.json").read_text(encoding="utf-8"))
    assert runtime_manifest["schema"] == "iris.runtime_lock_manifest/v1"
    assert "created_at" in runtime_manifest
    applied_event = applied_segment0[0]
    assert applied_event["policy_bundle_sha256"]
    assert applied_event["data_realization_policy_id"] == "p1-bootstrap-c-v1"
    assert applied_event["decontam_policy_id"] == "global-decontam-v2"
    assert applied_event["learning_objective_bundle_id"] == "p1-phase-c-bundle-v1"
    assert applied_event["parser_provenance_id"] == "math-doc-pipeline-v1"
    assert applied_event["formalizer_provenance_id"] == "formalizer-skeleton-v1"
    assert applied_event["verifier_provenance_id"] == "verifier-stack-v1"

    metrics_rows = (output_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert metrics_rows
    metrics_record = json.loads(metrics_rows[-1])
    assert metrics_record["runtime_lock_manifest_id"]
    assert metrics_record["runtime_lock_manifest_sha256"]
    assert metrics_record["code_version_hash"]
    assert metrics_record["config_hash"]
    assert metrics_record["policy_bundle_sha256"]
    assert metrics_record["data_realization_policy_id"] == "p1-bootstrap-c-v1"
    assert metrics_record["learning_objective_bundle_id"] == "p1-phase-c-bundle-v1"
    assert metrics_record["parser_provenance_refs"]["semantic_unit_typer_manifest_id"] == "semantic-unit-typer-v1"

    checkpoint = load_checkpoint(Path(applied_event["checkpoint_ref"]))
    assert checkpoint["schema"] == "iris.training_checkpoint/v2"
    assert checkpoint["policy_bundle_sha256"] == applied_event["policy_bundle_sha256"]
    assert checkpoint["profile_id"] == "P1"
    assert checkpoint["phase"] == "C"
    assert checkpoint["data_realization_policy_id"] == "p1-bootstrap-c-v1"
    assert checkpoint["learning_objective_bundle_id"] == "p1-phase-c-bundle-v1"
    assert checkpoint["parser_provenance_id"] == "math-doc-pipeline-v1"
    assert checkpoint["formalizer_provenance_id"] == "formalizer-skeleton-v1"
    assert checkpoint["verifier_provenance_id"] == "verifier-stack-v1"


def test_runtime_lock_manifest_can_be_pinned_across_runs(tmp_path: Path) -> None:
    pinned_manifest_path = tmp_path / "pinned_runtime_lock_manifest.json"
    pinned_manifest_payload = _pinned_manifest_payload()
    pinned_manifest_path.write_text(
        json.dumps(pinned_manifest_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    output_a = tmp_path / "run_a"
    output_b = tmp_path / "run_b"
    for output_dir in (output_a, output_b):
        config = ToyTrainConfig(
            output_dir=output_dir,
            segments=1,
            micro_steps=1,
            device="cpu",
            backend="jax",
            strict_jax=True,
            level_impl="jax_transition",
            runtime_lock_manifest_path=pinned_manifest_path,
        )
        summary = run_toy_training(config)
        assert summary["status"] == "Done"

    manifest_a = json.loads((output_a / "runtime_lock_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((output_b / "runtime_lock_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a == pinned_manifest_payload
    assert manifest_b == pinned_manifest_payload

    metrics_a = json.loads((output_a / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    metrics_b = json.loads((output_b / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert metrics_a["runtime_lock_manifest_id"] == metrics_b["runtime_lock_manifest_id"]
    assert metrics_a["runtime_lock_manifest_sha256"] == metrics_b["runtime_lock_manifest_sha256"]
    assert metrics_a["policy_bundle_sha256"] == metrics_b["policy_bundle_sha256"]
    assert metrics_a["data_realization_policy_id"] == metrics_b["data_realization_policy_id"]
