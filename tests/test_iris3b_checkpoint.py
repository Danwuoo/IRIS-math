from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.train.iris3b_checkpoint import (
    checkpoint_manifest_path,
    checkpoint_payload_dir,
    load_iris3b_checkpoint,
    prune_checkpoint_payloads,
)


def _write_payload(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * int(size_bytes))


def test_prune_checkpoint_payloads_keeps_latest_and_explicit_segments(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_sizes = {0: 11, 1: 23, 2: 31}
    for segment_id, size_bytes in expected_sizes.items():
        checkpoint_manifest_path(checkpoint_dir, segment_id).write_text("{}", encoding="utf-8")
        _write_payload(
            checkpoint_payload_dir(checkpoint_dir, segment_id) / "params" / "tensor.bin",
            size_bytes,
        )

    summary = prune_checkpoint_payloads(
        checkpoint_dir=checkpoint_dir,
        keep_last=1,
        keep_segment_ids=(0,),
    )

    assert summary["retention_limit"] == 1
    assert summary["kept_segment_ids"] == [0, 2]
    assert summary["pruned_segment_ids"] == [1]
    assert summary["deleted_bytes"] >= expected_sizes[1]
    assert checkpoint_payload_dir(checkpoint_dir, 0).exists()
    assert not checkpoint_payload_dir(checkpoint_dir, 1).exists()
    assert checkpoint_payload_dir(checkpoint_dir, 2).exists()


def test_load_iris3b_checkpoint_reports_pruned_payloads(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    manifest_path = checkpoint_manifest_path(checkpoint_dir, 3)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "iris.training_checkpoint/v2",
                "payload_refs": {
                    "params": "payloads/segment_000003/params",
                    "optimizer_state": "payloads/segment_000003/optimizer_state",
                    "rng_state": "payloads/segment_000003/rng_state",
                },
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="pruned by retention policy"):
        load_iris3b_checkpoint(manifest_path)
