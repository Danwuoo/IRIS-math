from __future__ import annotations

import json
from pathlib import Path

from iris.train.hf_sync import HFRepoSpec, bootstrap_checkpoint_run, write_latest_pointer


def test_bootstrap_checkpoint_run_restores_run_tree_from_latest_pointer(tmp_path: Path, monkeypatch) -> None:
    remote_root = tmp_path / "remote_repo"
    run_root = remote_root / "checkpoints" / "p1-run"
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "segment_000001.json").write_text(
        json.dumps({"schema": "iris.training_checkpoint/v2"}, sort_keys=True),
        encoding="utf-8",
    )
    (run_root / "segment_journal.jsonl").write_text(
        json.dumps({"status": "APPLIED", "segment_id": 1, "checkpoint_ref": "checkpoints/segment_000001.json"}) + "\n",
        encoding="utf-8",
    )
    latest_path = remote_root / "latest" / "latest.json"
    write_latest_pointer(
        output_path=latest_path,
        run_id="p1-run",
        segment_id=1,
        checkpoint_manifest_path="checkpoints/p1-run/checkpoints/segment_000001.json",
    )

    def fake_snapshot_download(*, repo, local_dir, token, allow_patterns=None):
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for pattern in allow_patterns or []:
            if pattern == "latest/latest.json":
                target = destination / "latest" / "latest.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(latest_path.read_text(encoding="utf-8"), encoding="utf-8")
            elif pattern.startswith("checkpoints/p1-run/"):
                import shutil

                shutil.copytree(run_root, destination / "checkpoints" / "p1-run", dirs_exist_ok=True)
        return destination

    monkeypatch.setattr("iris.train.hf_sync.snapshot_download", fake_snapshot_download)

    local_dir = tmp_path / "local_run"
    pointer = bootstrap_checkpoint_run(
        repo=HFRepoSpec(repo_id="Danwuoo/IRIS-math"),
        local_dir=local_dir,
        token="token",
    )

    assert pointer is not None
    assert pointer["run_id"] == "p1-run"
    assert (local_dir / "segment_journal.jsonl").exists()
    assert (local_dir / "checkpoints" / "segment_000001.json").exists()
