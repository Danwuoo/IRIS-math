from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from iris.train.hf_sync import HFRepoSpec, bootstrap_checkpoint_run, sync_checkpoint_run, write_latest_pointer


def test_bootstrap_checkpoint_run_restores_run_tree_from_latest_pointer(tmp_path: Path, monkeypatch) -> None:
    remote_root = tmp_path / "remote_repo"
    run_root = remote_root / "checkpoints" / "p1-run"
    (run_root / "checkpoints" / "payloads" / "segment_000000").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "payloads" / "segment_000001").mkdir(parents=True, exist_ok=True)
    (run_root / "data").mkdir(parents=True, exist_ok=True)
    (run_root / "tokenizer" / "iris_p1_tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "metrics.jsonl").write_text(
        json.dumps({"segment_id": 1}) + "\n",
        encoding="utf-8",
    )
    (run_root / "checkpoints" / "segment_000000.json").write_text(
        json.dumps({"schema": "iris.training_checkpoint/v2"}, sort_keys=True),
        encoding="utf-8",
    )
    (run_root / "checkpoints" / "segment_000001.json").write_text(
        json.dumps({"schema": "iris.training_checkpoint/v2"}, sort_keys=True),
        encoding="utf-8",
    )
    (run_root / "checkpoints" / "payloads" / "segment_000000" / "params").write_text(
        "old",
        encoding="utf-8",
    )
    (run_root / "checkpoints" / "payloads" / "segment_000001" / "params").write_text(
        "latest",
        encoding="utf-8",
    )
    (run_root / "segment_journal.jsonl").write_text(
        json.dumps({"status": "APPLIED", "segment_id": 1, "checkpoint_ref": "checkpoints/segment_000001.json"}) + "\n",
        encoding="utf-8",
    )
    (run_root / "data" / "p1_streaming_manifest_committed.json").write_text(
        json.dumps({"manifest_id": "p1"}),
        encoding="utf-8",
    )
    (run_root / "tokenizer" / "iris_p1_tokenizer" / "tokenizer_build_manifest.json").write_text(
        json.dumps({"manifest_id": "tok"}),
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
        del repo, token
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        normalized_patterns = [str(pattern) for pattern in allow_patterns or []]
        for source in remote_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(remote_root).as_posix()
            if not any(fnmatch.fnmatch(relative, pattern) for pattern in normalized_patterns):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
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
    assert (local_dir / "checkpoints" / "payloads" / "segment_000001" / "params").exists()
    assert (local_dir / "data" / "p1_streaming_manifest_committed.json").exists()
    assert (local_dir / "tokenizer" / "iris_p1_tokenizer" / "tokenizer_build_manifest.json").exists()
    assert not (local_dir / "checkpoints" / "segment_000000.json").exists()
    assert not (local_dir / "checkpoints" / "payloads" / "segment_000000").exists()
    assert not (tmp_path / "local_run_hf_bootstrap").exists()


def test_sync_checkpoint_run_uploads_external_checkpoint_payload_root(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    params_root = tmp_path / "spill_params" / "payloads"
    optimizer_root = tmp_path / "spill_optimizer" / "payloads"
    latest_pointer_path = run_dir / "hf_latest_pointer.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    params_root.mkdir(parents=True, exist_ok=True)
    optimizer_root.mkdir(parents=True, exist_ok=True)
    latest_pointer_path.write_text("{}", encoding="utf-8")

    calls: list[tuple[str, str, str]] = []

    def fake_ensure_repo_exists(*, repo, token, private=True):
        calls.append(("ensure", repo.repo_id, str(private)))

    def fake_upload_folder(*, repo, folder_path, path_in_repo, token, commit_message, allow_patterns=None):
        del token, allow_patterns
        calls.append(("folder", str(Path(folder_path)), str(path_in_repo)))
        assert repo.repo_id == "Danwuoo/IRIS-math"
        assert commit_message

    def fake_upload_file(*, repo, path_or_fileobj, path_in_repo, token, commit_message):
        del token
        calls.append(("file", str(Path(path_or_fileobj)), str(path_in_repo)))
        assert repo.repo_id == "Danwuoo/IRIS-math"
        assert commit_message

    monkeypatch.setattr("iris.train.hf_sync.ensure_repo_exists", fake_ensure_repo_exists)
    monkeypatch.setattr("iris.train.hf_sync.upload_folder", fake_upload_folder)
    monkeypatch.setattr("iris.train.hf_sync.upload_file", fake_upload_file)

    sync_checkpoint_run(
        repo=HFRepoSpec(repo_id="Danwuoo/IRIS-math"),
        run_dir=run_dir,
        run_id="p1-run",
        latest_pointer_path=latest_pointer_path,
        checkpoint_payload_roots={
            "params": params_root,
            "optimizer_state": optimizer_root,
            "rng_state": params_root,
        },
        token="token",
    )

    assert ("folder", str(run_dir), "checkpoints/p1-run") in calls
    assert ("folder", str(params_root), "checkpoints/p1-run/checkpoints/payloads") in calls
    assert ("folder", str(optimizer_root), "checkpoints/p1-run/checkpoints/payloads") in calls
