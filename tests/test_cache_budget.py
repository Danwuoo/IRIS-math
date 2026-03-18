from __future__ import annotations

import os
import shutil
from pathlib import Path

from iris.train.cache_budget import enforce_dataset_cache_budget


def _write_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * int(size))


def test_enforce_dataset_cache_budget_prunes_oldest_candidates(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    snapshot_root = tmp_path / "snapshots"

    oldest = cache_root / "hf" / "datasets" / "old.bin"
    newest = cache_root / "hf" / "hub" / "new.bin"
    snapshot = snapshot_root / "prooflang_document_aux" / "snapshot.bin"
    _write_file(oldest, 30)
    _write_file(newest, 25)
    _write_file(snapshot, 5)

    os.utime(oldest, (1, 1))
    os.utime(newest, (3, 3))
    os.utime(snapshot, (2, 2))

    summary = enforce_dataset_cache_budget(
        roots=(cache_root, snapshot_root),
        budget_bytes=35,
    )

    assert summary["before_bytes"] == 60
    assert summary["after_bytes"] <= 35
    assert not oldest.exists()
    assert newest.exists()
    assert snapshot.exists()


def test_enforce_dataset_cache_budget_prunes_when_free_space_floor_is_violated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root = tmp_path / "cache"

    oldest = cache_root / "hf" / "datasets" / "old.bin"
    newest = cache_root / "hf" / "hub" / "new.bin"
    _write_file(oldest, 30)
    _write_file(newest, 20)

    os.utime(oldest, (1, 1))
    os.utime(newest, (2, 2))

    fake_usage = shutil._ntuple_diskusage(total=100, used=90, free=10)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: fake_usage)

    summary = enforce_dataset_cache_budget(
        roots=(cache_root,),
        budget_bytes=100,
        min_free_bytes=35,
        monitor_roots=(cache_root,),
    )

    assert summary["budget_guard_triggered"] is False
    assert summary["low_disk_guard_triggered"] is True
    assert summary["free_bytes_before"] == 10
    assert summary["free_bytes_after"] >= 40
    assert not oldest.exists()
    assert newest.exists()
