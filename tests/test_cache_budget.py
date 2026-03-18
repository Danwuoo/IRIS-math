from __future__ import annotations

import os
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
