from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence

_GIB = 1024 ** 3


def gib_to_bytes(value: int | float) -> int:
    return int(max(float(value), 0.0) * float(_GIB))


def path_size_bytes(path: Path) -> int:
    candidate = Path(path)
    if not candidate.exists():
        return 0
    if candidate.is_file():
        return int(candidate.stat().st_size)
    total = 0
    for entry in candidate.rglob("*"):
        try:
            if entry.is_file():
                total += int(entry.stat().st_size)
        except OSError:
            continue
    return total


def _candidate_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    candidates: List[Path] = []
    for child in root.iterdir():
        if child.name in {".gitkeep"}:
            continue
        if child.name == "hf" and child.is_dir():
            candidates.extend(grandchild for grandchild in child.iterdir())
            continue
        candidates.append(child)
    return candidates


def enforce_dataset_cache_budget(
    *,
    roots: Sequence[Path | None],
    budget_bytes: int,
) -> Dict[str, object]:
    normalized_roots = [Path(root) for root in roots if root is not None]
    existing_roots = [root for root in normalized_roots if root.exists()]
    total_before = sum(path_size_bytes(root) for root in existing_roots)
    summary: Dict[str, object] = {
        "budget_bytes": int(max(budget_bytes, 0)),
        "before_bytes": int(total_before),
        "after_bytes": int(total_before),
        "deleted_bytes": 0,
        "pruned_paths": [],
    }
    if budget_bytes <= 0 or total_before <= budget_bytes:
        return summary

    candidates: List[tuple[float, int, Path]] = []
    for root in existing_roots:
        for candidate in _candidate_paths(root):
            size_bytes = path_size_bytes(candidate)
            if size_bytes <= 0:
                continue
            try:
                mtime = float(candidate.stat().st_mtime)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, size_bytes, candidate))
    candidates.sort(key=lambda item: (item[0], str(item[2])))

    deleted_bytes = 0
    pruned_paths: List[str] = []
    total_after = int(total_before)
    for _, size_bytes, candidate in candidates:
        if total_after <= budget_bytes:
            break
        try:
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            elif candidate.exists():
                candidate.unlink(missing_ok=True)
        except OSError:
            continue
        total_after -= int(size_bytes)
        deleted_bytes += int(size_bytes)
        pruned_paths.append(str(candidate))

    summary["after_bytes"] = int(max(total_after, 0))
    summary["deleted_bytes"] = int(deleted_bytes)
    summary["pruned_paths"] = pruned_paths
    return summary
