from __future__ import annotations

import os
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


def _filesystem_device_id(path: Path) -> int | None:
    try:
        return int(os.stat(path).st_dev)
    except OSError:
        return None


def _free_bytes_by_device(paths: Sequence[Path]) -> Dict[int, int]:
    free_bytes: Dict[int, int] = {}
    for path in paths:
        device_id = _filesystem_device_id(path)
        if device_id is None:
            continue
        try:
            available = int(shutil.disk_usage(path).free)
        except OSError:
            continue
        if device_id not in free_bytes:
            free_bytes[device_id] = available
            continue
        free_bytes[device_id] = min(int(free_bytes[device_id]), available)
    return free_bytes


def _budget_satisfied(*, total_bytes: int, budget_bytes: int) -> bool:
    return int(budget_bytes) <= 0 or int(total_bytes) <= int(budget_bytes)


def _free_space_satisfied(*, free_bytes_by_device: Dict[int, int], min_free_bytes: int) -> bool:
    if int(min_free_bytes) <= 0:
        return True
    if not free_bytes_by_device:
        return True
    return min(int(value) for value in free_bytes_by_device.values()) >= int(min_free_bytes)


def enforce_dataset_cache_budget(
    *,
    roots: Sequence[Path | None],
    budget_bytes: int,
    min_free_bytes: int = 0,
    monitor_roots: Sequence[Path | None] = (),
) -> Dict[str, object]:
    normalized_roots = [Path(root) for root in roots if root is not None]
    existing_roots = [root for root in normalized_roots if root.exists()]
    total_before = sum(path_size_bytes(root) for root in existing_roots)
    monitored_paths = [Path(root) for root in (*roots, *monitor_roots) if root is not None]
    existing_monitored_paths = [path for path in monitored_paths if path.exists()]
    free_bytes_by_device = _free_bytes_by_device(existing_monitored_paths)
    free_bytes_before = min((int(value) for value in free_bytes_by_device.values()), default=0)
    summary: Dict[str, object] = {
        "budget_bytes": int(max(budget_bytes, 0)),
        "before_bytes": int(total_before),
        "after_bytes": int(total_before),
        "deleted_bytes": 0,
        "pruned_paths": [],
        "free_space_floor_bytes": int(max(min_free_bytes, 0)),
        "free_bytes_before": int(free_bytes_before),
        "free_bytes_after": int(free_bytes_before),
        "budget_guard_triggered": bool(int(budget_bytes) > 0 and int(total_before) > int(budget_bytes)),
        "low_disk_guard_triggered": bool(int(min_free_bytes) > 0 and int(free_bytes_before) < int(min_free_bytes)),
    }
    budget_ok = _budget_satisfied(total_bytes=total_before, budget_bytes=budget_bytes)
    free_ok = _free_space_satisfied(free_bytes_by_device=free_bytes_by_device, min_free_bytes=min_free_bytes)
    if budget_ok and free_ok:
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
        if _budget_satisfied(total_bytes=total_after, budget_bytes=budget_bytes) and _free_space_satisfied(
            free_bytes_by_device=free_bytes_by_device,
            min_free_bytes=min_free_bytes,
        ):
            break
        device_id = _filesystem_device_id(candidate)
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
        if device_id is not None and device_id in free_bytes_by_device:
            free_bytes_by_device[device_id] = int(free_bytes_by_device[device_id]) + int(size_bytes)

    summary["after_bytes"] = int(max(total_after, 0))
    summary["deleted_bytes"] = int(deleted_bytes)
    summary["pruned_paths"] = pruned_paths
    summary["free_bytes_after"] = min((int(value) for value in free_bytes_by_device.values()), default=0)
    return summary
