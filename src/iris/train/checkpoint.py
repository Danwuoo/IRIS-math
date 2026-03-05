from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _encode_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _encode_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_json(item) for item in value]
    return value


def _decode_json(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        return np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {key: _decode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    return value


def checkpoint_path(checkpoint_dir: Path, segment_id: int) -> Path:
    return checkpoint_dir / f"segment_{segment_id:06d}.json"


def _fsync_directory(directory: Path) -> None:
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def save_checkpoint_atomic(
    checkpoint_dir: Path,
    segment_id: int,
    payload: Dict[str, Any],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_path(checkpoint_dir, segment_id)
    temp = target.with_suffix(".tmp")

    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_encode_json(payload), handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temp, target)
    _fsync_directory(checkpoint_dir)
    return target


def load_checkpoint(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _decode_json(payload)
