from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

_SEGMENT_MANIFEST_RE = re.compile(r"^segment_(\d+)\.json$")
_SEGMENT_PAYLOAD_RE = re.compile(r"^segment_(\d+)$")


def _require_checkpoint_stack() -> Any:
    try:
        import orbax.checkpoint as ocp
    except Exception as error:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "IRIS 3B checkpointing requires orbax-checkpoint and tensorstore."
        ) from error
    return ocp


def _encode_json(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is expected but optional in import path
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _encode_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_json(item) for item in value]
    return value


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


def checkpoint_manifest_path(checkpoint_dir: Path, segment_id: int) -> Path:
    return Path(checkpoint_dir) / f"segment_{int(segment_id):06d}.json"


def checkpoint_payload_dir(checkpoint_dir: Path, segment_id: int) -> Path:
    return Path(checkpoint_dir) / "payloads" / f"segment_{int(segment_id):06d}"


def _segment_id_from_name(name: str, *, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(str(name).strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _path_size_bytes(path: Path) -> int:
    candidate = Path(path)
    if not candidate.exists():
        return 0
    if candidate.is_file():
        try:
            return int(candidate.stat().st_size)
        except OSError:
            return 0
    total = 0
    for entry in candidate.rglob("*"):
        try:
            if entry.is_file():
                total += int(entry.stat().st_size)
        except OSError:
            continue
    return total


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(_encode_json(dict(payload)), handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    _fsync_directory(path.parent)
    return path


def _save_pytree(path: Path, tree: Any) -> None:
    ocp = _require_checkpoint_stack()
    checkpointer = ocp.PyTreeCheckpointer()
    shutil.rmtree(path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer.save(str(path), tree, force=True)


def _restore_pytree(path: Path) -> Any:
    ocp = _require_checkpoint_stack()
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(str(path))


def _resolve_ref(manifest_path: Path, ref: str) -> Path:
    candidate = Path(str(ref))
    if candidate.is_absolute():
        return candidate
    return manifest_path.parent / candidate


def prune_checkpoint_payloads(
    *,
    checkpoint_dir: Path,
    keep_last: int = 1,
    keep_segment_ids: tuple[int, ...] = (),
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    retention_limit = int(max(keep_last, 0))
    if retention_limit <= 0:
        return {
            "retention_limit": 0,
            "kept_segment_ids": [],
            "pruned_segment_ids": [],
            "deleted_bytes": 0,
            "pruned_payload_paths": [],
        }

    manifest_segment_ids = sorted(
        segment_id
        for segment_id in (
            _segment_id_from_name(path.name, pattern=_SEGMENT_MANIFEST_RE)
            for path in checkpoint_dir.glob("segment_*.json")
        )
        if segment_id is not None
    )
    keep_ids = set(manifest_segment_ids[-retention_limit:])
    keep_ids.update(int(segment_id) for segment_id in keep_segment_ids if int(segment_id) >= 0)

    deleted_bytes = 0
    pruned_segment_ids: list[int] = []
    pruned_payload_paths: list[str] = []
    payload_root = checkpoint_dir / "payloads"
    if payload_root.exists():
        for child in sorted(payload_root.iterdir(), key=lambda item: item.name):
            segment_id = _segment_id_from_name(child.name, pattern=_SEGMENT_PAYLOAD_RE)
            if segment_id is None or segment_id in keep_ids:
                continue
            size_bytes = _path_size_bytes(child)
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                elif child.exists():
                    child.unlink(missing_ok=True)
            except OSError:
                continue
            deleted_bytes += int(size_bytes)
            pruned_segment_ids.append(int(segment_id))
            pruned_payload_paths.append(str(child))

    return {
        "retention_limit": retention_limit,
        "kept_segment_ids": sorted(int(segment_id) for segment_id in keep_ids),
        "pruned_segment_ids": sorted(pruned_segment_ids),
        "deleted_bytes": int(deleted_bytes),
        "pruned_payload_paths": pruned_payload_paths,
    }


def save_iris3b_checkpoint(
    *,
    checkpoint_dir: Path,
    segment_id: int,
    params: Any,
    opt_state: Any,
    rng_state: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_manifest_path(checkpoint_dir, segment_id)
    payload_root = checkpoint_payload_dir(checkpoint_dir, segment_id)
    params_path = payload_root / "params"
    opt_state_path = payload_root / "optimizer_state"
    rng_state_path = payload_root / "rng_state"
    _save_pytree(params_path, params)
    _save_pytree(opt_state_path, opt_state)
    _save_pytree(rng_state_path, dict(rng_state))
    manifest = dict(manifest_payload)
    manifest.update(
        {
            "schema": "iris.training_checkpoint/v2",
            "checkpoint_kind": "orbax_sidecar",
            "checkpoint_payload_ref": str(Path("payloads") / payload_root.name),
            "payload_format": "orbax.pytree/v1",
            "payload_refs": {
                "params": str(Path("payloads") / payload_root.name / "params"),
                "optimizer_state": str(Path("payloads") / payload_root.name / "optimizer_state"),
                "rng_state": str(Path("payloads") / payload_root.name / "rng_state"),
            },
        }
    )
    return _write_json_atomic(manifest_path, manifest)


def load_iris3b_checkpoint(path: Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint manifest must be a JSON object: {manifest_path}")
    if "model_state" in payload:
        return dict(payload)
    if str(payload.get("schema", "")) != "iris.training_checkpoint/v2":
        raise RuntimeError(f"Unsupported checkpoint schema in {manifest_path}: {payload.get('schema')!r}")
    payload_refs = dict(payload.get("payload_refs", {}))
    if not payload_refs:
        raise RuntimeError(f"Checkpoint manifest is missing payload_refs: {manifest_path}")
    resolved_refs = {
        name: _resolve_ref(manifest_path, str(payload_refs.get(name, "")))
        for name in ("params", "optimizer_state", "rng_state")
    }
    missing_refs = {name: str(candidate) for name, candidate in resolved_refs.items() if not candidate.exists()}
    if missing_refs:
        missing_desc = ", ".join(f"{name}={candidate}" for name, candidate in sorted(missing_refs.items()))
        raise RuntimeError(
            "Checkpoint payload is missing or was pruned by retention policy for "
            f"{manifest_path}: {missing_desc}"
        )
    params = _restore_pytree(resolved_refs["params"])
    opt_state = _restore_pytree(resolved_refs["optimizer_state"])
    rng_state = _restore_pytree(resolved_refs["rng_state"])
    restored = dict(payload)
    restored["params"] = params
    restored["opt_state"] = opt_state
    restored["rng_state"] = rng_state
    restored["checkpoint_manifest_path"] = str(manifest_path)
    return restored


def export_params_msgpack(*, params: Any, output_path: Path) -> Path:
    try:
        from flax import serialization as flax_serialization
    except Exception as error:  # pragma: no cover - optional runtime
        raise RuntimeError("Flax serialization is required to export final params.") from error
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(flax_serialization.to_bytes(params))
    return output_path
