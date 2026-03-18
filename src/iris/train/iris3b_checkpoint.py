from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping


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
    params = _restore_pytree(_resolve_ref(manifest_path, str(payload_refs.get("params", ""))))
    opt_state = _restore_pytree(_resolve_ref(manifest_path, str(payload_refs.get("optimizer_state", ""))))
    rng_state = _restore_pytree(_resolve_ref(manifest_path, str(payload_refs.get("rng_state", ""))))
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
