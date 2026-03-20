from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

_SEGMENT_MANIFEST_RE = re.compile(r"^segment_(\d+)\.json$")
_SEGMENT_PAYLOAD_RE = re.compile(r"^segment_(\d+)$")
_PAYLOAD_COMPONENTS = ("params", "optimizer_state", "rng_state")


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


def checkpoint_payload_dir(
    checkpoint_dir: Path,
    segment_id: int,
    *,
    payload_root: Path | None = None,
) -> Path:
    root = Path(payload_root) if payload_root is not None else (Path(checkpoint_dir) / "payloads")
    return root / f"segment_{int(segment_id):06d}"


def _component_payload_root_map(
    checkpoint_dir: Path,
    *,
    payload_root: Path | None = None,
    payload_roots: Mapping[str, Path | str | None] | None = None,
) -> dict[str, Path]:
    default_root = Path(payload_root) if payload_root is not None else (Path(checkpoint_dir) / "payloads")
    root_map = {
        component: default_root
        for component in _PAYLOAD_COMPONENTS
    }
    for component, root in dict(payload_roots or {}).items():
        if component not in root_map or root is None:
            continue
        root_map[component] = Path(root)
    return root_map


def _iter_payload_root_values(payload_roots: Any) -> list[Path]:
    if payload_roots is None:
        return []
    if isinstance(payload_roots, Mapping):
        values: list[Path] = []
        for root in payload_roots.values():
            if root is None:
                continue
            if isinstance(root, (list, tuple, set)):
                values.extend(Path(candidate) for candidate in root if candidate is not None)
                continue
            values.append(Path(root))
        return values
    return [Path(root) for root in payload_roots if root is not None]


def _component_search_roots(
    checkpoint_dir: Path,
    *,
    payload_roots: Any = None,
) -> dict[str, tuple[Path, ...]]:
    default_root = Path(checkpoint_dir) / "payloads"
    search_roots = {
        component: [default_root]
        for component in _PAYLOAD_COMPONENTS
    }
    if isinstance(payload_roots, Mapping):
        for component in _PAYLOAD_COMPONENTS:
            root = payload_roots.get(component)
            if root is None:
                continue
            if isinstance(root, (list, tuple, set)):
                search_roots[component].extend(Path(candidate) for candidate in root if candidate is not None)
                continue
            search_roots[component].append(Path(root))
    else:
        generic_roots = _iter_payload_root_values(payload_roots)
        for component in _PAYLOAD_COMPONENTS:
            search_roots[component].extend(generic_roots)

    normalized: dict[str, tuple[Path, ...]] = {}
    for component, roots in search_roots.items():
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(Path(root).resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            unique.append(Path(root))
        normalized[component] = tuple(unique)
    return normalized


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


def _resolve_payload_ref(
    manifest_path: Path,
    ref: str,
    *,
    payload_roots: Sequence[Path] | None = None,
) -> Path:
    candidate = _resolve_ref(manifest_path, ref)
    if candidate.exists():
        return candidate
    raw_ref = Path(str(ref))
    if raw_ref.is_absolute():
        return candidate
    parts = raw_ref.parts
    if not parts or parts[0] != "payloads":
        return candidate
    suffix = Path(*parts[1:]) if len(parts) > 1 else Path()
    for root in payload_roots or ():
        alternate = Path(root) / suffix
        if alternate.exists():
            return alternate
    return candidate


def _unique_payload_roots(
    checkpoint_dir: Path,
    payload_roots: Any = None,
) -> tuple[Path, ...]:
    roots: list[Path] = [Path(checkpoint_dir) / "payloads"]
    roots.extend(_iter_payload_root_values(payload_roots))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return tuple(unique)


def prune_checkpoint_payloads(
    *,
    checkpoint_dir: Path,
    keep_last: int = 1,
    keep_segment_ids: tuple[int, ...] = (),
    payload_roots: Any = None,
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
    for payload_root in _unique_payload_roots(checkpoint_dir, payload_roots):
        if not payload_root.exists():
            continue
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
        "pruned_segment_ids": sorted({int(segment_id) for segment_id in pruned_segment_ids}),
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
    payload_root: Path | None = None,
    payload_roots: Mapping[str, Path | str | None] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_manifest_path(checkpoint_dir, segment_id)
    payload_dir = checkpoint_payload_dir(checkpoint_dir, segment_id, payload_root=payload_root)
    payload_root_map = _component_payload_root_map(
        checkpoint_dir,
        payload_root=payload_root,
        payload_roots=payload_roots,
    )
    params_path = checkpoint_payload_dir(
        checkpoint_dir,
        segment_id,
        payload_root=payload_root_map["params"],
    ) / "params"
    opt_state_path = checkpoint_payload_dir(
        checkpoint_dir,
        segment_id,
        payload_root=payload_root_map["optimizer_state"],
    ) / "optimizer_state"
    rng_state_path = checkpoint_payload_dir(
        checkpoint_dir,
        segment_id,
        payload_root=payload_root_map["rng_state"],
    ) / "rng_state"
    _save_pytree(params_path, params)
    _save_pytree(opt_state_path, opt_state)
    _save_pytree(rng_state_path, dict(rng_state))
    manifest = dict(manifest_payload)
    manifest.update(
        {
            "schema": "iris.training_checkpoint/v2",
            "checkpoint_kind": "orbax_sidecar",
            "checkpoint_payload_ref": str(Path("payloads") / payload_dir.name),
            "payload_format": "orbax.pytree/v1",
            "payload_refs": {
                "params": str(Path("payloads") / payload_dir.name / "params"),
                "optimizer_state": str(Path("payloads") / payload_dir.name / "optimizer_state"),
                "rng_state": str(Path("payloads") / payload_dir.name / "rng_state"),
            },
        }
    )
    return _write_json_atomic(manifest_path, manifest)


def load_iris3b_checkpoint(
    path: Path,
    *,
    payload_roots: Any = None,
) -> Dict[str, Any]:
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
    component_search_roots = _component_search_roots(
        manifest_path.parent,
        payload_roots=payload_roots,
    )
    resolved_refs = {
        name: _resolve_payload_ref(
            manifest_path,
            str(payload_refs.get(name, "")),
            payload_roots=component_search_roots.get(name, ()),
        )
        for name in _PAYLOAD_COMPONENTS
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
