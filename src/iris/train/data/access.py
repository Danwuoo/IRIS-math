from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

from .contracts import DatasetSourceSpec

StreamingMode = str
_ALLOWED_STREAMING_MODES = {"auto", "hf_online", "local_snapshot"}


class DatasetAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceAccessResult:
    source_id: str
    iterable: Iterable[Mapping[str, Any]]
    effective_mode: str


LoadDatasetFn = Callable[..., Iterable[Mapping[str, Any]]]


def _default_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise DatasetAccessError(
            "datasets package is required for streaming pretrain mode. Install datasets first."
        ) from error
    return load_dataset(*args, **kwargs)


def _normalize_streaming_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized not in _ALLOWED_STREAMING_MODES:
        raise DatasetAccessError(
            "streaming_mode must be one of auto|hf_online|local_snapshot"
        )
    return normalized


def _open_hf_stream(
    source: DatasetSourceSpec,
    *,
    loader: LoadDatasetFn,
) -> Iterable[Mapping[str, Any]]:
    kwargs: Dict[str, Any] = {
        "split": source.split,
        "streaming": True,
        "revision": source.revision,
    }
    if source.hf_name:
        kwargs["name"] = source.hf_name
    return loader(source.hf_path, **kwargs)


def _snapshot_candidates(snapshot_root: Path, source: DatasetSourceSpec) -> Tuple[Path, ...]:
    root = Path(snapshot_root)
    direct = root / source.source_id
    by_path = root / source.hf_path
    by_flat_path = root / source.hf_path.replace("/", "__")
    return (direct, by_path, by_flat_path)


def _open_files_snapshot(
    source: DatasetSourceSpec,
    *,
    snapshot_path: Path,
    loader: LoadDatasetFn,
) -> Iterable[Mapping[str, Any]]:
    if snapshot_path.is_file():
        suffix = snapshot_path.suffix.lower()
        if suffix == ".parquet":
            return loader("parquet", data_files=str(snapshot_path), split="train", streaming=True)
        if suffix in {".json", ".jsonl"}:
            return loader("json", data_files=str(snapshot_path), split="train", streaming=True)

    pattern = source.local_snapshot_pattern or "**/*.parquet"
    matches = sorted(snapshot_path.glob(pattern)) if snapshot_path.exists() else []
    if not matches:
        # Try common fallback formats.
        for fallback_pattern in ("**/*.parquet", "**/*.jsonl", "**/*.json"):
            matches = sorted(snapshot_path.glob(fallback_pattern)) if snapshot_path.exists() else []
            if matches:
                break

    if not matches:
        raise DatasetAccessError(
            f"No local snapshot files found for source '{source.source_id}' under {snapshot_path}."
        )

    extension = matches[0].suffix.lower()
    if extension == ".parquet":
        builder = "parquet"
    elif extension in {".json", ".jsonl"}:
        builder = "json"
    else:
        raise DatasetAccessError(
            f"Unsupported local snapshot extension for source '{source.source_id}': {extension}"
        )

    return loader(builder, data_files=[str(path) for path in matches], split="train", streaming=True)


def _open_local_snapshot(
    source: DatasetSourceSpec,
    *,
    snapshot_root: Path | None,
    loader: LoadDatasetFn,
) -> Iterable[Mapping[str, Any]]:
    if snapshot_root is None:
        raise DatasetAccessError(
            f"Local snapshot mode requested for source '{source.source_id}' but snapshot_root is missing."
        )

    for candidate in _snapshot_candidates(Path(snapshot_root), source):
        if not candidate.exists():
            continue
        if candidate.is_dir() or candidate.is_file():
            return _open_files_snapshot(source, snapshot_path=candidate, loader=loader)

    raise DatasetAccessError(
        f"No local snapshot directory found for source '{source.source_id}' under {snapshot_root}."
    )


def open_streaming_source(
    source: DatasetSourceSpec,
    *,
    streaming_mode: StreamingMode,
    snapshot_root: Path | None = None,
    loader: LoadDatasetFn | None = None,
) -> SourceAccessResult:
    loader = loader or _default_loader
    normalized_mode = _normalize_streaming_mode(streaming_mode)

    hf_error: Exception | None = None
    local_error: Exception | None = None

    if normalized_mode in {"auto", "hf_online"}:
        try:
            iterable = _open_hf_stream(source, loader=loader)
            return SourceAccessResult(source_id=source.source_id, iterable=iterable, effective_mode="hf_online")
        except Exception as error:
            hf_error = error
            if normalized_mode == "hf_online":
                raise DatasetAccessError(
                    f"Failed to open HuggingFace streaming source '{source.source_id}': {error}"
                ) from error

    if normalized_mode in {"auto", "local_snapshot"}:
        try:
            iterable = _open_local_snapshot(source, snapshot_root=snapshot_root, loader=loader)
            return SourceAccessResult(source_id=source.source_id, iterable=iterable, effective_mode="local_snapshot")
        except Exception as error:
            local_error = error
            if normalized_mode == "local_snapshot":
                raise DatasetAccessError(
                    f"Failed to open local snapshot source '{source.source_id}': {error}"
                ) from error

    raise DatasetAccessError(
        "Unable to access source '{source_id}' in auto mode. "
        "hf_online_error={hf_error}; local_snapshot_error={local_error}".format(
            source_id=source.source_id,
            hf_error=hf_error,
            local_error=local_error,
        )
    )
