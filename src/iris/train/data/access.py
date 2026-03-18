from __future__ import annotations

import fnmatch
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple
from zipfile import ZipFile

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


def _metadata_mapping(metadata: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, Mapping):
        return {}
    return {str(inner_key): inner_value for inner_key, inner_value in value.items()}


def _metadata_list(metadata: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = metadata.get(key)
    if value in (None, ""):
        return ()
    if isinstance(value, (str, Path)):
        normalized = str(value).strip()
        return (normalized,) if normalized else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _loader_kwargs(metadata: Mapping[str, Any], key: str) -> Dict[str, Any]:
    raw_kwargs = _metadata_mapping(metadata, key)
    return {str(inner_key): inner_value for inner_key, inner_value in raw_kwargs.items()}


def _resolve_split_metadata(metadata: Mapping[str, Any], key: str, split: str) -> Tuple[str, ...]:
    by_split = _metadata_mapping(metadata, key)
    if by_split:
        return _metadata_list(by_split, split)
    return _metadata_list(metadata, key)


def _builder_from_path(path: Path) -> Tuple[str, Dict[str, Any]]:
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    suffix_chain = "".join(suffixes[-3:])
    if ".parquet" in suffixes:
        return "parquet", {}
    if ".jsonl" in suffix_chain or ".json" in suffixes:
        return "json", {}
    if ".tsv" in suffixes:
        return "csv", {"delimiter": "\t"}
    if ".csv" in suffixes:
        return "csv", {}
    raise DatasetAccessError(f"Unsupported dataset file extension: {path}")


def _stable_snapshot_revision(revision: str) -> str:
    normalized = _runtime_revision(revision)
    return normalized.replace("/", "__").replace(":", "__")


def _hf_dataset_uri(repo_id: str, revision: str, file_path: str) -> str:
    return f"hf://datasets/{repo_id}@{_runtime_revision(revision)}/{file_path}"


def _runtime_revision(revision: str) -> str:
    normalized = str(revision or "main").strip() or "main"
    if normalized.startswith("resolve:"):
        normalized = normalized.split(":", 1)[1].strip() or "main"
    return normalized


def _list_hf_dataset_files(repo_id: str, revision: str) -> Tuple[str, ...]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise DatasetAccessError(
            "huggingface_hub is required to enumerate dataset files for file-based streaming fallback."
        ) from error
    entries = HfApi().list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=_runtime_revision(revision),
        recursive=True,
    )
    file_paths = []
    for entry in entries:
        entry_path = getattr(entry, "path", None)
        entry_type = getattr(entry, "type", None)
        if not entry_path:
            continue
        if entry_type is not None and str(entry_type).lower() != "file":
            continue
        file_paths.append(str(entry_path))
    return tuple(sorted(file_paths))


def _download_hf_dataset_file(
    repo_id: str,
    revision: str,
    filename: str,
    *,
    cache_dir: Path | None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise DatasetAccessError(
            "huggingface_hub is required to materialize dataset archives for local snapshot mode."
        ) from error
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=_runtime_revision(revision),
            filename=filename,
            cache_dir=(str(cache_dir) if cache_dir is not None else None),
        )
    )


def _resolve_hf_data_files(source: DatasetSourceSpec) -> Tuple[str, ...]:
    metadata = source.metadata
    explicit_files = _resolve_split_metadata(metadata, "hf_repo_files_by_split", source.split)
    if explicit_files:
        return explicit_files
    explicit_files = _metadata_list(metadata, "hf_repo_files")
    if explicit_files:
        return explicit_files

    patterns = _resolve_split_metadata(metadata, "hf_repo_file_patterns_by_split", source.split)
    if not patterns:
        patterns = _metadata_list(metadata, "hf_repo_file_patterns")
    if not patterns:
        raise DatasetAccessError(
            f"Source '{source.source_id}' is missing hf_repo_file_patterns metadata for file-based fallback."
        )

    repo_files = _list_hf_dataset_files(source.hf_path, source.revision)
    matched = [
        file_path
        for file_path in repo_files
        if any(fnmatch.fnmatch(file_path, pattern) for pattern in patterns)
    ]
    if not matched:
        raise DatasetAccessError(
            f"No dataset files matched {list(patterns)} for source '{source.source_id}' at revision {source.revision}."
        )
    return tuple(matched)


def _open_hf_file_fallback(
    source: DatasetSourceSpec,
    *,
    loader: LoadDatasetFn,
) -> Iterable[Mapping[str, Any]]:
    metadata = source.metadata
    builder = str(metadata.get("hf_file_fallback_builder", "")).strip().lower()
    if not builder:
        raise DatasetAccessError(
            f"Source '{source.source_id}' has no hf_file_fallback_builder configured."
        )
    file_paths = _resolve_hf_data_files(source)
    data_files = [
        _hf_dataset_uri(source.hf_path, source.revision, file_path)
        for file_path in file_paths
    ]
    loader_kwargs = {
        "data_files": data_files[0] if len(data_files) == 1 else data_files,
        "split": "train",
        "streaming": True,
    }
    loader_kwargs.update(_loader_kwargs(metadata, "hf_file_fallback_loader_kwargs"))
    return loader(builder, **loader_kwargs)


def _open_hf_stream(
    source: DatasetSourceSpec,
    *,
    loader: LoadDatasetFn,
) -> Iterable[Mapping[str, Any]]:
    if bool(source.metadata.get("hf_force_file_fallback", False)):
        # TEMPORARY TECHNICAL DEBT: force file-based routing for sources such as peS2o
        # whose repo-level iterator order mixes or front-loads disallowed source families.
        # Remove once manifest-resolved source-family partitioning can select compliant
        # document slices directly without relying on shard-order heuristics.
        return _open_hf_file_fallback(source, loader=loader)
    kwargs: Dict[str, Any] = {
        "split": source.split,
        "streaming": True,
        "revision": _runtime_revision(source.revision),
    }
    if source.hf_name:
        kwargs["name"] = source.hf_name
    try:
        return loader(source.hf_path, **kwargs)
    except Exception as repo_error:
        if "hf_file_fallback_builder" not in source.metadata:
            raise
        try:
            return _open_hf_file_fallback(source, loader=loader)
        except Exception as fallback_error:
            raise DatasetAccessError(
                f"repo-level load failed for source '{source.source_id}': {repo_error}; "
                f"file-based fallback failed: {fallback_error}"
            ) from fallback_error


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
        builder, loader_kwargs = _builder_from_path(snapshot_path)
        return loader(
            builder,
            data_files=str(snapshot_path),
            split="train",
            streaming=True,
            **loader_kwargs,
        )

    pattern = source.local_snapshot_pattern or "**/*.parquet"
    matches = sorted(snapshot_path.glob(pattern)) if snapshot_path.exists() else []
    if not matches:
        for fallback_pattern in (
            "**/*.parquet",
            "**/*.jsonl",
            "**/*.json",
            "**/*.json.gz",
            "**/*.jsonl.gz",
            "**/*.csv",
            "**/*.tsv",
        ):
            matches = sorted(snapshot_path.glob(fallback_pattern)) if snapshot_path.exists() else []
            if matches:
                break

    if not matches:
        raise DatasetAccessError(
            f"No local snapshot files found for source '{source.source_id}' under {snapshot_path}."
        )

    builder, loader_kwargs = _builder_from_path(matches[0])
    return loader(
        builder,
        data_files=[str(path) for path in matches],
        split="train",
        streaming=True,
        **loader_kwargs,
    )


def _materialize_snapshot_from_hf(
    source: DatasetSourceSpec,
    *,
    snapshot_root: Path,
) -> Path | None:
    archive_names = _metadata_list(source.metadata, "hf_snapshot_archives")
    if not archive_names:
        return None

    materialized_root = (
        Path(snapshot_root)
        / source.source_id
        / "_materialized"
        / _stable_snapshot_revision(source.revision)
    )
    completion_marker = materialized_root / ".materialized.json"
    if completion_marker.exists():
        return materialized_root

    extracted_root = materialized_root / "extracted"
    copied_root = materialized_root / "files"
    extracted_root.mkdir(parents=True, exist_ok=True)
    copied_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(snapshot_root) / "_hf_hub_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    materialized_files = []
    for archive_name in archive_names:
        archive_path = _download_hf_dataset_file(
            source.hf_path,
            source.revision,
            archive_name,
            cache_dir=cache_dir,
        )
        suffixes = [suffix.lower() for suffix in archive_path.suffixes]
        if ".zip" in suffixes:
            destination_dir = extracted_root / Path(archive_name).stem
            destination_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path) as handle:
                handle.extractall(destination_dir)
            materialized_files.append(str(destination_dir))
            continue
        destination_path = copied_root / Path(archive_name).name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not destination_path.exists():
            shutil.copy2(archive_path, destination_path)
        materialized_files.append(str(destination_path))

    completion_marker.write_text(
        json.dumps(
            {
                "schema": "iris.local_snapshot_materialization/v1",
                "source_id": source.source_id,
                "hf_path": source.hf_path,
                "revision": source.revision,
                "materialized_files": materialized_files,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return materialized_root


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

    materialized_snapshot = _materialize_snapshot_from_hf(
        source,
        snapshot_root=Path(snapshot_root),
    )
    if materialized_snapshot is not None:
        return _open_files_snapshot(source, snapshot_path=materialized_snapshot, loader=loader)

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
