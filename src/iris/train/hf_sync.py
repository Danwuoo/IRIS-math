from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class HFSyncError(RuntimeError):
    pass


def _load_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise HFSyncError(
            "huggingface_hub is required for checkpoint/final repo sync."
        ) from error
    return HfApi()


def _coerce_patterns(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    return [str(value) for value in values if str(value).strip()]


@dataclass(frozen=True)
class HFRepoSpec:
    repo_id: str
    repo_type: str = "model"
    revision: str = "main"


def resolve_dataset_commit_sha(
    *,
    dataset_id: str,
    config_name: str | None,
    revision_hint: str,
    token: str | None = None,
) -> str:
    del config_name
    api = _load_hf_api()
    info = api.dataset_info(
        repo_id=str(dataset_id),
        revision=str(revision_hint or "main"),
        token=token,
        files_metadata=False,
    )
    sha = str(getattr(info, "sha", "")).strip()
    if not sha:
        raise HFSyncError(
            f"Unable to resolve immutable dataset revision for {dataset_id}@{revision_hint}."
        )
    return sha


def ensure_repo_exists(
    *,
    repo: HFRepoSpec,
    token: str | None,
    private: bool = True,
) -> None:
    api = _load_hf_api()
    api.create_repo(
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
        token=token,
        private=private,
        exist_ok=True,
    )


def snapshot_download(
    *,
    repo: HFRepoSpec,
    local_dir: Path,
    token: str | None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as error:
        raise HFSyncError(
            "huggingface_hub is required for checkpoint bootstrap download."
        ) from error
    downloaded_path = hf_snapshot_download(
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
        revision=repo.revision,
        token=token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=_coerce_patterns(allow_patterns),
    )
    return Path(downloaded_path)


def _bootstrap_allow_patterns(
    *,
    pointer_path_in_repo: str,
    pointer: Mapping[str, Any],
) -> list[str]:
    run_id = str(pointer.get("run_id", "")).strip()
    run_root = str(pointer.get("run_root", f"checkpoints/{run_id}")).strip()
    checkpoint_manifest_path = str(pointer.get("checkpoint_manifest_path", "")).strip()
    try:
        segment_id = int(pointer.get("segment_id", -1))
    except (TypeError, ValueError):
        segment_id = -1

    patterns: list[str] = [str(pointer_path_in_repo)]
    if run_id:
        patterns.append(f"latest/{run_id}.json")
    if run_root:
        patterns.extend(
            [
                f"{run_root}/segment_journal.jsonl",
                f"{run_root}/metrics.jsonl",
                f"{run_root}/data/**",
                f"{run_root}/tokenizer/**",
                f"{run_root}/hf_latest_pointer.json",
            ]
        )
    if checkpoint_manifest_path:
        patterns.append(checkpoint_manifest_path)
    if run_root and segment_id >= 0:
        patterns.append(f"{run_root}/checkpoints/payloads/segment_{segment_id:06d}/**")

    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        normalized = str(pattern).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _restore_downloaded_run_tree(*, source_root: Path, local_dir: Path) -> None:
    source_root = Path(source_root)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for child in source_root.iterdir():
        destination = local_dir / child.name
        if child.is_dir() and destination.exists() and destination.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
            shutil.rmtree(child, ignore_errors=True)
            continue
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination, ignore_errors=True)
            else:
                destination.unlink(missing_ok=True)
        shutil.move(str(child), str(destination))


def write_latest_pointer(
    *,
    output_path: Path,
    run_id: str,
    segment_id: int,
    checkpoint_manifest_path: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema": "iris.hf_checkpoint_pointer/v1",
        "run_id": str(run_id),
        "segment_id": int(segment_id),
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "run_root": f"checkpoints/{run_id}",
    }
    for key, value in dict(extra or {}).items():
        payload[str(key)] = value
    pointer_path = Path(output_path)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return pointer_path


def read_latest_pointer(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise HFSyncError(f"Latest pointer must be a JSON object: {path}")
    return payload


def upload_folder(
    *,
    repo: HFRepoSpec,
    folder_path: Path,
    path_in_repo: str,
    token: str | None,
    commit_message: str,
    allow_patterns: Sequence[str] | None = None,
) -> Any:
    api = _load_hf_api()
    return api.upload_folder(
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
        folder_path=str(folder_path),
        path_in_repo=str(path_in_repo),
        token=token,
        revision=repo.revision,
        allow_patterns=_coerce_patterns(allow_patterns),
        commit_message=commit_message,
    )


def upload_file(
    *,
    repo: HFRepoSpec,
    path_or_fileobj: Path,
    path_in_repo: str,
    token: str | None,
    commit_message: str,
) -> Any:
    api = _load_hf_api()
    return api.upload_file(
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
        path_or_fileobj=str(path_or_fileobj),
        path_in_repo=str(path_in_repo),
        token=token,
        revision=repo.revision,
        commit_message=commit_message,
    )


def sync_checkpoint_run(
    *,
    repo: HFRepoSpec,
    run_dir: Path,
    run_id: str,
    latest_pointer_path: Path,
    checkpoint_payload_root: Path | None = None,
    checkpoint_payload_roots: Mapping[str, Path | str | None] | None = None,
    token: str | None,
) -> None:
    ensure_repo_exists(repo=repo, token=token)
    run_dir = Path(run_dir)
    upload_folder(
        repo=repo,
        folder_path=run_dir,
        path_in_repo=f"checkpoints/{run_id}",
        token=token,
        commit_message=f"sync checkpoint run {run_id}",
    )
    default_payload_root = run_dir / "checkpoints" / "payloads"
    payload_roots: list[Path] = []
    if checkpoint_payload_root is not None:
        payload_roots.append(Path(checkpoint_payload_root))
    for root in dict(checkpoint_payload_roots or {}).values():
        if root is None:
            continue
        payload_roots.append(Path(root))
    uploaded_roots: set[str] = set()
    default_root_key = str(default_payload_root.resolve(strict=False))
    for external_payload_root in payload_roots:
        root_key = str(external_payload_root.resolve(strict=False))
        if root_key == default_root_key or root_key in uploaded_roots:
            continue
        uploaded_roots.add(root_key)
        if external_payload_root.exists():
            upload_folder(
                repo=repo,
                folder_path=external_payload_root,
                path_in_repo=f"checkpoints/{run_id}/checkpoints/payloads",
                token=token,
                commit_message=f"sync checkpoint payloads for {run_id}",
            )
    upload_file(
        repo=repo,
        path_or_fileobj=latest_pointer_path,
        path_in_repo=f"latest/{run_id}.json",
        token=token,
        commit_message=f"update latest pointer for {run_id}",
    )
    upload_file(
        repo=repo,
        path_or_fileobj=latest_pointer_path,
        path_in_repo="latest/latest.json",
        token=token,
        commit_message=f"update global latest pointer for {run_id}",
    )


def bootstrap_checkpoint_run(
    *,
    repo: HFRepoSpec,
    local_dir: Path,
    token: str | None,
    pointer_path_in_repo: str = "latest/latest.json",
) -> dict[str, Any] | None:
    download_root = Path(local_dir).parent / f"{Path(local_dir).name}_hf_bootstrap"
    try:
        shutil.rmtree(download_root, ignore_errors=True)
        download_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo=repo,
            local_dir=download_root,
            token=token,
            allow_patterns=[pointer_path_in_repo],
        )
    except Exception:
        shutil.rmtree(download_root, ignore_errors=True)
        return None
    try:
        pointer_path = download_root / Path(pointer_path_in_repo)
        if not pointer_path.exists():
            return None
        pointer = read_latest_pointer(pointer_path)
        run_root = str(pointer.get("run_root", f"checkpoints/{pointer.get('run_id', '')}")).strip()
        if not run_root:
            return None
        shutil.rmtree(download_root, ignore_errors=True)
        download_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo=repo,
            local_dir=download_root,
            token=token,
            allow_patterns=_bootstrap_allow_patterns(
                pointer_path_in_repo=pointer_path_in_repo,
                pointer=pointer,
            ),
        )
        source_root = download_root / Path(run_root)
        if not source_root.exists():
            raise HFSyncError(f"Downloaded checkpoint run root is missing: {source_root}")
        _restore_downloaded_run_tree(source_root=source_root, local_dir=Path(local_dir))
        return pointer
    finally:
        shutil.rmtree(download_root, ignore_errors=True)


def sync_final_release(
    *,
    repo: HFRepoSpec,
    release_dir: Path,
    token: str | None,
) -> None:
    ensure_repo_exists(repo=repo, token=token)
    upload_folder(
        repo=repo,
        folder_path=Path(release_dir),
        path_in_repo=".",
        token=token,
        commit_message="publish IRIS P1 3B final release",
    )
