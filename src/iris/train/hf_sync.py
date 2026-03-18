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
    token: str | None,
) -> None:
    ensure_repo_exists(repo=repo, token=token)
    upload_folder(
        repo=repo,
        folder_path=Path(run_dir),
        path_in_repo=f"checkpoints/{run_id}",
        token=token,
        commit_message=f"sync checkpoint run {run_id}",
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
    shutil.rmtree(download_root, ignore_errors=True)
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo=repo,
            local_dir=download_root,
            token=token,
            allow_patterns=[pointer_path_in_repo],
        )
    except Exception:
        return None
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
        allow_patterns=[pointer_path_in_repo, f"{run_root}/**", f"latest/{pointer.get('run_id', '')}.json"],
    )
    source_root = download_root / Path(run_root)
    if not source_root.exists():
        raise HFSyncError(f"Downloaded checkpoint run root is missing: {source_root}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, local_dir, dirs_exist_ok=True)
    return pointer


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
