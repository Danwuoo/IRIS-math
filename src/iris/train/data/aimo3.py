from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ...runtime import load_task_adjudication_policy_registry, resolve_task_adjudication_policy
from .policies import BenchmarkFamilyPolicy, DataPolicyBundle, load_default_policy_bundle

_AIMO3_DATASET_ID = "aimo-3-kaggle-official-v1"
_AIMO3_BENCHMARK_FAMILY_ID = "aimo-v1"
_AIMO3_REFERENCE_SPLIT = "reference"
_AIMO3_TEST_SPLIT = "test"
_REQUIRED_FILES: Tuple[str, ...] = (
    "reference.csv",
    "test.csv",
    "sample_submission.csv",
    "AIMO3_Reference_Problems.pdf",
)
_FULL_ITEM_ALLOWED_TIERS = {"Tier 2", "Tier 3"}


class AIMO3DatasetError(ValueError):
    pass


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_text(field_name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise AIMO3DatasetError(f"{field_name} is required.")
    return text


def _directory_manifest(path: Path) -> Tuple[Dict[str, str], ...]:
    files = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        files.append(
            {
                "relative_path": candidate.relative_to(path).as_posix(),
                "sha256": _sha256_bytes(candidate.read_bytes()),
            }
        )
    return tuple(files)


def _csv_rows(path: Path, *, required_columns: Sequence[str]) -> Tuple[Dict[str, str], ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AIMO3DatasetError(f"CSV file is missing a header row: {path}")
        fieldnames = tuple(str(field).strip() for field in reader.fieldnames if str(field).strip())
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise AIMO3DatasetError(
                f"CSV file {path} is missing required columns: {missing}."
            )
        rows = []
        for row in reader:
            normalized = {
                str(key).strip(): str(value).strip()
                for key, value in dict(row).items()
                if key is not None
            }
            rows.append(normalized)
    return tuple(rows)


def _validate_unique_ids(rows: Sequence[Mapping[str, str]], *, field_name: str, split_name: str) -> None:
    ids = [_require_text(f"{split_name}.{field_name}", row.get(field_name, "")) for row in rows]
    if len(set(ids)) != len(ids):
        raise AIMO3DatasetError(f"{split_name} contains duplicate {field_name} values.")


def _full_item_tier(
    tier: str,
    *,
    split_name: str,
    benchmark_policy: BenchmarkFamilyPolicy,
) -> str:
    normalized = _require_text(f"{split_name}.benchmark_tier", tier)
    if normalized not in set(benchmark_policy.allowed_tiers):
        raise AIMO3DatasetError(
            f"{split_name} benchmark tier {normalized!r} is not allowed by {benchmark_policy.benchmark_family_id}."
        )
    if normalized not in _FULL_ITEM_ALLOWED_TIERS:
        raise AIMO3DatasetError(
            f"{split_name} full official AIMO 3 items cannot use {normalized!r}. "
            "The active AIMO contract only allows Tier 1 structural labels/process fragments, "
            "not raw official problem statements or answer labels."
        )
    return normalized


def _default_task_family(benchmark_policy: BenchmarkFamilyPolicy) -> str:
    default_family = str(dict(benchmark_policy.default_task_family_map).get("default", "")).strip()
    return default_family or "answer_only"


@dataclass(frozen=True)
class AIMO3Item:
    item_id: str
    split: str
    problem: str
    answer: str | None
    benchmark_family_id: str
    benchmark_tier: str
    task_family: str
    task_adjudication_policy_id: str
    task_adjudication_policy_resolution_source: str
    benchmark_family_override_ref: str | None

    @property
    def answer_available(self) -> bool:
        return self.answer is not None

    def eval_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "item_id": self.item_id,
            "split": self.split,
            "problem": self.problem,
            "benchmark_family_id": self.benchmark_family_id,
            "benchmark_tier": self.benchmark_tier,
            "task_family": self.task_family,
            "task_adjudication_policy_id": self.task_adjudication_policy_id,
            "task_adjudication_policy_resolution_source": self.task_adjudication_policy_resolution_source,
            "benchmark_family_override_ref": self.benchmark_family_override_ref,
        }
        if self.answer is not None:
            payload["expected_answer"] = self.answer
        return payload


@dataclass(frozen=True)
class AIMO3LocalDataset:
    dataset_id: str
    root_path: Path
    benchmark_family_id: str
    source_snapshot_sha256: str
    reference_pdf_path: Path
    kaggle_gateway_path: Path | None
    reference_items: Tuple[AIMO3Item, ...]
    test_items: Tuple[AIMO3Item, ...]
    sample_submission_ids: Tuple[str, ...]

    @property
    def kaggle_gateway_present(self) -> bool:
        return self.kaggle_gateway_path is not None

    def split_items(self, split: str) -> Tuple[AIMO3Item, ...]:
        normalized = _require_text("split", split).lower()
        if normalized == _AIMO3_REFERENCE_SPLIT:
            return self.reference_items
        if normalized == _AIMO3_TEST_SPLIT:
            return self.test_items
        raise AIMO3DatasetError(
            f"Unsupported AIMO 3 split {split!r}. Expected 'reference' or 'test'."
        )

    def manifest(self) -> Dict[str, Any]:
        reference_tier = self.reference_items[0].benchmark_tier if self.reference_items else ""
        test_tier = self.test_items[0].benchmark_tier if self.test_items else ""
        return {
            "schema": "iris.local_benchmark_manifest/v1",
            "dataset_id": self.dataset_id,
            "benchmark_family_id": self.benchmark_family_id,
            "root_path": str(self.root_path),
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "reference_pdf_path": str(self.reference_pdf_path),
            "kaggle_gateway_present": self.kaggle_gateway_present,
            "splits": {
                "reference": {
                    "count": len(self.reference_items),
                    "benchmark_tier": reference_tier,
                    "answer_available_count": sum(1 for item in self.reference_items if item.answer_available),
                },
                "test": {
                    "count": len(self.test_items),
                    "benchmark_tier": test_tier,
                    "answer_available_count": sum(1 for item in self.test_items if item.answer_available),
                    "sample_submission_count": len(self.sample_submission_ids),
                },
            },
        }

    def eval_manifest(self, split: str) -> Dict[str, Any]:
        items = self.split_items(split)
        benchmark_tier = items[0].benchmark_tier if items else ""
        return {
            "schema": "iris.local_benchmark_eval_manifest/v1",
            "dataset_id": self.dataset_id,
            "split": str(split).lower(),
            "benchmark_family_id": self.benchmark_family_id,
            "benchmark_tier": benchmark_tier,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "reference_pdf_path": str(self.reference_pdf_path),
            "kaggle_gateway_present": self.kaggle_gateway_present,
            "items": [item.eval_payload() for item in items],
        }


def _snapshot_sha256(root_path: Path) -> str:
    entries = []
    for file_name in _REQUIRED_FILES:
        file_path = root_path / file_name
        entries.append(
            {
                "path": file_name,
                "sha256": _sha256_bytes(file_path.read_bytes()),
            }
        )
    kaggle_gateway_path = root_path / "kaggle_evaluation"
    if kaggle_gateway_path.exists() and kaggle_gateway_path.is_dir():
        entries.append(
            {
                "path": "kaggle_evaluation",
                "sha256": "sha256:" + _stable_sha256(_directory_manifest(kaggle_gateway_path)),
            }
        )
    return "sha256:" + _stable_sha256(entries)


def _build_item(
    row: Mapping[str, str],
    *,
    split_name: str,
    benchmark_tier: str,
    benchmark_policy: BenchmarkFamilyPolicy,
    task_family: str,
    task_adjudication_policy_id: str,
    task_adjudication_policy_resolution_source: str,
    benchmark_family_override_ref: str | None,
) -> AIMO3Item:
    answer = str(row.get("answer", "")).strip()
    return AIMO3Item(
        item_id=_require_text(f"{split_name}.id", row.get("id", "")),
        split=split_name,
        problem=_require_text(f"{split_name}.problem", row.get("problem", "")),
        answer=(answer or None),
        benchmark_family_id=benchmark_policy.benchmark_family_id,
        benchmark_tier=benchmark_tier,
        task_family=task_family,
        task_adjudication_policy_id=task_adjudication_policy_id,
        task_adjudication_policy_resolution_source=task_adjudication_policy_resolution_source,
        benchmark_family_override_ref=benchmark_family_override_ref,
    )


def load_aimo3_local_dataset(
    root_path: Path,
    *,
    policy_bundle: DataPolicyBundle | None = None,
    reference_tier: str = "Tier 2",
    test_tier: str = "Tier 3",
) -> AIMO3LocalDataset:
    root = Path(root_path)
    if not root.exists():
        raise AIMO3DatasetError(f"AIMO 3 dataset root does not exist: {root}")
    if not root.is_dir():
        raise AIMO3DatasetError(f"AIMO 3 dataset root must be a directory: {root}")

    for file_name in _REQUIRED_FILES:
        candidate = root / file_name
        if not candidate.exists() or not candidate.is_file():
            raise AIMO3DatasetError(f"AIMO 3 dataset is missing required file: {candidate}")

    resolved_bundle = policy_bundle or load_default_policy_bundle()
    benchmark_policy = dict(resolved_bundle.benchmark_family_policies).get(_AIMO3_BENCHMARK_FAMILY_ID)
    if benchmark_policy is None:
        raise AIMO3DatasetError(
            f"Policy bundle does not expose benchmark family {_AIMO3_BENCHMARK_FAMILY_ID!r}."
        )

    reference_benchmark_tier = _full_item_tier(
        reference_tier,
        split_name=_AIMO3_REFERENCE_SPLIT,
        benchmark_policy=benchmark_policy,
    )
    test_benchmark_tier = _full_item_tier(
        test_tier,
        split_name=_AIMO3_TEST_SPLIT,
        benchmark_policy=benchmark_policy,
    )

    reference_rows = _csv_rows(root / "reference.csv", required_columns=("id", "problem", "answer"))
    test_rows = _csv_rows(root / "test.csv", required_columns=("id", "problem"))
    sample_submission_rows = _csv_rows(root / "sample_submission.csv", required_columns=("id", "answer"))
    _validate_unique_ids(reference_rows, field_name="id", split_name=_AIMO3_REFERENCE_SPLIT)
    _validate_unique_ids(test_rows, field_name="id", split_name=_AIMO3_TEST_SPLIT)
    _validate_unique_ids(sample_submission_rows, field_name="id", split_name="sample_submission")
    for row in reference_rows:
        _require_text("reference.answer", row.get("answer", ""))

    sample_submission_ids = tuple(
        _require_text("sample_submission.id", row.get("id", "")) for row in sample_submission_rows
    )
    test_ids = tuple(_require_text("test.id", row.get("id", "")) for row in test_rows)
    if sample_submission_ids != test_ids:
        raise AIMO3DatasetError(
            "sample_submission.csv ids must match test.csv ids in order for local replay."
        )

    task_family = _default_task_family(benchmark_policy)
    policy_registry = load_task_adjudication_policy_registry()
    adjudication_policy, policy_source, override_ref = resolve_task_adjudication_policy(
        task_family,
        benchmark_family_policy=benchmark_policy,
        registry=policy_registry,
    )

    reference_items = tuple(
        _build_item(
            row,
            split_name=_AIMO3_REFERENCE_SPLIT,
            benchmark_tier=reference_benchmark_tier,
            benchmark_policy=benchmark_policy,
            task_family=task_family,
            task_adjudication_policy_id=adjudication_policy.task_adjudication_policy_id,
            task_adjudication_policy_resolution_source=policy_source,
            benchmark_family_override_ref=override_ref,
        )
        for row in reference_rows
    )
    test_items = tuple(
        _build_item(
            row,
            split_name=_AIMO3_TEST_SPLIT,
            benchmark_tier=test_benchmark_tier,
            benchmark_policy=benchmark_policy,
            task_family=task_family,
            task_adjudication_policy_id=adjudication_policy.task_adjudication_policy_id,
            task_adjudication_policy_resolution_source=policy_source,
            benchmark_family_override_ref=override_ref,
        )
        for row in test_rows
    )

    kaggle_gateway_path = root / "kaggle_evaluation"
    if not kaggle_gateway_path.exists() or not kaggle_gateway_path.is_dir():
        kaggle_gateway_path = None

    return AIMO3LocalDataset(
        dataset_id=_AIMO3_DATASET_ID,
        root_path=root.resolve(),
        benchmark_family_id=benchmark_policy.benchmark_family_id,
        source_snapshot_sha256=_snapshot_sha256(root),
        reference_pdf_path=(root / "AIMO3_Reference_Problems.pdf").resolve(),
        kaggle_gateway_path=(kaggle_gateway_path.resolve() if kaggle_gateway_path is not None else None),
        reference_items=reference_items,
        test_items=test_items,
        sample_submission_ids=sample_submission_ids,
    )
