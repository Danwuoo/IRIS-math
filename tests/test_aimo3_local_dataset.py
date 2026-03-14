from __future__ import annotations

from pathlib import Path

import pytest

from iris.train import AIMO3DatasetError, load_aimo3_local_dataset


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_dataset_root(root: Path) -> Path:
    dataset_root = root / "AIMO 3"
    _write_text(
        dataset_root / "reference.csv",
        'id,problem,answer\nref001,"What is 1+1?",2\nref002,"What is 2+2?",4\n',
    )
    _write_text(
        dataset_root / "test.csv",
        'id,problem\n000aaa,"What is 3-3?"\n111bbb,"What is 5-5?"\n',
    )
    _write_text(
        dataset_root / "sample_submission.csv",
        "id,answer\n000aaa,0\n111bbb,0\n",
    )
    (dataset_root / "AIMO3_Reference_Problems.pdf").write_bytes(b"%PDF-1.4\n% bootstrap fixture\n")
    _write_text(dataset_root / "kaggle_evaluation" / "__init__.py", "# bootstrap\n")
    return dataset_root


def test_load_aimo3_local_dataset_builds_eval_manifests_from_local_root(tmp_path: Path) -> None:
    dataset_root = _make_dataset_root(tmp_path)

    dataset = load_aimo3_local_dataset(dataset_root)
    manifest = dataset.manifest()
    reference_eval = dataset.eval_manifest("reference")
    test_eval = dataset.eval_manifest("test")

    assert dataset.dataset_id == "aimo-3-kaggle-official-v1"
    assert dataset.benchmark_family_id == "aimo-v1"
    assert dataset.kaggle_gateway_present is True
    assert dataset.reference_items[0].benchmark_tier == "Tier 2"
    assert dataset.test_items[0].benchmark_tier == "Tier 3"
    assert dataset.reference_items[0].task_family == "answer_only"
    assert dataset.reference_items[0].task_adjudication_policy_id == "aimo-answer-only-tight-v1"
    assert dataset.reference_items[0].task_adjudication_policy_resolution_source == "benchmark_family_default"
    assert dataset.test_items[0].answer is None
    assert manifest["splits"]["reference"]["count"] == 2
    assert manifest["splits"]["test"]["sample_submission_count"] == 2
    assert reference_eval["items"][0]["expected_answer"] == "2"
    assert "expected_answer" not in test_eval["items"][0]


def test_load_aimo3_local_dataset_rejects_tier1_full_problem_exposure(tmp_path: Path) -> None:
    dataset_root = _make_dataset_root(tmp_path)

    with pytest.raises(AIMO3DatasetError):
        load_aimo3_local_dataset(dataset_root, reference_tier="Tier 1")

    with pytest.raises(AIMO3DatasetError):
        load_aimo3_local_dataset(dataset_root, test_tier="Tier 1")


def test_load_aimo3_local_dataset_rejects_sample_submission_mismatch(tmp_path: Path) -> None:
    dataset_root = _make_dataset_root(tmp_path)
    _write_text(dataset_root / "sample_submission.csv", "id,answer\nnot-test-id,0\n111bbb,0\n")

    with pytest.raises(AIMO3DatasetError):
        load_aimo3_local_dataset(dataset_root)
