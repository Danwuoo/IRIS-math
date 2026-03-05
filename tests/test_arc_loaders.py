from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.arc import load_conceptarc_tasks, load_rearc_tasks


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_conceptarc_tasks_parses_train_test_schema(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "corpus" / "Copy" / "copy_a.json",
        {
            "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 2]], "output": [[2, 2]]}],
        },
    )

    tasks = load_conceptarc_tasks(tmp_path / "corpus")
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "copy_a"
    assert task.source_name == "ConceptARC"
    assert task.concept_id == "Copy"
    assert len(task.train_examples) == 1
    assert len(task.test_examples) == 1


def test_load_rearc_tasks_parses_list_schema(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "tasks" / "pair_task.json",
        [
            {"input": [[1]], "output": [[1]]},
            {"input": [[2]], "output": [[2]]},
        ],
    )
    tasks = load_rearc_tasks(tmp_path / "tasks")
    assert len(tasks) == 1
    task = tasks[0]
    assert task.source_name == "re_arc"
    assert len(task.train_examples) == 0
    assert len(task.test_examples) == 2


def test_loaders_reject_missing_example_fields(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "corpus" / "Bad" / "bad.json",
        {
            "train": [{"input": [[1, 1]]}],
            "test": [{"input": [[2, 2]], "output": [[2, 2]]}],
        },
    )
    with pytest.raises(ValueError):
        load_conceptarc_tasks(tmp_path / "corpus")


def test_loaders_reject_invalid_grid_values(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "tasks" / "bad_grid.json",
        [
            {"input": [[1]], "output": [[10]]},
            {"input": [[2]], "output": [[2]]},
        ],
    )
    with pytest.raises(ValueError):
        load_rearc_tasks(tmp_path / "tasks")
