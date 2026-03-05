from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .types import ArcExample, ArcTask, validate_grid


def _parse_example(raw: Mapping[str, object], *, field_prefix: str) -> ArcExample:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_prefix} example must be an object.")
    if "input" not in raw or "output" not in raw:
        raise ValueError(f"{field_prefix} example must contain 'input' and 'output'.")
    input_grid = validate_grid(raw["input"], field_name=f"{field_prefix}.input")
    output_grid = validate_grid(raw["output"], field_name=f"{field_prefix}.output")
    return ArcExample(input_grid=input_grid, output_grid=output_grid)


def _parse_examples(
    raw_examples: object,
    *,
    field_prefix: str,
) -> Tuple[ArcExample, ...]:
    if not isinstance(raw_examples, Sequence) or isinstance(raw_examples, (str, bytes, bytearray)):
        raise ValueError(f"{field_prefix} must be a list of examples.")
    examples = [
        _parse_example(example, field_prefix=f"{field_prefix}[{idx}]")
        for idx, example in enumerate(raw_examples)
    ]
    return tuple(examples)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file: {path}") from error


def _parse_arc_task_file(
    *,
    task_path: Path,
    source_name: str,
    concept_id: str | None,
) -> ArcTask:
    raw_payload = _load_json(task_path)
    train_examples: Tuple[ArcExample, ...] = tuple()
    test_examples: Tuple[ArcExample, ...] = tuple()

    if isinstance(raw_payload, Mapping):
        train_examples = _parse_examples(
            raw_payload.get("train", []),
            field_prefix=f"{task_path.name}.train",
        )
        test_examples = _parse_examples(
            raw_payload.get("test", []),
            field_prefix=f"{task_path.name}.test",
        )
    elif isinstance(raw_payload, Sequence) and not isinstance(raw_payload, (str, bytes, bytearray)):
        test_examples = _parse_examples(raw_payload, field_prefix=f"{task_path.name}.examples")
    else:
        raise ValueError(f"Unsupported ARC task payload type: {task_path}")

    if not train_examples and not test_examples:
        raise ValueError(f"No examples found in ARC task file: {task_path}")

    return ArcTask(
        task_id=task_path.stem,
        train_examples=train_examples,
        test_examples=test_examples,
        source_path=str(task_path),
        source_name=source_name,
        concept_id=concept_id,
    )


def load_conceptarc_tasks(corpus_root: Path) -> List[ArcTask]:
    corpus_root = Path(corpus_root)
    if not corpus_root.exists():
        raise ValueError(f"ConceptARC corpus path does not exist: {corpus_root}")
    if not corpus_root.is_dir():
        raise ValueError(f"ConceptARC corpus path is not a directory: {corpus_root}")

    tasks: List[ArcTask] = []
    for concept_dir in sorted(path for path in corpus_root.iterdir() if path.is_dir()):
        concept_id = concept_dir.name
        for task_file in sorted(concept_dir.glob("*.json")):
            tasks.append(
                _parse_arc_task_file(
                    task_path=task_file,
                    source_name="ConceptARC",
                    concept_id=concept_id,
                )
            )
    if not tasks:
        raise ValueError(f"No ConceptARC task files found in {corpus_root}.")
    return tasks


def load_rearc_tasks(tasks_root: Path) -> List[ArcTask]:
    tasks_root = Path(tasks_root)
    if not tasks_root.exists():
        raise ValueError(f"re_arc tasks path does not exist: {tasks_root}")
    if not tasks_root.is_dir():
        raise ValueError(f"re_arc tasks path is not a directory: {tasks_root}")

    tasks: List[ArcTask] = []
    for task_file in sorted(tasks_root.glob("*.json")):
        tasks.append(
            _parse_arc_task_file(
                task_path=task_file,
                source_name="re_arc",
                concept_id=None,
            )
        )
    if not tasks:
        raise ValueError(f"No re_arc task files found in {tasks_root}.")
    return tasks


def group_tasks_by_concept(tasks: Iterable[ArcTask]) -> Dict[str, List[ArcTask]]:
    grouped: Dict[str, List[ArcTask]] = {}
    for task in tasks:
        concept_id = task.concept_id or "UNKNOWN"
        grouped.setdefault(concept_id, []).append(task)
    return grouped

