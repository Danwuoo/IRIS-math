from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .inference import ArcDiagnosticRunner, ArcEvalConfig
from .types import ArcExample, ArcTask, validate_grid


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_example(raw: Mapping[str, Any], *, fallback_output: Sequence[Sequence[int]] | None = None) -> ArcExample:
    input_grid = validate_grid(raw.get("input", []), field_name="input")
    output_raw = raw.get("output")
    if output_raw is None and fallback_output is not None:
        output_raw = fallback_output
    if output_raw is None:
        raise ValueError("Benchmark task test example is missing required 'output' for offline scoring/export.")
    output_grid = validate_grid(output_raw, field_name="output")
    return ArcExample(input_grid=input_grid, output_grid=output_grid)


def load_benchmark_tasks(tasks_dir: Path, *, max_tasks: int | None = None) -> List[ArcTask]:
    root = Path(tasks_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Benchmark task directory does not exist: {root}")

    task_files = sorted(root.glob("*.json"))
    if max_tasks is not None and max_tasks > 0:
        task_files = task_files[: int(max_tasks)]
    tasks: List[ArcTask] = []

    for task_file in task_files:
        payload = json.loads(task_file.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Unsupported benchmark task payload: {task_file}")

        train_examples = tuple(
            _parse_example(example)
            for example in payload.get("train", [])
            if isinstance(example, Mapping)
        )
        test_examples = tuple(
            _parse_example(example)
            for example in payload.get("test", [])
            if isinstance(example, Mapping)
        )
        if not test_examples:
            raise ValueError(f"Task has no test examples: {task_file}")

        tasks.append(
            ArcTask(
                task_id=task_file.stem,
                train_examples=train_examples,
                test_examples=test_examples,
                source_path=str(task_file),
                source_name="arc-agi-benchmarking",
                concept_id=None,
            )
        )

    if not tasks:
        raise ValueError(f"No benchmark task files found in {root}")
    return tasks


def _build_attempt_payload(
    *,
    task_id: str,
    pair_index: int,
    answer: Sequence[Sequence[int]],
    model_name: str,
    provider_name: str,
    test_id: str,
) -> Dict[str, Any]:
    now = _utc_iso()
    return {
        "answer": [list(map(int, row)) for row in answer],
        "metadata": {
            "model": model_name,
            "provider": provider_name,
            "start_timestamp": now,
            "end_timestamp": now,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "IRIS benchmark prediction",
                    },
                }
            ],
            "reasoning_summary": None,
            "kwargs": {},
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
            "cost": {
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "reasoning_cost": None,
                "total_cost": 0.0,
            },
            "task_id": task_id,
            "pair_index": int(pair_index),
            "test_id": test_id,
        },
        "correct": None,
    }


def export_benchmark_submission(
    *,
    model_run_dir: Path,
    tasks_dir: Path,
    submission_dir: Path,
    model_name: str = "iris-phase-e",
    provider_name: str = "iris",
    test_id: str = "phase-e-bridge",
    max_tasks: int | None = None,
    attempts_per_pair: int = 1,
    max_reasoning_cycles: int = 3,
    termination_threshold: float = 0.7,
    seed: int = 17,
    device: str = "cpu",
) -> Dict[str, Any]:
    tasks = load_benchmark_tasks(Path(tasks_dir), max_tasks=max_tasks)
    config = ArcEvalConfig(
        model_run_dir=Path(model_run_dir),
        max_reasoning_cycles=int(max_reasoning_cycles),
        termination_threshold=float(termination_threshold),
        seed=int(seed),
        device=device,
    )
    runner = ArcDiagnosticRunner(config=config)

    output_root = Path(submission_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    files_written: List[str] = []
    total_pairs = 0
    for task in tasks:
        test_pairs_payload: List[Dict[str, Any]] = []
        for pair_index in range(len(task.test_examples)):
            record = runner.evaluate_case(task=task, case_index=pair_index, mode="benchmark.export")
            pair_payload: Dict[str, Any] = {}
            for attempt_index in range(max(int(attempts_per_pair), 1)):
                pair_payload[f"attempt_{attempt_index + 1}"] = _build_attempt_payload(
                    task_id=task.task_id,
                    pair_index=pair_index,
                    answer=record.predicted_grid,
                    model_name=model_name,
                    provider_name=provider_name,
                    test_id=test_id,
                )
            test_pairs_payload.append(pair_payload)
            total_pairs += 1

        output_path = output_root / f"{task.task_id}.json"
        output_path.write_text(json.dumps(test_pairs_payload, sort_keys=True, indent=2), encoding="utf-8")
        files_written.append(str(output_path))

    return {
        "status": "PASS",
        "tasks_exported": len(tasks),
        "pairs_exported": total_pairs,
        "submission_dir": str(output_root),
        "files": files_written,
    }
