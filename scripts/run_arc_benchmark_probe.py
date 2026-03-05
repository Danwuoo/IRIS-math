from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import _bootstrap  # noqa: F401
from iris.arc import export_benchmark_submission


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_benchmark_import() -> Any:
    root = Path(__file__).resolve().parents[1]
    vendor_src = root / "tools" / "arc-agi-benchmarking" / "src"
    if str(vendor_src) not in sys.path:
        sys.path.insert(0, str(vendor_src))
    from arc_agi_benchmarking.scoring.scoring import ARCScorer

    return ARCScorer


def _attempt_payload(*, task_id: str, pair_index: int, answer: list[list[int]], test_id: str) -> Dict[str, Any]:
    now = _utc_iso()
    return {
        "answer": answer,
        "metadata": {
            "model": "baseline-random",
            "provider": "baseline",
            "start_timestamp": now,
            "end_timestamp": now,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "random baseline"},
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


def _random_grid(rows: int, cols: int, rng: random.Random) -> list[list[int]]:
    return [[int(rng.randrange(0, 10)) for _ in range(cols)] for _ in range(rows)]


def _build_random_baseline_submission(*, tasks_dir: Path, submission_dir: Path, seed: int) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    submission_dir.mkdir(parents=True, exist_ok=True)

    files_written = []
    pair_count = 0
    for task_file in sorted(Path(tasks_dir).glob("*.json")):
        payload = json.loads(task_file.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            continue
        rows = []
        tests = payload.get("test", [])
        if not isinstance(tests, list):
            continue
        for pair_index, test_case in enumerate(tests):
            if not isinstance(test_case, Mapping):
                continue
            target = test_case.get("output")
            if not isinstance(target, list):
                target = test_case.get("input", [])
            grid_rows = len(target) if isinstance(target, list) else 1
            grid_cols = len(target[0]) if grid_rows > 0 and isinstance(target[0], list) else 1
            answer = _random_grid(max(grid_rows, 1), max(grid_cols, 1), rng)
            rows.append(
                {
                    "attempt_1": _attempt_payload(
                        task_id=task_file.stem,
                        pair_index=pair_index,
                        answer=answer,
                        test_id="baseline-random",
                    )
                }
            )
            pair_count += 1

        output_file = submission_dir / task_file.name
        output_file.write_text(json.dumps(rows, sort_keys=True, indent=2), encoding="utf-8")
        files_written.append(str(output_file))

    return {
        "status": "PASS",
        "tasks_exported": len(files_written),
        "pairs_exported": pair_count,
        "submission_dir": str(submission_dir),
        "files": files_written,
    }


def _score_submission(
    *,
    scorer_cls: Any,
    tasks_dir: Path,
    submission_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    scorer = scorer_cls(
        task_dir=str(tasks_dir),
        submission_dir=str(submission_dir),
        print_logs=False,
        results_dir=str(results_dir),
    )
    total_score, total_tasks = scorer.score_submission()
    normalized_score = float(total_score) / float(total_tasks) if total_tasks > 0 else 0.0
    result_path = Path(results_dir) / "results.json"
    payload: Dict[str, Any] = {
        "total_score": float(total_score),
        "total_tasks": int(total_tasks),
        "normalized_score": normalized_score,
        "results_json_path": str(result_path),
    }
    if result_path.exists():
        try:
            payload["results"] = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            payload["results"] = {}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run arc-agi-benchmarking probe (baseline + IRIS bridge).")
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("tools/arc-agi-benchmarking/data/sample/tasks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase_e_benchmark/probe"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--attempts-per-pair", type=int, default=1)
    parser.add_argument("--max-reasoning-cycles", type=int, default=3)
    parser.add_argument("--termination-threshold", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--hard-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when probe status is FAIL.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ARCScorer = _ensure_benchmark_import()

    baseline_submission_dir = output_dir / "submission_baseline"
    iris_submission_dir = output_dir / "submission_iris"
    baseline_results_dir = output_dir / "scoring_baseline"
    iris_results_dir = output_dir / "scoring_iris"

    baseline_export = _build_random_baseline_submission(
        tasks_dir=args.tasks_dir,
        submission_dir=baseline_submission_dir,
        seed=args.seed,
    )
    baseline_scoring = _score_submission(
        scorer_cls=ARCScorer,
        tasks_dir=args.tasks_dir,
        submission_dir=baseline_submission_dir,
        results_dir=baseline_results_dir,
    )

    max_tasks = int(args.max_tasks) if int(args.max_tasks) > 0 else None
    iris_export = export_benchmark_submission(
        model_run_dir=args.model_run_dir,
        tasks_dir=args.tasks_dir,
        submission_dir=iris_submission_dir,
        model_name="iris-phase-e",
        provider_name="iris",
        test_id="phase-e-probe",
        max_tasks=max_tasks,
        attempts_per_pair=args.attempts_per_pair,
        max_reasoning_cycles=args.max_reasoning_cycles,
        termination_threshold=args.termination_threshold,
        seed=args.seed,
        device=args.device,
    )
    iris_scoring = _score_submission(
        scorer_cls=ARCScorer,
        tasks_dir=args.tasks_dir,
        submission_dir=iris_submission_dir,
        results_dir=iris_results_dir,
    )

    baseline_score = float(baseline_scoring.get("normalized_score", 0.0))
    iris_score = float(iris_scoring.get("normalized_score", 0.0))
    baseline_non_regression = iris_score >= baseline_score

    block_reasons = []
    if not baseline_non_regression:
        block_reasons.append(
            f"IRIS normalized score {iris_score:.6f} is below baseline {baseline_score:.6f}."
        )

    status = "PASS" if not block_reasons else "FAIL"
    artifact = {
        "schema": "iris.arc_benchmark_probe/v1",
        "generated_at_utc": _utc_iso(),
        "status": status,
        "probe_a_baseline": {
            "export": baseline_export,
            "scoring": baseline_scoring,
        },
        "probe_b_iris": {
            "export": iris_export,
            "scoring": iris_scoring,
        },
        "baseline_non_regression": baseline_non_regression,
        "block_reasons": block_reasons,
    }

    artifact_path = output_dir / "arc_benchmark_probe.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "artifact_path": str(artifact_path),
                "baseline_non_regression": baseline_non_regression,
                "baseline_score": baseline_score,
                "iris_score": iris_score,
            },
            sort_keys=True,
        )
    )
    if args.hard_fail and status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
