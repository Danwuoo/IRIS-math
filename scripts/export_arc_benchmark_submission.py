from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.arc import export_benchmark_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Export IRIS predictions into arc-agi-benchmarking submission schema.")
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("tools/arc-agi-benchmarking/data/sample/tasks"),
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path("artifacts/phase_e_benchmark/submissions/iris"),
    )
    parser.add_argument("--model-name", type=str, default="iris-phase-e")
    parser.add_argument("--provider-name", type=str, default="iris")
    parser.add_argument("--test-id", type=str, default="phase-e-bridge")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--attempts-per-pair", type=int, default=1)
    parser.add_argument("--max-reasoning-cycles", type=int, default=3)
    parser.add_argument("--termination-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])

    args = parser.parse_args()
    max_tasks = int(args.max_tasks) if int(args.max_tasks) > 0 else None

    try:
        summary = export_benchmark_submission(
            model_run_dir=args.model_run_dir,
            tasks_dir=args.tasks_dir,
            submission_dir=args.submission_dir,
            model_name=args.model_name,
            provider_name=args.provider_name,
            test_id=args.test_id,
            max_tasks=max_tasks,
            attempts_per_pair=args.attempts_per_pair,
            max_reasoning_cycles=args.max_reasoning_cycles,
            termination_threshold=args.termination_threshold,
            seed=args.seed,
            device=args.device,
        )
    except Exception as error:
        print(f"EXPORT FAILED: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
