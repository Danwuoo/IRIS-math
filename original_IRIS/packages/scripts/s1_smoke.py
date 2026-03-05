from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

import _bootstrap  # noqa: F401
from iris.levels import LevelInput, build_level_stack
from iris.metrics import build_canonical_metrics, neutral_failure_credit
from iris.runtime import assert_jax_runtime
from iris.train.synthetic import generate_synthetic_state
from iris.trunk import SingleTrunk


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 smoke check for IRIS skeleton.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    assert_jax_runtime(
        device=args.device,
        require_gpu=str(args.device).lower() == "gpu",
    )

    state = generate_synthetic_state(
        run_id="s1-smoke",
        dataset_slice_id="slice-smoke",
        segment_id=0,
        micro_step_idx=0,
        hidden_dim=args.hidden_dim,
        data_seed=13,
    )

    level_stack = build_level_stack(
        implementation="mounted",
        hidden_dim=args.hidden_dim,
        seed=0,
    )
    current_state = state
    for level_id in [f"L{idx}" for idx in range(7)]:
        level_output = level_stack[level_id].run(LevelInput(state_in=current_state))
        current_state = level_output.state_out

    trunk = SingleTrunk(hidden_dim=args.hidden_dim, seed=0, backend="jax")
    trunk_output = trunk.forward(current_state)
    if trunk_output.backend != "jax":
        print("S1 FAIL: strict JAX path was not used.", file=sys.stderr)
        return 1
    metrics = build_canonical_metrics(
        state=trunk_output.state_out,
        failure_credit=neutral_failure_credit(),
        task_validity_score=0.5,
        task_confidence=0.5,
        extra={
            "phase": "C",
            "suite": "S1",
            "device.requested": args.device,
            "trunk.backend": trunk_output.backend,
        },
    )

    sequence = trunk_output.state_out.to_canonical_sequence()
    if not np.isfinite(sequence).all():
        print("S1 FAIL: non-finite values detected.", file=sys.stderr)
        return 1
    if any(math.isnan(float(value)) for value in trunk_output.diagnostics.values()):
        print("S1 FAIL: NaN diagnostics detected.", file=sys.stderr)
        return 1

    report = {
        "status": "PASS",
        "suite": "S1",
        "trunk_backend": trunk_output.backend,
        "token_count": trunk_output.state_out.total_tokens,
        "metrics_subset": {
            "task.validity_score": metrics["task.validity_score"],
            "task.confidence": metrics["task.confidence"],
            "process.failure_distribution_entropy": metrics[
                "process.failure_distribution_entropy"
            ],
        },
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
