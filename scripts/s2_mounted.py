from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import _bootstrap  # noqa: F401
from iris.levels import LevelInput, build_level_stack
from iris.runtime import assert_jax_runtime
from iris.schema import STATE_IR_TOKEN_ORDER, StateIR


def _build_state(hidden_dim: int) -> StateIR:
    zeros = lambda rows: np.zeros((rows, hidden_dim), dtype=np.float32)
    return StateIR(T=zeros(1), G=zeros(1), O=zeros(2), R=zeros(1), X=zeros(1), M=zeros(1))


def _validate_l6_credit(diagnostics: dict) -> bool:
    credit = diagnostics.get("failure.credit")
    if not isinstance(credit, dict):
        return False
    keys = sorted(credit.keys())
    expected = [f"L{i}" for i in range(7)]
    if keys != expected:
        return False
    values = np.asarray([float(credit[level]) for level in expected], dtype=np.float64)
    return bool(np.all(values >= 0.0) and np.all(values <= 1.0) and np.isclose(np.sum(values), 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="S2M mounted structural check for IRIS.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    assert_jax_runtime(
        device=args.device,
        require_gpu=str(args.device).lower() == "gpu",
    )

    state = _build_state(hidden_dim=args.hidden_dim)
    stack = build_level_stack(
        implementation="mounted",
        hidden_dim=args.hidden_dim,
        seed=0,
    )
    if sorted(stack.keys()) != [f"L{i}" for i in range(7)]:
        print("S2M FAIL: L0-L6 interfaces are incomplete.", file=sys.stderr)
        return 1

    current_state = state
    for level_id in [f"L{i}" for i in range(7)]:
        output = stack[level_id].run(LevelInput(state_in=current_state))
        current_state = output.state_out
        if output.diagnostics.get("disabled", True):
            print(f"S2M FAIL: {level_id} should be mounted (disabled=False).", file=sys.stderr)
            return 1
        if output.control_out.get("mode") != "mounted":
            print(f"S2M FAIL: {level_id} did not emit mounted control mode.", file=sys.stderr)
            return 1
        logits = output.control_out.get("level_invocation_logits", [])
        if len(logits) != 7 or not np.isfinite(np.asarray(logits, dtype=np.float32)).all():
            print(f"S2M FAIL: {level_id} invocation logits are invalid.", file=sys.stderr)
            return 1
        termination = output.control_out.get("termination_logit")
        if termination is None or not np.isfinite(float(termination)):
            print(f"S2M FAIL: {level_id} termination logit is invalid.", file=sys.stderr)
            return 1
        if level_id == "L6" and not _validate_l6_credit(output.diagnostics):
            print("S2M FAIL: L6 failure.credit is invalid.", file=sys.stderr)
            return 1

    sequence = current_state.to_canonical_sequence()
    if not np.isfinite(sequence).all():
        print("S2M FAIL: non-finite values detected.", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "suite": "S2M",
                "state_ir_token_order": STATE_IR_TOKEN_ORDER,
                "levels": [f"L{i}" for i in range(7)],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
