from __future__ import annotations

import json
import sys

import numpy as np

import _bootstrap  # noqa: F401
from iris.levels import LevelInput, build_level_stack
from iris.schema import STATE_IR_TOKEN_ORDER, StateIR, StateIRValidationError


def _build_state(hidden_dim: int) -> StateIR:
    zeros = lambda rows: np.zeros((rows, hidden_dim), dtype=np.float32)
    return StateIR(T=zeros(1), G=zeros(1), O=zeros(1), R=zeros(0), X=zeros(0), M=zeros(0))


def main() -> int:
    hidden_dim = 8
    state = _build_state(hidden_dim=hidden_dim)

    try:
        StateIR.from_ordered_sections(
            [
                ("G", state.G),
                ("T", state.T),
                ("O", state.O),
                ("R", state.R),
                ("X", state.X),
                ("M", state.M),
            ]
        )
        print("S2 FAIL: ordering violation was not rejected.", file=sys.stderr)
        return 1
    except StateIRValidationError:
        pass

    try:
        StateIR.from_token_map(
            {
                "T": state.T,
                "G": state.G,
                "O": state.O,
                "R": state.R,
                "X": state.X,
                "M": state.M,
                "Q": state.M,
            }
        )
        print("S2 FAIL: unknown token category was not rejected.", file=sys.stderr)
        return 1
    except StateIRValidationError:
        pass

    level_stack = build_level_stack(implementation="stub")
    if sorted(level_stack.keys()) != [f"L{i}" for i in range(7)]:
        print("S2 FAIL: L0-L6 interfaces are incomplete.", file=sys.stderr)
        return 1

    current_state = state
    diagnostics = {}
    for level_id in [f"L{i}" for i in range(7)]:
        result = level_stack[level_id].run(LevelInput(state_in=current_state))
        diagnostics[level_id] = result.diagnostics
        current_state = result.state_out
        if not result.diagnostics.get("disabled", False):
            print(f"S2 FAIL: {level_id} stub did not report disabled marker.", file=sys.stderr)
            return 1
        if result.control_out.get("mode") != "neutral":
            print(f"S2 FAIL: {level_id} stub did not emit neutral control.", file=sys.stderr)
            return 1

    report = {
        "status": "PASS",
        "suite": "S2",
        "state_ir_token_order": STATE_IR_TOKEN_ORDER,
        "levels": list(diagnostics.keys()),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
