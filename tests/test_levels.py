from __future__ import annotations

import numpy as np

from iris.levels import LevelInput, build_level_stack
from iris.schema import StateIR


def _state(hidden_dim: int = 8) -> StateIR:
    rng = np.random.default_rng(0)
    return StateIR(
        T=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        G=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        O=rng.normal(size=(2, hidden_dim)).astype(np.float32),
        R=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        X=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        M=rng.normal(size=(0, hidden_dim)).astype(np.float32),
    )


def test_all_levels_exist_and_emit_stub_diagnostics() -> None:
    state = _state()
    stack = build_level_stack(implementation="stub")
    assert sorted(stack.keys()) == [f"L{i}" for i in range(7)]

    current_state = state
    baseline_sequence = state.to_canonical_sequence()
    for level_id in [f"L{i}" for i in range(7)]:
        output = stack[level_id].run(LevelInput(state_in=current_state))
        assert output.diagnostics["disabled"] is True
        assert output.control_out["mode"] == "neutral"
        current_state = output.state_out

    np.testing.assert_allclose(current_state.to_canonical_sequence(), baseline_sequence)
