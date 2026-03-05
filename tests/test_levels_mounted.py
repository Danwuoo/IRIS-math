from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.levels import LevelInput, build_level_stack
from iris.schema import StateIR


def _state(hidden_dim: int = 8) -> StateIR:
    rng = np.random.default_rng(7)
    return StateIR(
        T=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        G=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        O=rng.normal(size=(2, hidden_dim)).astype(np.float32),
        R=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        X=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        M=rng.normal(size=(1, hidden_dim)).astype(np.float32),
    )


def test_mounted_levels_preserve_state_shape_and_emit_l6_credit() -> None:
    state = _state()
    initial_lengths = state.section_lengths()
    stack = build_level_stack(implementation="mounted", hidden_dim=state.hidden_dim, seed=0)

    current_state = state
    l6_credit = None
    for level_id in [f"L{i}" for i in range(7)]:
        output = stack[level_id].run(LevelInput(state_in=current_state))
        assert output.diagnostics["disabled"] is False
        assert output.control_out["mode"] == "mounted"
        assert len(output.control_out["level_invocation_logits"]) == 7
        current_state = output.state_out
        if level_id == "L6":
            l6_credit = output.diagnostics.get("failure.credit")

    assert current_state.section_lengths() == initial_lengths
    assert np.isfinite(current_state.to_canonical_sequence()).all()

    assert isinstance(l6_credit, dict)
    values = np.asarray([float(l6_credit[f"L{i}"]) for i in range(7)], dtype=np.float64)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.isclose(np.sum(values), 1.0)
