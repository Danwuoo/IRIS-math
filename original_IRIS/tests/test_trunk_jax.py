from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.schema import StateIR
from iris.trunk import SingleTrunk


def _state(hidden_dim: int = 8) -> StateIR:
    rng = np.random.default_rng(42)
    return StateIR(
        T=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        G=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        O=rng.normal(size=(2, hidden_dim)).astype(np.float32),
        R=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        X=rng.normal(size=(1, hidden_dim)).astype(np.float32),
        M=rng.normal(size=(1, hidden_dim)).astype(np.float32),
    )


def test_trunk_forward_uses_jax_and_produces_finite_outputs() -> None:
    state = _state()
    trunk = SingleTrunk(hidden_dim=state.hidden_dim, seed=0, backend="jax")
    output = trunk.forward(state)

    assert output.backend == "jax"
    assert output.control_logits["level_invocation"].shape == (7,)
    assert output.control_logits["termination"].shape == (1,)
    assert np.isfinite(output.state_out.to_canonical_sequence()).all()
    assert np.isfinite(output.control_logits["level_invocation"]).all()
    assert np.isfinite(output.control_logits["termination"]).all()
