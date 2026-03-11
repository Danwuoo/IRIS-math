from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.trunk import SingleTrunk

from tests.state_ir_factory import make_state_ir


def test_trunk_forward_uses_jax_and_produces_finite_outputs() -> None:
    state = make_state_ir(seed=42)
    trunk = SingleTrunk(hidden_dim=state.hidden_dim, seed=0, backend="jax")
    output = trunk.forward(state)

    assert output.backend == "jax"
    assert output.control_logits["level_invocation"].shape == (7,)
    assert output.control_logits["termination"].shape == (1,)
    assert np.isfinite(output.state_out.to_canonical_sequence()).all()
    assert np.isfinite(output.control_logits["level_invocation"]).all()
    assert np.isfinite(output.control_logits["termination"]).all()
