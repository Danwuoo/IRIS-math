from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from iris.levels import LevelInput, build_level_stack

from tests.state_ir_factory import make_state_ir


def test_mounted_levels_preserve_state_shape_and_emit_l6_credit() -> None:
    state = make_state_ir(seed=7)
    initial_lengths = state.section_lengths()
    stack = build_level_stack(implementation="mounted", hidden_dim=state.hidden_dim, seed=0)

    current_state = state
    l3_heads = None
    l6_credit = None
    l6_heads = None
    for level_id in [f"L{i}" for i in range(7)]:
        output = stack[level_id].run(LevelInput(state_in=current_state))
        assert output.diagnostics["disabled"] is False
        assert output.diagnostics["implementation_status"] == "mounted"
        assert output.control_out["mode"] == "mounted"
        assert len(output.control_out["level_invocation_logits"]) == 7
        current_state = output.state_out
        if level_id == "L3":
            l3_heads = output.diagnostics.get("internal_heads")
        if level_id == "L6":
            l6_credit = output.diagnostics.get("failure.credit")
            l6_heads = output.diagnostics.get("internal_heads")

    assert current_state.section_lengths() == initial_lengths
    assert np.isfinite(current_state.to_canonical_sequence()).all()

    assert l3_heads is not None
    assert set(l3_heads.keys()) == {"branch_controller", "budget_allocator", "repair_scheduler"}
    assert isinstance(l6_credit, dict)
    values = np.asarray([float(l6_credit[f"L{i}"]) for i in range(7)], dtype=np.float64)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.isclose(np.sum(values), 1.0)
    assert l6_heads is not None
    assert set(l6_heads.keys()) == {"verifier_aggregator", "credit_router", "calibration_head"}
