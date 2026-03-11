from __future__ import annotations

import numpy as np

from iris.levels import LevelInput, build_level_stack

from tests.state_ir_factory import make_state_ir


def test_all_levels_exist_and_emit_stub_diagnostics() -> None:
    state = make_state_ir(seed=0)
    stack = build_level_stack(implementation="stub")
    assert sorted(stack.keys()) == [f"L{i}" for i in range(7)]

    current_state = state
    baseline_sequence = state.to_canonical_sequence()
    l3_heads = None
    l6_heads = None
    for level_id in [f"L{i}" for i in range(7)]:
        output = stack[level_id].run(LevelInput(state_in=current_state))
        assert output.diagnostics["disabled"] is True
        assert output.diagnostics["implementation_status"] == "stub"
        assert output.control_out["mode"] == "neutral"
        if level_id == "L3":
            l3_heads = output.diagnostics["internal_heads"]
        if level_id == "L6":
            l6_heads = output.diagnostics["internal_heads"]
        current_state = output.state_out

    assert output.diagnostics["target_summary"]["frontier_count"] == len(state.FR)
    assert l3_heads is not None
    assert l3_heads["branch_controller"]["status"] == "disabled"
    assert l6_heads is not None
    assert l6_heads["credit_router"]["status"] == "disabled"
    np.testing.assert_allclose(current_state.to_canonical_sequence(), baseline_sequence)
