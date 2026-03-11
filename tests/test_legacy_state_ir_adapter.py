from __future__ import annotations

from iris.schema import build_legacy_state_ir_adapter_report

from tests.state_ir_factory import make_state_ir


def test_legacy_state_ir_adapter_report_is_explicitly_transitional() -> None:
    state = make_state_ir(hidden_dim=4, seed=3, include_lm=False, include_vs=False)
    report = build_legacy_state_ir_adapter_report(state)
    assert report.schema == "iris.legacy_state_ir_adapter/v1"
    assert report.implementation_status == "transition_only"
    assert report.lossy_projection is True
    assert report.native_slot_lengths["PF"] == 1
    assert report.legacy_section_lengths["T"] == 1
    assert report.legacy_section_lengths["X"] == len(state.FR) + len(state.VS)
    assert report.v2_slot_status["PF"] == "native_v2_slot_live"
    assert any("TEMPORARY TECHNICAL DEBT" in note for note in report.notes)
