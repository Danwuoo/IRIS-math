from __future__ import annotations

import numpy as np
import pytest

from iris.schema import (
    CANONICAL_ADJUDICATION_STATUSES,
    CANONICAL_RUNTIME_STATUSES,
    AdjudicationState,
    BudgetState,
    ControlAction,
    ControlState,
    ProblemFrame,
    RequiredOutput,
    STATE_IR_TOKEN_ORDER,
    StateIR,
    StateIRValidationError,
)

from tests.state_ir_factory import make_state_ir


def _minimal_state(hidden_dim: int = 4) -> StateIR:
    zero = np.zeros((hidden_dim,), dtype=np.float32)
    return StateIR(
        PF=ProblemFrame(
            task_type="classification",
            target_spec="return label",
            required_output=RequiredOutput(
                output_kind="label",
                answer_channel="structured_object",
            ),
            problem_assumptions=(),
            domain_tags=(),
            source_anchor_refs=(),
            frame_status="draft",
            vector=zero,
        ),
        CS=ControlState(
            selected_action=ControlAction(
                action_id="action-0",
                action_type="continue",
            ),
            budget_state=BudgetState(global_step_budget_remaining=0),
            runtime_status="in_progress",
            uncertainty_state="unknown",
            escalation_state="inactive",
            adjudication_state=AdjudicationState(
                task_adjudication_policy_id="task-family-answer-only-default-v1",
                adjudication_status="pending",
            ),
            vector=zero,
        ),
    )


def test_rejects_non_canonical_order() -> None:
    sample = make_state_ir(hidden_dim=4, seed=1)
    with pytest.raises(StateIRValidationError):
        StateIR.from_ordered_sections(
            [
                ("SY", sample.SY),
                ("PF", sample.PF),
                ("CG", sample.CG),
                ("FR", sample.FR),
                ("LM", sample.LM),
                ("VS", sample.VS),
                ("CS", sample.CS),
            ]
        )


def test_rejects_unknown_token_category() -> None:
    sample = make_state_ir(hidden_dim=4, seed=2)
    token_map = {
        "PF": sample.PF,
        "SY": sample.SY,
        "CG": sample.CG,
        "FR": sample.FR,
        "LM": sample.LM,
        "VS": sample.VS,
        "CS": sample.CS,
        "Q": (),
    }
    with pytest.raises(StateIRValidationError):
        StateIR.from_token_map(token_map)


def test_empty_optional_sections_are_allowed() -> None:
    state = _minimal_state()
    sequence = state.to_canonical_sequence()
    assert sequence.shape == (2, 4)
    assert state.section_lengths() == {
        "PF": 1,
        "SY": 0,
        "CG": 0,
        "FR": 0,
        "LM": 0,
        "VS": 0,
        "CS": 1,
    }
    assert STATE_IR_TOKEN_ORDER == ("PF", "SY", "CG", "FR", "LM", "VS", "CS")


def test_control_state_normalizes_scope_and_status_vocabularies() -> None:
    state = _minimal_state()
    assert state.CS.runtime_status in CANONICAL_RUNTIME_STATUSES
    assert state.CS.adjudication_state is not None
    assert state.CS.adjudication_state.adjudication_status in CANONICAL_ADJUDICATION_STATUSES


def test_budget_exhausted_requires_blocked_adjudication() -> None:
    zero = np.zeros((4,), dtype=np.float32)
    with pytest.raises(StateIRValidationError):
        ControlState(
            selected_action=ControlAction(action_id="action-stop", action_type="stop"),
            budget_state=BudgetState(global_step_budget_remaining=0),
            runtime_status="budget_exhausted",
            uncertainty_state="high",
            escalation_state="inactive",
            adjudication_state=AdjudicationState(
                task_adjudication_policy_id="task-family-answer-only-default-v1",
                adjudication_status="ready",
            ),
            vector=zero,
        )
