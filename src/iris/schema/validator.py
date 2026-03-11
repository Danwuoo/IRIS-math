from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from .state_ir import STATE_IR_SLOT_ORDER, StateIR, StateIRValidationError


def validate_canonical_order(ordered_sections: Sequence[Tuple[str, Any]]) -> None:
    observed_order = tuple(token_type for token_type, _ in ordered_sections)
    if observed_order != STATE_IR_SLOT_ORDER:
        raise StateIRValidationError(
            f"Canonical State IR order is {STATE_IR_SLOT_ORDER}, got {observed_order}."
        )


def validate_token_map(token_map: Mapping[str, Any]) -> StateIR:
    return StateIR.from_slot_map(token_map)


def validate_state_ir(state: StateIR) -> StateIR:
    if not isinstance(state, StateIR):
        raise StateIRValidationError(f"Expected StateIR instance, got {type(state).__name__}.")
    state.to_canonical_sequence()
    return state
