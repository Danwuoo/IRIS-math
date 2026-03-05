from __future__ import annotations

import numpy as np
import pytest

from iris.schema import STATE_IR_TOKEN_ORDER, StateIR, StateIRValidationError


def _section(rows: int, hidden_dim: int = 4) -> np.ndarray:
    return np.ones((rows, hidden_dim), dtype=np.float32)


def test_rejects_non_canonical_order() -> None:
    with pytest.raises(StateIRValidationError):
        StateIR.from_ordered_sections(
            [
                ("G", _section(1)),
                ("T", _section(1)),
                ("O", _section(1)),
                ("R", _section(0)),
                ("X", _section(0)),
                ("M", _section(0)),
            ]
        )


def test_rejects_unknown_token_category() -> None:
    token_map = {
        "T": _section(1),
        "G": _section(1),
        "O": _section(0),
        "R": _section(0),
        "X": _section(0),
        "M": _section(0),
        "Q": _section(0),
    }
    with pytest.raises(StateIRValidationError):
        StateIR.from_token_map(token_map)


def test_empty_non_singleton_sections_are_allowed() -> None:
    state = StateIR(
        T=_section(1),
        G=_section(1),
        O=_section(0),
        R=_section(0),
        X=_section(0),
        M=_section(0),
    )
    sequence = state.to_canonical_sequence()
    assert sequence.shape == (2, 4)
    assert STATE_IR_TOKEN_ORDER == ("T", "G", "O", "R", "X", "M")
