from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

STATE_IR_TOKEN_ORDER: Tuple[str, ...] = ("T", "G", "O", "R", "X", "M")
_SINGLETON_TOKEN_TYPES = {"T", "G"}


class StateIRValidationError(ValueError):
    pass


def _normalize_section_array(token_type: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise StateIRValidationError(
            f"State IR section {token_type} must be rank-2, got ndim={array.ndim}."
        )
    if token_type in _SINGLETON_TOKEN_TYPES and array.shape[0] != 1:
        raise StateIRValidationError(
            f"State IR section {token_type} must contain exactly one token."
        )
    return array


@dataclass(frozen=True)
class StateIR:
    T: np.ndarray
    G: np.ndarray
    O: np.ndarray
    R: np.ndarray
    X: np.ndarray
    M: np.ndarray

    def __post_init__(self) -> None:
        hidden_dim = None
        for token_type in STATE_IR_TOKEN_ORDER:
            normalized = _normalize_section_array(token_type, getattr(self, token_type))
            object.__setattr__(self, token_type, normalized)
            if hidden_dim is None:
                hidden_dim = normalized.shape[1]
            elif normalized.shape[1] != hidden_dim:
                raise StateIRValidationError(
                    "All State IR sections must use the same hidden dimension."
                )

    @property
    def hidden_dim(self) -> int:
        return int(self.T.shape[1])

    @property
    def total_tokens(self) -> int:
        return int(sum(self.section_lengths().values()))

    def section_lengths(self) -> Dict[str, int]:
        return {
            token_type: int(getattr(self, token_type).shape[0])
            for token_type in STATE_IR_TOKEN_ORDER
        }

    def to_ordered_sections(self) -> Tuple[Tuple[str, np.ndarray], ...]:
        return tuple((token_type, getattr(self, token_type)) for token_type in STATE_IR_TOKEN_ORDER)

    def to_token_map(self) -> Dict[str, np.ndarray]:
        return {token_type: getattr(self, token_type) for token_type in STATE_IR_TOKEN_ORDER}

    def to_canonical_sequence(self) -> np.ndarray:
        return np.concatenate(
            [getattr(self, token_type) for token_type in STATE_IR_TOKEN_ORDER],
            axis=0,
        )

    def with_updated_sequence(self, updated_sequence: np.ndarray) -> "StateIR":
        sequence = np.asarray(updated_sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise StateIRValidationError("Updated sequence must be rank-2.")
        if sequence.shape[1] != self.hidden_dim:
            raise StateIRValidationError(
                f"Updated sequence hidden dim {sequence.shape[1]} does not match {self.hidden_dim}."
            )

        lengths = self.section_lengths()
        expected_rows = int(sum(lengths.values()))
        if sequence.shape[0] != expected_rows:
            raise StateIRValidationError(
                f"Updated sequence token count {sequence.shape[0]} does not match {expected_rows}."
            )

        offset = 0
        ordered_sections = []
        for token_type in STATE_IR_TOKEN_ORDER:
            section_len = lengths[token_type]
            ordered_sections.append((token_type, sequence[offset : offset + section_len]))
            offset += section_len
        return StateIR.from_ordered_sections(ordered_sections)

    @classmethod
    def empty(cls, hidden_dim: int) -> "StateIR":
        zeros = lambda rows: np.zeros((rows, hidden_dim), dtype=np.float32)
        return cls(T=zeros(1), G=zeros(1), O=zeros(0), R=zeros(0), X=zeros(0), M=zeros(0))

    @classmethod
    def from_ordered_sections(
        cls, ordered_sections: Sequence[Tuple[str, np.ndarray]]
    ) -> "StateIR":
        if len(ordered_sections) != len(STATE_IR_TOKEN_ORDER):
            raise StateIRValidationError("State IR must provide all canonical sections once.")

        observed_order = tuple(token_type for token_type, _ in ordered_sections)
        if observed_order != STATE_IR_TOKEN_ORDER:
            raise StateIRValidationError(
                f"Canonical State IR order is {STATE_IR_TOKEN_ORDER}, got {observed_order}."
            )

        return cls(
            T=ordered_sections[0][1],
            G=ordered_sections[1][1],
            O=ordered_sections[2][1],
            R=ordered_sections[3][1],
            X=ordered_sections[4][1],
            M=ordered_sections[5][1],
        )

    @classmethod
    def from_token_map(cls, token_map: Mapping[str, np.ndarray]) -> "StateIR":
        provided = set(token_map.keys())
        expected = set(STATE_IR_TOKEN_ORDER)
        unknown = sorted(provided - expected)
        missing = sorted(expected - provided)
        if unknown:
            raise StateIRValidationError(
                f"Unknown State IR token categories are forbidden: {unknown}."
            )
        if missing:
            raise StateIRValidationError(f"Missing required State IR token categories: {missing}.")

        ordered_sections = [(token_type, token_map[token_type]) for token_type in STATE_IR_TOKEN_ORDER]
        return cls.from_ordered_sections(ordered_sections)
