from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..schema import StateIR


@dataclass(frozen=True)
class LevelInput:
    state_in: StateIR
    context_in: Optional[Mapping[str, Any]] = None
    control_in: Optional[Mapping[str, Any]] = None
    resource_budget: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class LevelOutput:
    state_out: StateIR
    control_out: Dict[str, Any]
    diagnostics: Dict[str, Any]


def neutral_control_output() -> Dict[str, Any]:
    return {
        "mode": "neutral",
        "level_invocation_logits": [0.0] * 7,
        "termination_logit": 0.0,
    }


def basic_state_diagnostics(state: StateIR) -> Dict[str, Any]:
    lengths = state.section_lengths()
    return {
        "state.token.count": state.total_tokens,
        "rep.object.count": lengths["O"],
        "rep.relation.count": lengths["R"],
        "rep.event.count": lengths["X"],
        "abs.macro.count": lengths["M"],
    }


class LevelInterface(ABC):
    level_id: str

    def __init__(self, level_id: str, enabled: bool = False) -> None:
        self.level_id = level_id
        self.enabled = enabled

    @abstractmethod
    def run(self, level_input: LevelInput) -> LevelOutput:
        raise NotImplementedError
