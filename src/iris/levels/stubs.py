from __future__ import annotations

from typing import Dict, Mapping, Optional

from .base import LevelInput, LevelInterface, LevelOutput, basic_state_diagnostics, neutral_control_output


def _default_credit_hints() -> Dict[str, float]:
    return {f"L{level_index}": 0.0 for level_index in range(7)}


def _disabled_internal_heads(level_id: str) -> Dict[str, Dict[str, str]]:
    if level_id == "L3":
        return {
            "branch_controller": {"status": "disabled"},
            "budget_allocator": {"status": "disabled"},
            "repair_scheduler": {"status": "disabled"},
        }
    if level_id == "L6":
        return {
            "verifier_aggregator": {"status": "disabled"},
            "credit_router": {"status": "disabled"},
            "calibration_head": {"status": "disabled"},
        }
    return {}


class StubLevel(LevelInterface):
    def __init__(self, level_id: str, enabled: bool = False) -> None:
        super().__init__(level_id=level_id, enabled=enabled)

    def run(self, level_input: LevelInput) -> LevelOutput:
        diagnostics = basic_state_diagnostics(level_input.state_in)
        diagnostics.update(
            {
                "level": self.level_id,
                "enabled": self.enabled,
                "disabled": not self.enabled,
                "implementation_status": "stub",
                "invocation_outcome": "neutral_passthrough",
                "target_summary": {
                    "pf_task_type": level_input.state_in.PF.task_type,
                    "pf_target_spec": level_input.state_in.PF.target_spec,
                    "frontier_count": len(level_input.state_in.FR),
                },
                "emitted_object_refs": {"branch_ids": [], "subgoal_ids": [], "vs_ids": []},
                "evidence_trigger_refs": [],
                "confidence": 0.0,
                "uncertainty": 0.0,
                "failure_tags": [],
                "credit_hints": _default_credit_hints(),
                "internal_heads": _disabled_internal_heads(self.level_id),
            }
        )
        return LevelOutput(
            state_out=level_input.state_in,
            control_out=neutral_control_output(),
            diagnostics=diagnostics,
        )


class L0Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L0", enabled=enabled)


class L1Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L1", enabled=enabled)


class L2Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L2", enabled=enabled)


class L3Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L3", enabled=enabled)


class L4Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L4", enabled=enabled)


class L5Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L5", enabled=enabled)


class L6Level(StubLevel):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__(level_id="L6", enabled=enabled)


def build_level_stack(
    enabled_levels: Optional[Mapping[str, bool]] = None,
    *,
    implementation: str = "stub",
    hidden_dim: int = 16,
    seed: int = 0,
) -> Dict[str, LevelInterface]:
    enabled_levels = enabled_levels or {}
    if implementation not in {"stub", "mounted", "mixed"}:
        raise ValueError("implementation must be one of: stub, mounted, mixed.")

    if implementation == "stub":
        return {
            "L0": L0Level(enabled=enabled_levels.get("L0", False)),
            "L1": L1Level(enabled=enabled_levels.get("L1", False)),
            "L2": L2Level(enabled=enabled_levels.get("L2", False)),
            "L3": L3Level(enabled=enabled_levels.get("L3", False)),
            "L4": L4Level(enabled=enabled_levels.get("L4", False)),
            "L5": L5Level(enabled=enabled_levels.get("L5", False)),
            "L6": L6Level(enabled=enabled_levels.get("L6", False)),
        }

    from .mounted import (
        L0MountedLevel,
        L1MountedLevel,
        L2MountedLevel,
        L3MountedLevel,
        L4MountedLevel,
        L5MountedLevel,
        L6MountedLevel,
    )

    mounted = {
        "L0": L0MountedLevel(hidden_dim=hidden_dim, seed=seed + 1),
        "L1": L1MountedLevel(hidden_dim=hidden_dim, seed=seed + 2),
        "L2": L2MountedLevel(hidden_dim=hidden_dim, seed=seed + 3),
        "L3": L3MountedLevel(hidden_dim=hidden_dim, seed=seed + 4),
        "L4": L4MountedLevel(hidden_dim=hidden_dim, seed=seed + 5),
        "L5": L5MountedLevel(hidden_dim=hidden_dim, seed=seed + 6),
        "L6": L6MountedLevel(hidden_dim=hidden_dim, seed=seed + 7),
    }
    if implementation == "mounted":
        return mounted

    mixed: Dict[str, LevelInterface] = {}
    for level_id, mounted_level in mounted.items():
        if enabled_levels.get(level_id, False):
            mixed[level_id] = mounted_level
        else:
            mixed[level_id] = StubLevel(level_id=level_id, enabled=False)
    return mixed
