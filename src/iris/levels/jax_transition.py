from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..runtime import assert_jax_runtime
from .base import LevelInput, LevelInterface, LevelOutput, basic_state_diagnostics

LEVEL_IDS: Tuple[str, ...] = tuple(f"L{index}" for index in range(7))


def _state_target_summary(state_out: Any) -> Dict[str, Any]:
    return {
        "pf_task_type": state_out.PF.task_type,
        "pf_target_spec": state_out.PF.target_spec,
        "frontier_count": len(state_out.FR),
        "verifier_evidence_count": len(state_out.VS),
    }


def _emitted_object_refs(state_out: Any) -> Dict[str, Any]:
    branch_ids = [entry.branch_id for entry in state_out.FR if hasattr(entry, "branch_id")]
    subgoal_ids = [entry.subgoal_id for entry in state_out.FR if hasattr(entry, "subgoal_id")]
    vs_ids = [entry.vs_id for entry in state_out.VS if hasattr(entry, "vs_id")]
    lm_ids = [entry.lm_id for entry in state_out.LM if hasattr(entry, "lm_id")]
    return {
        "branch_ids": branch_ids,
        "subgoal_ids": subgoal_ids,
        "vs_ids": vs_ids,
        "lm_ids": lm_ids,
    }


def _l3_internal_heads(state_out: Any, control_raw: jax.Array) -> Dict[str, Any]:
    branch_ids = [entry.branch_id for entry in state_out.FR if hasattr(entry, "branch_id")]
    subgoal_ids = [entry.subgoal_id for entry in state_out.FR if hasattr(entry, "subgoal_id")]
    return {
        "branch_controller": {
            "status": "mounted",
            "selected_branch_id": branch_ids[0] if branch_ids else None,
            "selected_subgoal_id": subgoal_ids[0] if subgoal_ids else None,
            "strategy_transition_score": float(np.asarray(control_raw[0])),
        },
        "budget_allocator": {
            "status": "mounted",
            "budget_pressure": float(jax.nn.sigmoid(control_raw[1])),
            "global_step_budget_remaining": state_out.CS.budget_state.global_step_budget_remaining,
            "verifier_probe_budget_remaining": state_out.CS.budget_state.verifier_probe_budget_remaining,
        },
        "repair_scheduler": {
            "status": "mounted",
            "recovery_target": state_out.CS.recovery_target,
            "selected_action_type": state_out.CS.selected_action.action_type,
            "repair_score": float(np.asarray(control_raw[2])),
        },
    }


def _l6_internal_heads(
    state_out: Any,
    control_raw: jax.Array,
    credit: Mapping[str, float],
) -> Dict[str, Any]:
    confidence = float(jax.nn.sigmoid(control_raw[7]))
    false_accept_risk = max(0.0, 1.0 - confidence)
    false_reject_risk = max(0.0, confidence - 0.5)
    evidence_classes = sorted({entry.evidence_class for entry in state_out.VS if hasattr(entry, "evidence_class")})
    dominant_level = max(credit, key=lambda key: float(credit[key]))
    return {
        "verifier_aggregator": {
            "status": "mounted",
            "evidence_classes": evidence_classes,
            "supporting_vs_ids": [entry.vs_id for entry in state_out.VS if hasattr(entry, "vs_id")],
            "disagreement": float(abs(np.asarray(control_raw[0]) - np.asarray(control_raw[1]))),
        },
        "credit_router": {
            "status": "mounted",
            "failure.credit": dict(credit),
            "dominant_level": dominant_level,
            "multi_level": len([value for value in credit.values() if value >= 0.1]) > 1,
        },
        "calibration_head": {
            "status": "mounted",
            "confidence": confidence,
            "abstention_margin": float(abs(confidence - 0.5)),
            "false_accept_risk": float(false_accept_risk),
            "false_reject_risk": float(false_reject_risk),
        },
    }


def _random_matrix(key: jax.Array, shape: Tuple[int, ...], scale: float = 0.02) -> jax.Array:
    return scale * jax.random.normal(key, shape=shape, dtype=jnp.float32)


def init_level_params(hidden_dim: int, seed: int, include_credit: bool = False) -> Dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    split_count = 8 if include_credit else 6
    keys = jax.random.split(key, split_count)
    params: Dict[str, jax.Array] = {
        "res_w": _random_matrix(keys[0], (hidden_dim, hidden_dim)),
        "res_b": _random_matrix(keys[1], (hidden_dim,)),
        "gate_w": _random_matrix(keys[2], (hidden_dim, hidden_dim)),
        "gate_b": _random_matrix(keys[3], (hidden_dim,)),
        "ctrl_w": _random_matrix(keys[4], (hidden_dim, 8)),
        "ctrl_b": _random_matrix(keys[5], (8,)),
    }
    if include_credit:
        params["credit_w"] = _random_matrix(keys[6], (hidden_dim, len(LEVEL_IDS)))
        params["credit_b"] = _random_matrix(keys[7], (len(LEVEL_IDS),))
    return params


def level_forward(
    params: Mapping[str, jax.Array],
    sequence: jax.Array,
    alpha: float = 0.1,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    pooled = jnp.mean(sequence, axis=0)
    gate = jax.nn.sigmoid(pooled @ params["gate_w"] + params["gate_b"])
    delta = jnp.tanh(sequence @ params["res_w"] + params["res_b"]) * gate
    updated_sequence = sequence + alpha * delta
    control_raw = pooled @ params["ctrl_w"] + params["ctrl_b"]
    return updated_sequence, control_raw, pooled


def l6_credit_from_params(params: Mapping[str, jax.Array], pooled: jax.Array) -> jax.Array:
    logits = pooled @ params["credit_w"] + params["credit_b"]
    return jax.nn.softmax(logits, axis=-1)


def init_level_stack_params(hidden_dim: int, seed: int = 0) -> Dict[str, Dict[str, jax.Array]]:
    return {
        level_id: init_level_params(
            hidden_dim=hidden_dim,
            seed=seed + level_index + 1,
            include_credit=(level_id == "L6"),
        )
        for level_index, level_id in enumerate(LEVEL_IDS)
    }


def apply_level_stack_params(
    level_params: Mapping[str, Mapping[str, jax.Array]],
    sequence: jax.Array,
    alpha: float = 0.1,
) -> Tuple[jax.Array, Dict[str, jax.Array], jax.Array]:
    controls: Dict[str, jax.Array] = {}
    current = sequence
    l6_credit = jnp.ones((len(LEVEL_IDS),), dtype=jnp.float32) / float(len(LEVEL_IDS))
    for level_id in LEVEL_IDS:
        current, control_raw, pooled = level_forward(level_params[level_id], current, alpha=alpha)
        controls[level_id] = control_raw
        if level_id == "L6":
            l6_credit = l6_credit_from_params(level_params[level_id], pooled)
    return current, controls, l6_credit


class _MountedLevel(LevelInterface):
    def __init__(
        self,
        level_id: str,
        hidden_dim: int,
        seed: int,
        *,
        alpha: float = 0.1,
        include_credit: bool = False,
    ) -> None:
        super().__init__(level_id=level_id, enabled=True)
        assert_jax_runtime(device="cpu", require_gpu=False)
        self.hidden_dim = hidden_dim
        self.alpha = float(alpha)
        self.include_credit = include_credit
        self.params = init_level_params(
            hidden_dim=hidden_dim,
            seed=seed,
            include_credit=include_credit,
        )

    def run(self, level_input: LevelInput) -> LevelOutput:
        sequence = jnp.asarray(level_input.state_in.to_canonical_sequence(), dtype=jnp.float32)
        updated_sequence, control_raw, pooled = level_forward(
            self.params,
            sequence,
            alpha=self.alpha,
        )
        state_out = level_input.state_in.with_updated_sequence(
            np.asarray(updated_sequence, dtype=np.float32)
        )
        diagnostics = basic_state_diagnostics(state_out)
        diagnostics.update(
            {
                "level": self.level_id,
                "enabled": True,
                "disabled": False,
                "implementation_status": "jax_transition",
                "invocation_outcome": "updated_state",
                "target_summary": _state_target_summary(state_out),
                "emitted_object_refs": _emitted_object_refs(state_out),
                "evidence_trigger_refs": [],
                "confidence": float(jax.nn.sigmoid(control_raw[7])),
                "uncertainty": float(1.0 - jax.nn.sigmoid(control_raw[7])),
                "failure_tags": [],
                "credit_hints": {level_id: 0.0 for level_id in LEVEL_IDS},
                "technical_debt": {
                    "label": "TEMPORARY TECHNICAL DEBT",
                    "note": "Vector-only JAX level path remains available only for transition/training plumbing.",
                    "removal_criterion": "Semantic mounted levels replace numeric transition path in training and failure.credit.collapse_rate stays within tolerance.",
                    "intended_learned_replacement": "semantic_mounted_levels",
                },
            }
        )
        control_out: Dict[str, Any] = {
            "mode": "jax_transition",
            "level_invocation_logits": [float(value) for value in np.asarray(control_raw[:7])],
            "termination_logit": float(np.asarray(control_raw[7])),
        }
        if self.include_credit:
            probs = l6_credit_from_params(self.params, pooled)
            credit = {
                level_id: float(np.asarray(probs[index]))
                for index, level_id in enumerate(LEVEL_IDS)
            }
            diagnostics["failure.credit"] = credit
            diagnostics["credit_hints"] = credit
            diagnostics["internal_heads"] = _l6_internal_heads(state_out, control_raw, credit)
        elif self.level_id == "L3":
            diagnostics["internal_heads"] = _l3_internal_heads(state_out, control_raw)
        else:
            diagnostics["internal_heads"] = {}

        return LevelOutput(
            state_out=state_out,
            control_out=control_out,
            diagnostics=diagnostics,
        )


class L0MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L0", hidden_dim=hidden_dim, seed=seed)


class L1MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L1", hidden_dim=hidden_dim, seed=seed)


class L2MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L2", hidden_dim=hidden_dim, seed=seed)


class L3MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L3", hidden_dim=hidden_dim, seed=seed)


class L4MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L4", hidden_dim=hidden_dim, seed=seed)


class L5MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L5", hidden_dim=hidden_dim, seed=seed)


class L6MountedLevel(_MountedLevel):
    def __init__(self, hidden_dim: int, seed: int) -> None:
        super().__init__("L6", hidden_dim=hidden_dim, seed=seed, include_credit=True)
