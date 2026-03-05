from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..runtime import assert_jax_runtime
from ..schema import STATE_IR_TOKEN_ORDER, StateIR


@dataclass(frozen=True)
class TrunkOutput:
    state_out: StateIR
    control_logits: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any]
    backend: str


def _section_lengths_tuple(state: StateIR) -> Tuple[int, ...]:
    lengths = state.section_lengths()
    return tuple(int(lengths[token_type]) for token_type in STATE_IR_TOKEN_ORDER)


def _random_matrix(key: jax.Array, shape: Tuple[int, ...], scale: float = 0.02) -> jax.Array:
    return scale * jax.random.normal(key, shape=shape, dtype=jnp.float32)


def init_trunk_params(hidden_dim: int, seed: int = 0) -> Dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    key_type, key_seq_w, key_seq_b, key_ctrl_w, key_ctrl_b = jax.random.split(key, 5)
    return {
        "type_embeddings": _random_matrix(
            key_type,
            (len(STATE_IR_TOKEN_ORDER), hidden_dim),
        ),
        "seq_w": _random_matrix(key_seq_w, (hidden_dim, hidden_dim)),
        "seq_b": _random_matrix(key_seq_b, (hidden_dim,)),
        "ctrl_w": _random_matrix(key_ctrl_w, (hidden_dim, 8)),
        "ctrl_b": _random_matrix(key_ctrl_b, (8,)),
    }


def expand_type_embeddings(type_embeddings: jax.Array, section_lengths: Tuple[int, ...]) -> jax.Array:
    rows = []
    for index, count in enumerate(section_lengths):
        if count <= 0:
            continue
        rows.append(jnp.repeat(type_embeddings[index : index + 1], repeats=count, axis=0))
    if not rows:
        raise ValueError("StateIR section lengths produced an empty canonical sequence.")
    return jnp.concatenate(rows, axis=0)


def build_typed_sequence(
    sequence: jax.Array,
    section_lengths: Tuple[int, ...],
    type_embeddings: jax.Array,
) -> jax.Array:
    return sequence + expand_type_embeddings(type_embeddings, section_lengths)


def forward_with_params(params: Mapping[str, jax.Array], typed_sequence: jax.Array) -> Tuple[jax.Array, jax.Array]:
    updated_sequence = jnp.tanh(typed_sequence @ params["seq_w"] + params["seq_b"])
    pooled = jnp.mean(updated_sequence, axis=0)
    control_raw = pooled @ params["ctrl_w"] + params["ctrl_b"]
    return updated_sequence, control_raw


def _serialize_param_tree(tree: Mapping[str, jax.Array]) -> Dict[str, Any]:
    return {
        key: np.asarray(value, dtype=np.float32).tolist()
        for key, value in tree.items()
    }


def _deserialize_param_tree(tree: Mapping[str, Any]) -> Dict[str, jax.Array]:
    return {
        key: jnp.asarray(np.asarray(value, dtype=np.float32))
        for key, value in tree.items()
    }


class SingleTrunk:
    def __init__(self, hidden_dim: int, seed: int = 0, backend: str = "jax") -> None:
        assert_jax_runtime(device="cpu", require_gpu=False)
        if backend != "jax":
            raise ValueError("Strict JAX mode: SingleTrunk backend must be 'jax'.")
        self.hidden_dim = hidden_dim
        self.backend = "jax"
        self.params = init_trunk_params(hidden_dim=hidden_dim, seed=seed)

    @staticmethod
    def available_backends() -> List[str]:
        return ["jax"]

    def typed_sequence(self, state: StateIR) -> Tuple[jax.Array, Tuple[int, ...]]:
        base_sequence = jnp.asarray(state.to_canonical_sequence(), dtype=jnp.float32)
        section_lengths = _section_lengths_tuple(state)
        typed_sequence = build_typed_sequence(
            sequence=base_sequence,
            section_lengths=section_lengths,
            type_embeddings=self.params["type_embeddings"],
        )
        return typed_sequence, section_lengths

    def forward(self, state: StateIR) -> TrunkOutput:
        typed_sequence, _ = self.typed_sequence(state)
        updated_sequence, control_raw = forward_with_params(self.params, typed_sequence)
        control_logits = {
            "level_invocation": np.asarray(control_raw[:7], dtype=np.float32),
            "termination": np.asarray([control_raw[7]], dtype=np.float32),
        }
        state_out = state.with_updated_sequence(np.asarray(updated_sequence, dtype=np.float32))
        diagnostics = {
            "trunk.gain": 1.0,
            "trunk.sequence_norm": float(jnp.linalg.norm(updated_sequence)),
            "trunk.control_norm": float(jnp.linalg.norm(control_raw)),
        }
        return TrunkOutput(
            state_out=state_out,
            control_logits=control_logits,
            diagnostics=diagnostics,
            backend=self.backend,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": "iris.trunk_state/v1",
            "backend": self.backend,
            "hidden_dim": int(self.hidden_dim),
            "params": _serialize_param_tree(self.params),
        }

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        backend = str(payload.get("backend", "jax"))
        if backend != "jax":
            raise ValueError("Strict JAX mode: trunk checkpoint backend must be 'jax'.")
        self.hidden_dim = int(payload["hidden_dim"])
        self.backend = "jax"
        self.params = _deserialize_param_tree(payload["params"])
