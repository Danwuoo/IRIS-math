from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..schema import StateIR
from .types import ArcExample, ArcTask, grid_shape


@dataclass(frozen=True)
class ArcEncodingConfig:
    object_cap: int = 32
    relation_cap: int = 64
    event_cap: int = 16
    macro_cap: int = 8


def _hash_to_vector(payload: str, hidden_dim: int) -> np.ndarray:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    base = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    if hidden_dim <= base.shape[0]:
        return base[:hidden_dim]
    repeats = int(np.ceil(hidden_dim / float(base.shape[0])))
    tiled = np.tile(base, repeats)
    return tiled[:hidden_dim]


def _grid_stats_vector(grid: Sequence[Sequence[int]], hidden_dim: int) -> np.ndarray:
    rows, cols = grid_shape(grid)
    cells = np.asarray(grid, dtype=np.int32).reshape(-1)
    counts = np.bincount(cells, minlength=10).astype(np.float32)
    counts = counts / max(float(cells.size), 1.0)
    meta = np.asarray(
        [
            float(rows),
            float(cols),
            float(rows * cols),
            float(np.mean(cells)) if cells.size else 0.0,
            float(np.std(cells)) if cells.size else 0.0,
            float(np.min(cells)) if cells.size else 0.0,
            float(np.max(cells)) if cells.size else 0.0,
            float(np.sum(cells % 2 == 0)) / max(float(cells.size), 1.0),
        ],
        dtype=np.float32,
    )
    raw = np.concatenate([counts, meta], axis=0)
    if hidden_dim <= raw.shape[0]:
        return raw[:hidden_dim]
    padded = np.zeros((hidden_dim,), dtype=np.float32)
    padded[: raw.shape[0]] = raw
    return padded


def grid_to_vector(grid: Sequence[Sequence[int]], hidden_dim: int, *, salt: str) -> np.ndarray:
    stats = _grid_stats_vector(grid, hidden_dim=hidden_dim)
    hashed = _hash_to_vector(
        json.dumps({"salt": salt, "grid": grid}, sort_keys=True),
        hidden_dim=hidden_dim,
    )
    return (0.65 * stats + 0.35 * hashed).astype(np.float32)


def _vector_pair_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (right - left).astype(np.float32)


def _mode_shape(examples: Sequence[ArcExample], fallback: Tuple[int, int]) -> Tuple[int, int]:
    if not examples:
        return fallback
    counts: dict[Tuple[int, int], int] = {}
    for example in examples:
        shape = example.output_shape
        counts[shape] = counts.get(shape, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0] if ranked else fallback


def infer_target_shape(task: ArcTask, test_example: ArcExample) -> Tuple[int, int]:
    candidates = [example for example in task.train_examples if example.input_shape == test_example.input_shape]
    if candidates:
        return candidates[0].output_shape
    return _mode_shape(task.train_examples, fallback=test_example.input_shape)


def encode_arc_case_to_state(
    *,
    task: ArcTask,
    test_example: ArcExample,
    hidden_dim: int,
    config: ArcEncodingConfig | None = None,
    train_examples_override: Sequence[ArcExample] | None = None,
) -> StateIR:
    config = config or ArcEncodingConfig()
    train_examples = tuple(train_examples_override) if train_examples_override is not None else tuple(task.train_examples)
    test_input_vec = grid_to_vector(test_example.input_grid, hidden_dim=hidden_dim, salt=f"{task.task_id}/test_input")
    task_hash_vec = _hash_to_vector(
        json.dumps(
            {
                "task_id": task.task_id,
                "concept_id": task.concept_id,
                "source_name": task.source_name,
            },
            sort_keys=True,
        ),
        hidden_dim=hidden_dim,
    ).astype(np.float32)

    train_input_vecs: List[np.ndarray] = []
    train_output_vecs: List[np.ndarray] = []
    relation_vecs: List[np.ndarray] = []
    event_vecs: List[np.ndarray] = []
    for index, example in enumerate(train_examples):
        input_vec = grid_to_vector(
            example.input_grid,
            hidden_dim=hidden_dim,
            salt=f"{task.task_id}/train_input/{index}",
        )
        output_vec = grid_to_vector(
            example.output_grid,
            hidden_dim=hidden_dim,
            salt=f"{task.task_id}/train_output/{index}",
        )
        train_input_vecs.append(input_vec)
        train_output_vecs.append(output_vec)
        relation_vecs.append(_vector_pair_delta(input_vec, output_vec))
        event_vecs.append((0.5 * (input_vec + output_vec)).astype(np.float32))

    if train_input_vecs:
        train_mean = np.mean(np.stack(train_input_vecs + train_output_vecs, axis=0), axis=0).astype(np.float32)
    else:
        train_mean = np.zeros((hidden_dim,), dtype=np.float32)

    task_token = (0.55 * task_hash_vec + 0.45 * test_input_vec).astype(np.float32)
    global_token = (0.7 * train_mean + 0.3 * test_input_vec).astype(np.float32)

    object_tokens = train_input_vecs + train_output_vecs + [test_input_vec]
    object_tokens = object_tokens[: config.object_cap]
    relation_tokens = relation_vecs[: config.relation_cap]
    event_tokens = event_vecs[: config.event_cap]

    macro_tokens: List[np.ndarray] = []
    if relation_vecs:
        macro_tokens.append(np.mean(np.stack(relation_vecs, axis=0), axis=0).astype(np.float32))
    if train_output_vecs:
        macro_tokens.append(np.mean(np.stack(train_output_vecs, axis=0), axis=0).astype(np.float32))
    macro_tokens = macro_tokens[: config.macro_cap]

    def _stack_or_empty(vectors: Sequence[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.zeros((0, hidden_dim), dtype=np.float32)
        return np.stack(vectors, axis=0).astype(np.float32)

    return StateIR(
        T=np.expand_dims(task_token, axis=0),
        G=np.expand_dims(global_token, axis=0),
        O=_stack_or_empty(object_tokens),
        R=_stack_or_empty(relation_tokens),
        X=_stack_or_empty(event_tokens),
        M=_stack_or_empty(macro_tokens),
    )

