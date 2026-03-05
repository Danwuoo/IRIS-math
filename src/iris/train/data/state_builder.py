from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...schema import StateIR


def _token_vector(token_id: int, hidden_dim: int) -> np.ndarray:
    index = np.arange(1, hidden_dim + 1, dtype=np.float32)
    token = float(int(token_id) + 1)
    return (
        np.sin(token * index * 0.013).astype(np.float32)
        + np.cos((token + 2.0) * index * 0.007).astype(np.float32)
    )


def _chunk_means(embeddings: np.ndarray, chunk_count: int) -> np.ndarray:
    if embeddings.shape[0] <= 0 or chunk_count <= 0:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    chunk_count = min(chunk_count, embeddings.shape[0])
    chunks = np.array_split(embeddings, chunk_count)
    means = [np.mean(chunk, axis=0) for chunk in chunks if chunk.size > 0]
    if not means:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    return np.stack(means, axis=0).astype(np.float32)


def _g_vector(token_ids: Sequence[int], embeddings: np.ndarray) -> np.ndarray:
    token_count = float(len(token_ids))
    unique_ratio = float(len(set(int(token) for token in token_ids))) / float(max(len(token_ids), 1))
    mean_abs = float(np.mean(np.abs(embeddings))) if embeddings.size > 0 else 0.0
    std = float(np.std(embeddings)) if embeddings.size > 0 else 0.0
    features = np.asarray([token_count, unique_ratio, mean_abs, std], dtype=np.float32)
    repeats = int(np.ceil(float(embeddings.shape[1]) / float(features.shape[0])))
    tiled = np.tile(features, repeats)[: embeddings.shape[1]]
    return tiled.reshape(1, embeddings.shape[1]).astype(np.float32)


def text_to_state_ir(
    *,
    text: str,
    tokenizer: Any,
    hidden_dim: int,
    max_input_tokens: int = 256,
) -> StateIR:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        token_ids = [0]
    token_ids = [int(token) for token in token_ids[: int(max(max_input_tokens, 1))]]

    embeddings = np.stack([_token_vector(token, hidden_dim) for token in token_ids], axis=0).astype(np.float32)

    t_rows = min(8, embeddings.shape[0])
    t_section = np.mean(embeddings[:t_rows], axis=0, keepdims=True).astype(np.float32)
    g_section = _g_vector(token_ids, embeddings)

    o_section = _chunk_means(embeddings, chunk_count=4)
    if o_section.shape[0] > 1:
        relation_rows = [o_section[idx + 1] - o_section[idx] for idx in range(o_section.shape[0] - 1)]
        r_section = np.stack(relation_rows, axis=0).astype(np.float32)
    else:
        r_section = np.zeros((0, hidden_dim), dtype=np.float32)

    if embeddings.shape[0] > 1:
        delta = embeddings[1:] - embeddings[:-1]
        x_section = _chunk_means(delta, chunk_count=2)
    else:
        x_section = np.zeros((0, hidden_dim), dtype=np.float32)

    m_section = np.mean(embeddings, axis=0, keepdims=True).astype(np.float32)

    return StateIR(
        T=t_section,
        G=g_section,
        O=o_section,
        R=r_section,
        X=x_section,
        M=m_section,
    )
