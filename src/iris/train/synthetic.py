from __future__ import annotations

import hashlib

import numpy as np

from ..schema import StateIR


def dataset_slice_id_for_segment(segment_id: int) -> str:
    return f"slice-{segment_id:06d}"


def _stable_seed(
    run_id: str,
    dataset_slice_id: str,
    segment_id: int,
    micro_step_idx: int,
    data_seed: int,
) -> int:
    payload = (
        f"{run_id}|{dataset_slice_id}|{segment_id}|{micro_step_idx}|{data_seed}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def generate_synthetic_state(
    run_id: str,
    dataset_slice_id: str,
    segment_id: int,
    micro_step_idx: int,
    hidden_dim: int,
    data_seed: int,
) -> StateIR:
    seed = _stable_seed(
        run_id=run_id,
        dataset_slice_id=dataset_slice_id,
        segment_id=segment_id,
        micro_step_idx=micro_step_idx,
        data_seed=data_seed,
    )
    rng = np.random.default_rng(seed)
    object_count = int(rng.integers(1, 4))
    relation_count = int(rng.integers(0, 4))
    event_count = int(rng.integers(0, 3))
    macro_count = int(rng.integers(0, 2))
    return StateIR(
        T=rng.normal(0.0, 1.0, (1, hidden_dim)).astype(np.float32),
        G=rng.normal(0.0, 1.0, (1, hidden_dim)).astype(np.float32),
        O=rng.normal(0.0, 1.0, (object_count, hidden_dim)).astype(np.float32),
        R=rng.normal(0.0, 1.0, (relation_count, hidden_dim)).astype(np.float32),
        X=rng.normal(0.0, 1.0, (event_count, hidden_dim)).astype(np.float32),
        M=rng.normal(0.0, 1.0, (macro_count, hidden_dim)).astype(np.float32),
    )
