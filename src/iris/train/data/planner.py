from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .contracts import PureLMProfile


@dataclass(frozen=True)
class MicroStepPlan:
    micro_step_idx: int
    source_id: str
    target_tokens: int
    sample_key: str


@dataclass(frozen=True)
class SegmentPlan:
    segment_id: int
    dataset_slice_id: str
    plan_hash: str
    steps: Tuple[MicroStepPlan, ...]
    total_target_tokens: int


@dataclass(frozen=True)
class HybridSchedule:
    segment_id: int
    pure_step_flags: Tuple[bool, ...]
    pure_steps: int
    synthetic_steps: int


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_seed(payload: Any) -> int:
    digest = _stable_hash(payload)
    return int(digest[:16], 16)


def _allocate_step_counts(weights: Dict[str, float], micro_steps: int) -> Dict[str, int]:
    if micro_steps <= 0:
        return {source_id: 0 for source_id in weights.keys()}

    expected = {
        source_id: float(weight) * float(micro_steps)
        for source_id, weight in weights.items()
    }
    counts = {source_id: int(value) for source_id, value in expected.items()}
    allocated = sum(counts.values())
    remainder = max(micro_steps - allocated, 0)

    ranked = sorted(
        expected.items(),
        key=lambda item: (item[1] - int(item[1]), item[0]),
        reverse=True,
    )
    for index in range(remainder):
        source_id = ranked[index % len(ranked)][0]
        counts[source_id] += 1

    return counts


def build_pure_lm_segment_plan(
    *,
    profile: PureLMProfile,
    segment_id: int,
    micro_steps: int,
    tokens_per_micro_step: int,
    data_seed: int,
    tokenizer_fingerprint: str,
) -> SegmentPlan:
    if micro_steps <= 0:
        raise ValueError("micro_steps must be positive")
    if tokens_per_micro_step <= 0:
        raise ValueError("tokens_per_micro_step must be positive")

    weights: Dict[str, float] = {}
    ratio_total = float(profile.pure_lm_ratio_total)
    for source in profile.sources:
        weights[source.source_id] = float(source.ratio_total) / ratio_total

    step_counts = _allocate_step_counts(weights, micro_steps)
    source_sequence: List[str] = []
    for source_id in sorted(step_counts.keys()):
        source_sequence.extend([source_id] * int(step_counts[source_id]))

    shuffle_seed = _stable_seed(
        {
            "profile_id": profile.profile_id,
            "segment_id": int(segment_id),
            "micro_steps": int(micro_steps),
            "data_seed": int(data_seed),
            "tokenizer_fingerprint": str(tokenizer_fingerprint),
        }
    )
    rng = random.Random(shuffle_seed)
    rng.shuffle(source_sequence)

    step_payload = []
    steps: List[MicroStepPlan] = []
    for micro_step_idx, source_id in enumerate(source_sequence):
        sample_key = _stable_hash(
            {
                "segment_id": int(segment_id),
                "micro_step_idx": int(micro_step_idx),
                "source_id": str(source_id),
                "data_seed": int(data_seed),
            }
        )
        steps.append(
            MicroStepPlan(
                micro_step_idx=int(micro_step_idx),
                source_id=str(source_id),
                target_tokens=int(tokens_per_micro_step),
                sample_key=sample_key,
            )
        )
        step_payload.append(
            {
                "micro_step_idx": int(micro_step_idx),
                "source_id": str(source_id),
                "target_tokens": int(tokens_per_micro_step),
                "sample_key": sample_key,
            }
        )

    plan_hash = _stable_hash(
        {
            "profile_id": profile.profile_id,
            "segment_id": int(segment_id),
            "micro_steps": int(micro_steps),
            "tokens_per_micro_step": int(tokens_per_micro_step),
            "data_seed": int(data_seed),
            "tokenizer_fingerprint": str(tokenizer_fingerprint),
            "steps": step_payload,
        }
    )
    dataset_slice_id = "slice-" + _stable_hash(
        {
            "profile_id": profile.profile_id,
            "tokenizer_fingerprint": str(tokenizer_fingerprint),
            "segment_id": int(segment_id),
            "seed": int(data_seed),
            "plan_hash": plan_hash,
        }
    )[:16]

    return SegmentPlan(
        segment_id=int(segment_id),
        dataset_slice_id=dataset_slice_id,
        plan_hash=plan_hash,
        steps=tuple(steps),
        total_target_tokens=int(micro_steps) * int(tokens_per_micro_step),
    )


def build_hybrid_schedule(
    *,
    segment_id: int,
    micro_steps: int,
    pure_ratio: float,
    data_seed: int,
) -> HybridSchedule:
    if micro_steps <= 0:
        raise ValueError("micro_steps must be positive")
    pure_steps = int(round(float(micro_steps) * float(pure_ratio)))
    pure_steps = min(max(pure_steps, 0), int(micro_steps))
    indices = list(range(micro_steps))
    rng = random.Random(
        _stable_seed(
            {
                "segment_id": int(segment_id),
                "micro_steps": int(micro_steps),
                "pure_ratio": float(pure_ratio),
                "data_seed": int(data_seed),
            }
        )
    )
    rng.shuffle(indices)
    pure_index_set = set(indices[:pure_steps])
    flags = tuple(index in pure_index_set for index in range(micro_steps))
    return HybridSchedule(
        segment_id=int(segment_id),
        pure_step_flags=flags,
        pure_steps=int(sum(1 for flag in flags if flag)),
        synthetic_steps=int(sum(1 for flag in flags if not flag)),
    )
