from __future__ import annotations

from typing import Iterable, List

from .types import ArcPair, ArcTask


def build_rearc_pairs(
    tasks: Iterable[ArcTask],
    *,
    pairing_policy: str = "adjacent",
) -> List[ArcPair]:
    if pairing_policy != "adjacent":
        raise ValueError("Unsupported pairing_policy. Expected 'adjacent'.")

    pairs: List[ArcPair] = []
    for task in tasks:
        examples = list(task.all_examples)
        if len(examples) < 2:
            continue
        pair_count = len(examples) // 2
        for pair_index in range(pair_count):
            left = examples[pair_index * 2]
            right = examples[pair_index * 2 + 1]
            pairs.append(
                ArcPair(
                    task_id=task.task_id,
                    pair_index=pair_index,
                    left=left,
                    right=right,
                    source_path=task.source_path,
                    pairing_policy=pairing_policy,
                )
            )
    return pairs

