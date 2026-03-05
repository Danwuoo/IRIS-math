from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np

from ..schema import StateIR

_LEVEL_IDS = tuple(f"L{index}" for index in range(7))
_FAILURE_CODES = ("F_REP", "F_PROC", "F_SEARCH", "F_MEM", "F_ABS", "F_EVAL")
_FAILURE_CODE_LEVEL_MAP = {
    "F_REP": ("L0", "L1"),
    "F_PROC": ("L2",),
    "F_SEARCH": ("L3",),
    "F_MEM": ("L4",),
    "F_ABS": ("L5",),
    "F_EVAL": ("L6",),
}


def _normalize_failure_histogram(histogram: Mapping[str, float]) -> Dict[str, float]:
    values = {code: float(histogram.get(code, 0.0)) for code in _FAILURE_CODES}
    total = float(sum(values.values()))
    if total <= 0.0:
        return {code: 0.0 for code in _FAILURE_CODES}
    return {code: values[code] / total for code in _FAILURE_CODES}


def _failure_credit_to_code_distribution(failure_credit: Mapping[str, float]) -> Dict[str, float]:
    distribution: Dict[str, float] = {}
    for code, levels in _FAILURE_CODE_LEVEL_MAP.items():
        distribution[code] = float(sum(float(failure_credit.get(level, 0.0)) for level in levels))
    return _normalize_failure_histogram(distribution)


def _dominant_failure_code(histogram: Mapping[str, float]) -> str:
    normalized = _normalize_failure_histogram(histogram)
    return max(_FAILURE_CODES, key=lambda code: float(normalized.get(code, 0.0)))


def neutral_failure_credit() -> Dict[str, float]:
    weight = 1.0 / len(_LEVEL_IDS)
    return {level_id: weight for level_id in _LEVEL_IDS}


def validate_failure_credit(failure_credit: Mapping[str, float]) -> Dict[str, float]:
    keys = set(failure_credit.keys())
    if keys != set(_LEVEL_IDS):
        raise ValueError(f"failure.credit must use {_LEVEL_IDS}, got {sorted(keys)}.")

    values = np.asarray([float(failure_credit[level_id]) for level_id in _LEVEL_IDS], dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("failure.credit values must stay inside [0, 1].")
    if not np.isclose(float(np.sum(values)), 1.0, atol=1e-6):
        raise ValueError("failure.credit values must sum to 1.")
    return {level_id: float(failure_credit[level_id]) for level_id in _LEVEL_IDS}


def _normalized_entropy(values: np.ndarray) -> float:
    clipped = np.clip(values, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    return float(entropy / math.log(len(values)))


def build_canonical_metrics(
    state: StateIR,
    failure_credit: Mapping[str, float] | None = None,
    task_validity_score: float = 0.0,
    task_confidence: float = 0.0,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    extra = dict(extra or {})
    credit = validate_failure_credit(failure_credit or neutral_failure_credit())
    credit_values = np.asarray([credit[level_id] for level_id in _LEVEL_IDS], dtype=np.float64)
    failure_code_distribution = _failure_credit_to_code_distribution(credit)
    failure_code_dominant = _dominant_failure_code(failure_code_distribution)
    lengths = state.section_lengths()
    metrics = {
        "task.success": bool(task_validity_score >= 0.5),
        "task.validity_score": float(task_validity_score),
        "task.confidence": float(task_confidence),
        "cost.total_steps": int(extra.get("cost.total_steps", 0)),
        "cost.program_proposals": int(extra.get("cost.program_proposals", 0)),
        "cost.rollout_steps": int(extra.get("cost.rollout_steps", 0)),
        "cost.retrieval_calls": int(extra.get("cost.retrieval_calls", 0)),
        "failure.credit": credit,
        "failure.credit.collapse_rate": float(np.max(credit_values)),
        "failure.code_distribution": failure_code_distribution,
        "failure.code_dominant": failure_code_dominant,
        "rep.object.count": lengths["O"],
        "rep.relation.count": lengths["R"],
        "rep.event.count": lengths["X"],
        "rep.object.entropy": float(extra.get("rep.object.entropy", 0.0)),
        "dyn.violation_score": float(extra.get("dyn.violation_score", 0.0)),
        "dyn.uncertainty": float(extra.get("dyn.uncertainty", 0.0)),
        "rep.tokenizer.unk_rate": float(extra.get("rep.tokenizer.unk_rate", 0.0)),
        "rep.tokenizer.ir_fragmentation_rate": float(
            extra.get("rep.tokenizer.ir_fragmentation_rate", 0.0)
        ),
        "prog.count": int(extra.get("prog.count", 0)),
        "prog.diversity": float(extra.get("prog.diversity", 0.0)),
        "prog.exec.success_rate": float(extra.get("prog.exec.success_rate", 0.0)),
        "prog.exec.instability": float(extra.get("prog.exec.instability", 0.0)),
        "prog.score.spread": float(extra.get("prog.score.spread", 0.0)),
        "search.depth.max": int(extra.get("search.depth.max", 0)),
        "search.termination_margin": float(extra.get("search.termination_margin", 0.0)),
        "search.retry_count": int(extra.get("search.retry_count", 0)),
        "search.budget_pressure": float(extra.get("search.budget_pressure", 0.0)),
        "mem.read.k": int(extra.get("mem.read.k", 0)),
        "mem.read.similarity": float(extra.get("mem.read.similarity", 0.0)),
        "mem.write.gate": float(extra.get("mem.write.gate", 0.0)),
        "mem.consolidation.action": str(extra.get("mem.consolidation.action", "ignore")),
        "abs.macro.count": lengths["M"],
        "abs.granularity": float(extra.get("abs.granularity", 0.0)),
        "abs.override_rate": float(extra.get("abs.override_rate", 0.0)),
        "eval.false_accept_rate": float(extra.get("eval.false_accept_rate", 0.0)),
        "eval.false_reject_rate": float(extra.get("eval.false_reject_rate", 0.0)),
        "eval.calibration_error": float(extra.get("eval.calibration_error", 0.0)),
        "eval.disagreement": float(extra.get("eval.disagreement", 0.0)),
        "concept.leakage_score": float(extra.get("concept.leakage_score", 0.0)),
        "paired.invariance.gap": float(extra.get("paired.invariance.gap", 0.0)),
        "process.failure_distribution_entropy": _normalized_entropy(credit_values),
    }
    for key, value in extra.items():
        if key not in metrics:
            metrics[key] = value
    return metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MutableMapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(dict(record)), sort_keys=True) + "\n")
