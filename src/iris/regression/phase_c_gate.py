from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_LEVEL_IDS: Tuple[str, ...] = tuple(f"L{index}" for index in range(7))
_REQUIRED_S8_PATHS: Tuple[str, ...] = (
    "uninterrupted",
    "execute_crash",
    "pre_commit_crash",
    "post_commit_crash",
)
_S7_MONITORED_METRICS: Tuple[str, ...] = (
    "failure.credit.collapse_rate",
    "eval.calibration_error",
    "rep.tokenizer.ir_fragmentation_rate",
    "process.failure_distribution_entropy",
)
_S8_MONITORED_METRICS: Tuple[str, ...] = (
    "task.validity_score",
    "task.confidence",
    "failure.credit.collapse_rate",
    "eval.calibration_error",
    "rep.tokenizer.ir_fragmentation_rate",
    "paired.invariance.gap",
    "concept.leakage_score",
    "process.failure_distribution_entropy",
)
_MANDATORY_DOCS: Tuple[str, ...] = (
    "docs/數學模型建議.md",
    "docs/00_INDEX.md",
    "docs/10_Glossary_and_Normative_Status.md",
    "docs/13_Goals_and_Success_Criteria.md",
    "docs/07_Data_Constitution.md",
    "docs/01_Architecture_Constitution.md",
    "docs/02_State_IR_Spec.md",
    "docs/03_Level_Contracts_L0-L6.md",
    "docs/04_Credit_Assignment_and_Recovery.md",
    "docs/05_Eval_Metrics_Spec.md",
    "docs/06_Regression_and_Phase_Gates.md",
    "docs/08_Training_Run_Governance.md",
    "docs/09_Training_Profiles_and_Scaling.md",
    "docs/14_Multimodal_Document_Pipeline.md",
    "docs/15_Benchmark_Registry_and_Tiering_Playbook.md",
    "docs/16_Verifier_and_Formalization_Stack.md",
    "docs/17_Scaling_Promotion_and_Readiness.md",
)


@dataclass(frozen=True)
class GateContext:
    phase: str = "C"
    baseline_id: str = "toy-baseline"
    tolerance_profile_id: str = "toy-default"
    change_class: str = "Capability expansion (IRIS-math v2 documentation-first transition)"
    mandatory_docs_consulted: Tuple[str, ...] = _MANDATORY_DOCS


@dataclass(frozen=True)
class Tolerances:
    metric_epsilon: float = 1e-6
    failure_profile_kl_epsilon: float = 1e-6
    failure_credit_delta_epsilon: float = 1e-6
    concept_isolation_delta_epsilon: float = 1e-6
    concept_leakage_delta_epsilon: float = 1e-6
    paired_asymmetry_delta_epsilon: float = 1e-6
    paired_invariance_gap_delta_epsilon: float = 1e-6
    s8_metric_delta_epsilon: float = 1e-6


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_phase_c_or_later(phase: str) -> bool:
    return str(phase).strip().upper() in {"C", "D", "E"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return dict(payload)
    return {"data": payload}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(dict(parsed))
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(dict(payload)), sort_keys=True, indent=2)
    path.write_text(text, encoding="utf-8")


def _coerce_grid(value: Any) -> List[List[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    grid: List[List[int]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            continue
        grid.append([int(_safe_float(cell, 0.0)) for cell in row])
    return grid


def _cell_accuracy(input_grid: Any, output_grid: Any) -> float:
    left = _coerce_grid(input_grid)
    right = _coerce_grid(output_grid)
    max_rows = max(len(left), len(right))
    max_cols = max(
        max((len(row) for row in left), default=0),
        max((len(row) for row in right), default=0),
    )
    if max_rows == 0 or max_cols == 0:
        return 1.0
    matches = 0
    total = max_rows * max_cols
    for row_idx in range(max_rows):
        for col_idx in range(max_cols):
            left_value = 0
            right_value = 0
            if row_idx < len(left) and col_idx < len(left[row_idx]):
                left_value = int(left[row_idx][col_idx])
            if row_idx < len(right) and col_idx < len(right[row_idx]):
                right_value = int(right[row_idx][col_idx])
            if left_value == right_value:
                matches += 1
    return float(matches) / float(total)


def _canonical_credit_vector(raw_credit: Any) -> Dict[str, float]:
    if not isinstance(raw_credit, Mapping):
        return {}
    if any(level_id not in raw_credit for level_id in _LEVEL_IDS):
        return {}
    values = [_safe_float(raw_credit.get(level_id, 0.0)) for level_id in _LEVEL_IDS]
    total = float(sum(values))
    if total <= 0.0:
        return {}
    return {level_id: float(values[idx] / total) for idx, level_id in enumerate(_LEVEL_IDS)}


def _l1_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return float(
        sum(abs(_safe_float(left.get(level), 0.0) - _safe_float(right.get(level), 0.0)) for level in _LEVEL_IDS)
    )


def _max_abs_delta(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return float(
        max(abs(_safe_float(left.get(level), 0.0) - _safe_float(right.get(level), 0.0)) for level in _LEVEL_IDS)
    )


def _collapse_rate(credit_vector: Mapping[str, float]) -> float:
    if not credit_vector:
        return 0.0
    return float(max(_safe_float(credit_vector.get(level), 0.0) for level in _LEVEL_IDS))


def _vector_from_credit(credit_vector: Mapping[str, float]) -> List[float]:
    return [_safe_float(credit_vector.get(level), 0.0) for level in _LEVEL_IDS]


def _kl_divergence(p_values: Sequence[float], q_values: Sequence[float], epsilon: float = 1e-12) -> float:
    if len(p_values) != len(q_values) or not p_values:
        return float("inf")
    p_safe = [max(_safe_float(value), epsilon) for value in p_values]
    q_safe = [max(_safe_float(value), epsilon) for value in q_values]
    p_total = float(sum(p_safe))
    q_total = float(sum(q_safe))
    if p_total <= 0.0 or q_total <= 0.0:
        return float("inf")
    p_norm = [value / p_total for value in p_safe]
    q_norm = [value / q_total for value in q_safe]
    kl = 0.0
    for p_item, q_item in zip(p_norm, q_norm):
        kl += p_item * math.log(p_item / q_item)
    return float(kl)


def _metric_delta_dict(
    current_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    keys: Iterable[str],
) -> Dict[str, float]:
    return {
        key: abs(_safe_float(current_metrics.get(key, 0.0)) - _safe_float(baseline_metrics.get(key, 0.0)))
        for key in keys
    }


def build_concept_breakdown(conceptarc_corpus: Path, context: GateContext) -> Dict[str, Any]:
    conceptarc_corpus = Path(conceptarc_corpus)
    concepts: List[Dict[str, Any]] = []
    if not conceptarc_corpus.exists():
        return {
            "schema": "iris.regression.concept_breakdown/v1",
            "phase": context.phase,
            "baseline_id": context.baseline_id,
            "tolerance_profile_id": context.tolerance_profile_id,
            "status": "FAIL",
            "reason": f"ConceptARC corpus path does not exist: {conceptarc_corpus}",
            "concept_count": 0,
            "aggregate": {
                "concept.isolation_score.mean": 0.0,
                "concept.leakage_score.mean": 0.0,
            },
            "concepts": [],
        }

    for concept_dir in sorted(path for path in conceptarc_corpus.iterdir() if path.is_dir()):
        task_files = sorted(concept_dir.glob("*.json"))
        if not task_files:
            continue
        isolation_total = 0.0
        exact_matches = 0
        test_case_count = 0
        for task_file in task_files:
            try:
                payload = json.loads(task_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            task_cases: List[Any] = []
            if isinstance(payload, dict):
                if isinstance(payload.get("test"), list):
                    task_cases.extend(payload["test"])
                elif isinstance(payload.get("train"), list):
                    task_cases.extend(payload["train"])
            elif isinstance(payload, list):
                task_cases.extend(payload)
            for case in task_cases:
                if not isinstance(case, Mapping):
                    continue
                accuracy = _cell_accuracy(case.get("input"), case.get("output"))
                isolation_total += accuracy
                test_case_count += 1
                if math.isclose(accuracy, 1.0, rel_tol=0.0, abs_tol=1e-9):
                    exact_matches += 1
        if test_case_count == 0:
            continue
        isolation_score = isolation_total / float(test_case_count)
        leakage_score = max(0.0, 1.0 - isolation_score)
        concepts.append(
            {
                "concept_id": concept_dir.name,
                "task_count": len(task_files),
                "test_case_count": test_case_count,
                "proxy.exact_match_rate": float(exact_matches) / float(test_case_count),
                "concept.isolation_score": float(isolation_score),
                "concept.leakage_score": float(leakage_score),
            }
        )

    concept_count = len(concepts)
    isolation_mean = (
        sum(_safe_float(item.get("concept.isolation_score"), 0.0) for item in concepts) / float(concept_count)
        if concept_count
        else 0.0
    )
    leakage_mean = (
        sum(_safe_float(item.get("concept.leakage_score"), 0.0) for item in concepts) / float(concept_count)
        if concept_count
        else 0.0
    )
    status = "PASS" if concept_count > 0 else "FAIL"
    report: Dict[str, Any] = {
        "schema": "iris.regression.concept_breakdown/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "status": status,
        "concept_count": concept_count,
        "aggregate": {
            "concept.isolation_score.mean": float(isolation_mean),
            "concept.leakage_score.mean": float(leakage_mean),
        },
        "concepts": concepts,
    }
    if status != "PASS":
        report["reason"] = "No concept tasks could be parsed from ConceptARC corpus."
    return report


def build_paired_representation_diff(
    rearc_tasks: Path,
    context: GateContext,
    *,
    max_tasks: int = 128,
    max_pairs_per_task: int = 4,
) -> Dict[str, Any]:
    rearc_tasks = Path(rearc_tasks)
    if not rearc_tasks.exists():
        return {
            "schema": "iris.regression.paired_representation_diff/v1",
            "phase": context.phase,
            "baseline_id": context.baseline_id,
            "tolerance_profile_id": context.tolerance_profile_id,
            "status": "FAIL",
            "reason": f"re_arc tasks path does not exist: {rearc_tasks}",
            "pair_count": 0,
            "paired.asymmetry_rate": 1.0,
            "paired.invariance.gap": 1.0,
            "pairs": [],
        }

    task_files = sorted(rearc_tasks.glob("*.json"))
    if max_tasks > 0:
        task_files = task_files[:max_tasks]
    pairs: List[Dict[str, Any]] = []
    for task_file in task_files:
        try:
            payload = json.loads(task_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cases: List[Any] = []
        if isinstance(payload, list):
            cases.extend(payload)
        elif isinstance(payload, dict):
            if isinstance(payload.get("train"), list):
                cases.extend(payload["train"])
            if isinstance(payload.get("test"), list):
                cases.extend(payload["test"])
        usable_pair_count = len(cases) // 2
        if max_pairs_per_task > 0:
            usable_pair_count = min(usable_pair_count, max_pairs_per_task)
        for pair_index in range(usable_pair_count):
            left_case = cases[pair_index * 2]
            right_case = cases[pair_index * 2 + 1]
            if not isinstance(left_case, Mapping) or not isinstance(right_case, Mapping):
                continue
            left_accuracy = _cell_accuracy(left_case.get("input"), left_case.get("output"))
            right_accuracy = _cell_accuracy(right_case.get("input"), right_case.get("output"))
            invariance_gap = abs(left_accuracy - right_accuracy)
            asymmetry = (left_accuracy >= 0.5) != (right_accuracy >= 0.5)
            pairs.append(
                {
                    "task_id": task_file.stem,
                    "pair_index": pair_index,
                    "left.cell_accuracy": float(left_accuracy),
                    "right.cell_accuracy": float(right_accuracy),
                    "pair.invariance_gap": float(invariance_gap),
                    "pair.asymmetry": bool(asymmetry),
                }
            )

    pair_count = len(pairs)
    asymmetry_rate = (
        sum(1.0 for item in pairs if bool(item.get("pair.asymmetry", False))) / float(pair_count)
        if pair_count
        else 1.0
    )
    invariance_gap = (
        sum(_safe_float(item.get("pair.invariance_gap"), 0.0) for item in pairs) / float(pair_count)
        if pair_count
        else 1.0
    )
    status = "PASS" if pair_count > 0 else "FAIL"
    report = {
        "schema": "iris.regression.paired_representation_diff/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "status": status,
        "pair_count": pair_count,
        "paired.asymmetry_rate": float(asymmetry_rate),
        "paired.invariance.gap": float(invariance_gap),
        "pairs": pairs,
    }
    if status != "PASS":
        report["reason"] = "No paired examples were available from re_arc tasks."
    return report

def _validate_runtime_lock_manifest(manifest: Mapping[str, Any]) -> List[str]:
    required_fields = ("schema", "created_at", "phase", "host", "python", "jax")
    errors = [f"missing field '{field}'" for field in required_fields if field not in manifest]
    if str(manifest.get("schema", "")) != "iris.runtime_lock_manifest/v1":
        errors.append("schema must be iris.runtime_lock_manifest/v1")
    return errors


def _last_applied_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for event in reversed(events):
        if str(event.get("status", "")).upper() == "APPLIED":
            return event
    return None


def _resolve_checkpoint_path(checkpoint_ref: Any, run_dir: Path) -> tuple[str, bool]:
    if not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip():
        return "", False
    candidate = Path(checkpoint_ref)
    if candidate.is_absolute():
        return str(candidate), bool(candidate.exists())

    # Support both repository-relative refs and run-dir-relative refs.
    if candidate.exists():
        return str(candidate), True

    run_relative = run_dir / candidate
    if run_relative.exists():
        return str(run_relative), True

    basename_relative = run_dir / "checkpoints" / candidate.name
    if basename_relative.exists():
        return str(basename_relative), True

    return str(run_relative), False


def _summarize_journal(events: Sequence[Mapping[str, Any]]) -> Dict[str, int | str]:
    pending_count = sum(1 for event in events if str(event.get("status", "")).upper() == "PENDING")
    applied_events = [event for event in events if str(event.get("status", "")).upper() == "APPLIED"]
    applied_count = len(applied_events)
    applied_segment0_count = sum(1 for event in applied_events if int(event.get("segment_id", -1)) == 0)
    return {
        "events": len(events),
        "pending_count": pending_count,
        "applied_count": applied_count,
        "applied_segment0_count": applied_segment0_count,
        "last_status": str(events[-1].get("status", "NONE")) if events else "NONE",
    }


def _collect_path_snapshot(
    *,
    run_id: str,
    run_dir: Path,
    require_checkpoint_ref_exists: bool,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "resume_path_id": run_id,
        "present": bool(run_dir.exists()),
        "status": "FAIL",
        "reason": "",
        "runtime_lock_manifest_id": "",
        "runtime_lock_manifest_sha256": "",
        "segment_id": -1,
        "optimizer_step_id": -1,
        "dataset_slice_id": "",
        "data_seed": None,
        "journal": {
            "events": 0,
            "pending_count": 0,
            "applied_count": 0,
            "applied_segment0_count": 0,
            "last_status": "NONE",
        },
        "checkpoint_ref": "",
        "checkpoint_ref_exists": False,
        "credit_vector": {},
        "metric_deltas_vs_uninterrupted": {metric: 0.0 for metric in _S8_MONITORED_METRICS},
        "failure_profile_kl_vs_uninterrupted": 0.0,
        "failure_credit_l1_distance": 0.0,
        "failure_credit_max_abs_delta": 0.0,
        "drift_diagnosis_labels": {
            "runtime_drift": False,
            "rng_drift": False,
            "data_slice_drift": False,
            "optimizer_state_drift": False,
        },
        "_metric_values": {},
        "_rng_hash_pre": "",
        "_rng_hash_post": "",
    }
    if not run_dir.exists():
        snapshot["reason"] = f"Run path missing: {run_dir}"
        return snapshot

    reasons: List[str] = []

    manifest_path = run_dir / "runtime_lock_manifest.json"
    if not manifest_path.exists():
        reasons.append("runtime_lock_manifest.json missing")
    else:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest_sha = _sha256_text(manifest_text)
        snapshot["runtime_lock_manifest_sha256"] = manifest_sha
        snapshot["runtime_lock_manifest_id"] = manifest_sha[:12]
        try:
            manifest_payload = json.loads(manifest_text)
        except json.JSONDecodeError:
            reasons.append("runtime_lock_manifest.json is invalid JSON")
            manifest_payload = {}
        if isinstance(manifest_payload, Mapping):
            for error in _validate_runtime_lock_manifest(manifest_payload):
                reasons.append(f"runtime_lock_manifest: {error}")

    journal_path = run_dir / "segment_journal.jsonl"
    events = _read_jsonl(journal_path)
    snapshot["journal"] = _summarize_journal(events)
    applied_event = _last_applied_event(events)
    if applied_event is None:
        reasons.append("no APPLIED journal event")
    else:
        resolved_checkpoint_ref, checkpoint_exists = _resolve_checkpoint_path(
            applied_event.get("checkpoint_ref"),
            run_dir,
        )
        snapshot["checkpoint_ref"] = resolved_checkpoint_ref
        snapshot["checkpoint_ref_exists"] = checkpoint_exists
        if require_checkpoint_ref_exists and not checkpoint_exists:
            reasons.append("checkpoint_ref missing or unresolved for APPLIED event")

    metrics_path = run_dir / "metrics.jsonl"
    metric_rows = _read_jsonl(metrics_path)
    metric_row: Mapping[str, Any] = metric_rows[-1] if metric_rows else {}
    if not metric_rows:
        reasons.append("metrics.jsonl missing or empty")

    metric_lock_id = str(metric_row.get("runtime_lock_manifest_id", "")).strip()
    metric_lock_sha = str(metric_row.get("runtime_lock_manifest_sha256", "")).strip()
    if metric_lock_id:
        snapshot["runtime_lock_manifest_id"] = metric_lock_id
    if metric_lock_sha:
        snapshot["runtime_lock_manifest_sha256"] = metric_lock_sha
    if not snapshot["runtime_lock_manifest_id"]:
        reasons.append("runtime_lock_manifest_id missing")
    if not snapshot["runtime_lock_manifest_sha256"]:
        reasons.append("runtime_lock_manifest_sha256 missing")

    snapshot["segment_id"] = int(metric_row.get("segment_id", -1))
    snapshot["optimizer_step_id"] = int(metric_row.get("optimizer_step_id", -1))
    snapshot["dataset_slice_id"] = str(metric_row.get("dataset_slice_id", ""))
    data_seed_value = metric_row.get("data_seed")
    snapshot["data_seed"] = int(data_seed_value) if data_seed_value is not None else None
    snapshot["_rng_hash_pre"] = str(metric_row.get("rng_hash_pre", ""))
    snapshot["_rng_hash_post"] = str(metric_row.get("rng_hash_post", ""))
    snapshot["_metric_values"] = {
        metric_name: _safe_float(metric_row.get(metric_name, 0.0)) for metric_name in _S8_MONITORED_METRICS
    }
    snapshot["credit_vector"] = _canonical_credit_vector(metric_row.get("failure.credit", {}))
    if not snapshot["credit_vector"]:
        reasons.append("failure.credit vector missing or invalid")

    snapshot["status"] = "PASS" if not reasons else "FAIL"
    if reasons:
        snapshot["reason"] = "; ".join(reasons)
    return snapshot


def _populate_resume_deltas(path_snapshots: List[Dict[str, Any]]) -> bool:
    baseline = next((item for item in path_snapshots if item.get("resume_path_id") == "uninterrupted"), None)
    if baseline is None:
        return False
    baseline_metrics = baseline.get("_metric_values", {})
    baseline_credit = baseline.get("credit_vector", {})
    baseline_runtime_id = str(baseline.get("runtime_lock_manifest_id", ""))
    baseline_runtime_sha = str(baseline.get("runtime_lock_manifest_sha256", ""))
    baseline_dataset_slice = str(baseline.get("dataset_slice_id", ""))
    baseline_seed = baseline.get("data_seed")
    baseline_optimizer_step = int(baseline.get("optimizer_step_id", -1))
    baseline_rng_pre = str(baseline.get("_rng_hash_pre", ""))
    baseline_rng_post = str(baseline.get("_rng_hash_post", ""))

    for snapshot in path_snapshots:
        metric_values = snapshot.get("_metric_values", {})
        credit_vector = snapshot.get("credit_vector", {})
        snapshot["metric_deltas_vs_uninterrupted"] = _metric_delta_dict(
            metric_values,
            baseline_metrics,
            _S8_MONITORED_METRICS,
        )
        snapshot["failure_credit_l1_distance"] = _l1_distance(credit_vector, baseline_credit)
        snapshot["failure_credit_max_abs_delta"] = _max_abs_delta(credit_vector, baseline_credit)
        snapshot["failure_profile_kl_vs_uninterrupted"] = _kl_divergence(
            _vector_from_credit(credit_vector),
            _vector_from_credit(baseline_credit),
        )
        labels = {
            "runtime_drift": (
                str(snapshot.get("runtime_lock_manifest_id", "")) != baseline_runtime_id
                or str(snapshot.get("runtime_lock_manifest_sha256", "")) != baseline_runtime_sha
            ),
            "rng_drift": (
                str(snapshot.get("_rng_hash_pre", "")) != baseline_rng_pre
                or str(snapshot.get("_rng_hash_post", "")) != baseline_rng_post
            ),
            "data_slice_drift": (
                str(snapshot.get("dataset_slice_id", "")) != baseline_dataset_slice
                or snapshot.get("data_seed") != baseline_seed
            ),
            "optimizer_state_drift": int(snapshot.get("optimizer_step_id", -1)) != baseline_optimizer_step,
        }
        if snapshot.get("resume_path_id") == "uninterrupted":
            labels = {key: False for key in labels}
        snapshot["drift_diagnosis_labels"] = labels
    return True


def _build_resume_consistency_packet_internal(
    path_map: Mapping[str, Any],
    context: GateContext,
    *,
    require_checkpoint_ref_exists: bool,
) -> Dict[str, Any]:
    snapshots: List[Dict[str, Any]] = []
    for run_id in _REQUIRED_S8_PATHS:
        raw_path = path_map.get(run_id)
        run_dir = Path(raw_path) if raw_path is not None else Path("__missing_path__")
        snapshots.append(
            _collect_path_snapshot(
                run_id=run_id,
                run_dir=run_dir,
                require_checkpoint_ref_exists=require_checkpoint_ref_exists,
            )
        )

    _populate_resume_deltas(snapshots)

    coverage_matrix = {
        run_id: bool(next((item for item in snapshots if item.get("resume_path_id") == run_id), {}).get("present", False))
        for run_id in _REQUIRED_S8_PATHS
    }
    path_block_reasons = [
        f"{item.get('resume_path_id')}: {item.get('reason')}"
        for item in snapshots
        if str(item.get("status")) != "PASS"
    ]
    all_drift_labels_clear = all(
        not any(bool(value) for value in item.get("drift_diagnosis_labels", {}).values())
        for item in snapshots
    )
    packet_status = (
        "PASS"
        if all(coverage_matrix.values()) and not path_block_reasons and all_drift_labels_clear
        else "FAIL"
    )

    public_paths: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        public_snapshot = dict(snapshot)
        public_snapshot.pop("_metric_values", None)
        public_snapshot.pop("_rng_hash_pre", None)
        public_snapshot.pop("_rng_hash_post", None)
        public_paths.append(public_snapshot)

    return {
        "schema": "iris.regression.resume_consistency_packet/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "coverage_matrix": coverage_matrix,
        "all_drift_labels_clear": bool(all_drift_labels_clear),
        "paths": public_paths,
        "status": packet_status,
        "block_reasons": path_block_reasons,
    }


def build_resume_consistency_packet(
    path_map: Mapping[str, Any],
    context: GateContext,
) -> Dict[str, Any]:
    return _build_resume_consistency_packet_internal(
        path_map=path_map,
        context=context,
        require_checkpoint_ref_exists=True,
    )


def build_failure_profile_diff(
    resume_packet: Mapping[str, Any],
    context: GateContext,
) -> Dict[str, Any]:
    path_rows = []
    max_kl = 0.0
    for path_entry in resume_packet.get("paths", []):
        if str(path_entry.get("resume_path_id")) == "uninterrupted":
            continue
        kl_value = _safe_float(path_entry.get("failure_profile_kl_vs_uninterrupted"), float("inf"))
        if math.isfinite(kl_value):
            max_kl = max(max_kl, kl_value)
        path_rows.append(
            {
                "resume_path_id": str(path_entry.get("resume_path_id", "")),
                "failure_profile_kl_vs_uninterrupted": kl_value,
                "status": str(path_entry.get("status", "FAIL")),
            }
        )
    return {
        "schema": "iris.regression.failure_profile_diff/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "paths": path_rows,
        "max_failure_profile_kl": float(max_kl),
        "status": "PASS" if path_rows else "FAIL",
    }


def build_credit_routing_diff(
    resume_packet: Mapping[str, Any],
    context: GateContext,
) -> Dict[str, Any]:
    baseline = next(
        (entry for entry in resume_packet.get("paths", []) if str(entry.get("resume_path_id")) == "uninterrupted"),
        None,
    )
    baseline_collapse = _collapse_rate(dict(baseline.get("credit_vector", {})) if isinstance(baseline, Mapping) else {})
    path_rows = []
    max_l1 = 0.0
    for path_entry in resume_packet.get("paths", []):
        if str(path_entry.get("resume_path_id")) == "uninterrupted":
            continue
        l1_distance = _safe_float(path_entry.get("failure_credit_l1_distance"), float("inf"))
        max_abs_delta = _safe_float(path_entry.get("failure_credit_max_abs_delta"), float("inf"))
        collapse_rate = _collapse_rate(dict(path_entry.get("credit_vector", {})))
        collapse_delta = collapse_rate - baseline_collapse
        if math.isfinite(l1_distance):
            max_l1 = max(max_l1, l1_distance)
        path_rows.append(
            {
                "resume_path_id": str(path_entry.get("resume_path_id", "")),
                "failure_credit_l1_distance": l1_distance,
                "failure_credit_max_abs_delta": max_abs_delta,
                "failure.credit.collapse_rate": collapse_rate,
                "failure.credit.collapse_rate.delta_vs_uninterrupted": collapse_delta,
                "status": str(path_entry.get("status", "FAIL")),
            }
        )
    return {
        "schema": "iris.regression.credit_routing_diff/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "paths": path_rows,
        "max_failure_credit_l1_distance": float(max_l1),
        "status": "PASS" if path_rows else "FAIL",
    }

def evaluate_s3_status(
    failure_profile_diff: Mapping[str, Any],
    tolerances: Tolerances,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if str(failure_profile_diff.get("status")) != "PASS":
        reasons.append("failure_profile_diff artifact is unavailable")
    max_kl = _safe_float(failure_profile_diff.get("max_failure_profile_kl"), float("inf"))
    if not math.isfinite(max_kl):
        reasons.append("max_failure_profile_kl is non-finite")
    elif max_kl > tolerances.failure_profile_kl_epsilon:
        reasons.append(
            f"failure profile KL drift {max_kl:.6g} exceeds tolerance {tolerances.failure_profile_kl_epsilon:.6g}"
        )
    return ("PASS" if not reasons else "FAIL", reasons)


def evaluate_s6_status(
    resume_packet: Mapping[str, Any],
    credit_routing_diff: Mapping[str, Any],
    tolerances: Tolerances,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if str(credit_routing_diff.get("status")) != "PASS":
        reasons.append("credit_routing_diff artifact is unavailable")
    max_l1 = _safe_float(credit_routing_diff.get("max_failure_credit_l1_distance"), float("inf"))
    if not math.isfinite(max_l1):
        reasons.append("max_failure_credit_l1_distance is non-finite")
    elif max_l1 > tolerances.failure_credit_delta_epsilon:
        reasons.append(
            f"failure.credit L1 drift {max_l1:.6g} exceeds tolerance {tolerances.failure_credit_delta_epsilon:.6g}"
        )

    baseline = next(
        (entry for entry in resume_packet.get("paths", []) if str(entry.get("resume_path_id")) == "uninterrupted"),
        None,
    )
    baseline_collapse = _collapse_rate(dict(baseline.get("credit_vector", {})) if isinstance(baseline, Mapping) else {})
    for path_entry in resume_packet.get("paths", []):
        run_id = str(path_entry.get("resume_path_id", ""))
        if run_id == "uninterrupted":
            continue
        collapse_rate = _collapse_rate(dict(path_entry.get("credit_vector", {})))
        collapse_delta = collapse_rate - baseline_collapse
        if collapse_delta > tolerances.metric_epsilon:
            reasons.append(
                f"{run_id}: failure.credit.collapse_rate delta {collapse_delta:.6g} exceeds tolerance {tolerances.metric_epsilon:.6g}"
            )
        if collapse_rate >= 0.99:
            reasons.append(f"{run_id}: failure.credit collapsed to near single-level attribution ({collapse_rate:.6g})")
    return ("PASS" if not reasons else "FAIL", reasons)


def evaluate_s7_status(
    resume_packet: Mapping[str, Any],
    tolerances: Tolerances,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    comparable_paths = [
        entry for entry in resume_packet.get("paths", []) if str(entry.get("resume_path_id")) != "uninterrupted"
    ]
    if not comparable_paths:
        reasons.append("no resumed paths available for S7 comparisons")
    for path_entry in comparable_paths:
        run_id = str(path_entry.get("resume_path_id", ""))
        metric_deltas = path_entry.get("metric_deltas_vs_uninterrupted", {})
        if not isinstance(metric_deltas, Mapping):
            reasons.append(f"{run_id}: metric_deltas_vs_uninterrupted missing")
            continue
        for metric_name in _S7_MONITORED_METRICS:
            delta = _safe_float(metric_deltas.get(metric_name), 0.0)
            if abs(delta) > tolerances.metric_epsilon:
                reasons.append(
                    f"{run_id}: {metric_name} delta {delta:.6g} exceeds tolerance {tolerances.metric_epsilon:.6g}"
                )
    return ("PASS" if not reasons else "FAIL", reasons)


def evaluate_s8_status(
    resume_packet: Mapping[str, Any],
    tolerances: Tolerances | None = None,
) -> tuple[str, List[str]]:
    tolerances = tolerances or Tolerances()
    reasons: List[str] = []
    coverage_matrix = resume_packet.get("coverage_matrix", {})
    if not isinstance(coverage_matrix, Mapping):
        coverage_matrix = {}
    missing_paths = [run_id for run_id in _REQUIRED_S8_PATHS if not bool(coverage_matrix.get(run_id, False))]
    if missing_paths:
        reasons.append("missing crash-class coverage: " + ", ".join(missing_paths))

    if not bool(resume_packet.get("all_drift_labels_clear", False)):
        reasons.append("drift_diagnosis_labels indicate unresolved drift")

    paths = resume_packet.get("paths", [])
    for path_entry in paths:
        run_id = str(path_entry.get("resume_path_id", ""))
        if str(path_entry.get("status")) != "PASS":
            reasons.append(f"{run_id}: {path_entry.get('reason', 'path validation failed')}")
        if not str(path_entry.get("runtime_lock_manifest_id", "")):
            reasons.append(f"{run_id}: runtime_lock_manifest_id missing")
        if not str(path_entry.get("runtime_lock_manifest_sha256", "")):
            reasons.append(f"{run_id}: runtime_lock_manifest_sha256 missing")
        if not bool(path_entry.get("checkpoint_ref_exists", False)):
            reasons.append(f"{run_id}: checkpoint_ref is missing or not accessible")
        drift_labels = path_entry.get("drift_diagnosis_labels", {})
        if not isinstance(drift_labels, Mapping):
            reasons.append(f"{run_id}: drift_diagnosis_labels missing")
            drift_labels = {}
        required_labels = ("runtime_drift", "rng_drift", "data_slice_drift", "optimizer_state_drift")
        missing_labels = [label for label in required_labels if label not in drift_labels]
        if missing_labels:
            reasons.append(f"{run_id}: missing drift labels: {', '.join(missing_labels)}")
        for label_name, label_value in drift_labels.items():
            if bool(label_value):
                reasons.append(f"{run_id}: {label_name}=true")
        metric_deltas = path_entry.get("metric_deltas_vs_uninterrupted", {})
        if not isinstance(metric_deltas, Mapping):
            metric_deltas = {}
        for metric_name, delta in metric_deltas.items():
            delta_value = _safe_float(delta, 0.0)
            if abs(delta_value) > tolerances.s8_metric_delta_epsilon:
                reasons.append(
                    f"{run_id}: {metric_name} delta {delta_value:.6g} exceeds tolerance {tolerances.s8_metric_delta_epsilon:.6g}"
                )

    for reason in resume_packet.get("block_reasons", []) if isinstance(resume_packet.get("block_reasons"), list) else []:
        reasons.append(str(reason))

    return ("PASS" if not reasons else "FAIL", reasons)


def evaluate_s4_status(
    *,
    current_concept_breakdown: Mapping[str, Any],
    baseline_concept_breakdown: Mapping[str, Any] | None,
    tolerances: Tolerances,
    phase: str,
) -> tuple[str, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    details: List[Dict[str, Any]] = []
    current_concepts = current_concept_breakdown.get("concepts", [])
    if not isinstance(current_concepts, list) or not current_concepts:
        reasons.append("current concept_breakdown has no concept entries")
    if _is_phase_c_or_later(phase) and not isinstance(baseline_concept_breakdown, Mapping):
        reasons.append("baseline concept_breakdown is required in Phase C+")
        return "FAIL", reasons, details

    baseline_concepts = (
        baseline_concept_breakdown.get("concepts", [])
        if isinstance(baseline_concept_breakdown, Mapping)
        else []
    )
    baseline_by_id = {
        str(item.get("concept_id")): item for item in baseline_concepts if isinstance(item, Mapping)
    }

    for concept_entry in current_concepts if isinstance(current_concepts, list) else []:
        concept_id = str(concept_entry.get("concept_id", ""))
        baseline_entry = baseline_by_id.get(concept_id)
        if baseline_entry is None:
            details.append(
                {
                    "metric": "concept.baseline_presence",
                    "concept_id": concept_id,
                    "delta": None,
                    "threshold": "required",
                    "reason": "concept missing in baseline",
                }
            )
            continue
        current_isolation = _safe_float(concept_entry.get("concept.isolation_score"), 0.0)
        baseline_isolation = _safe_float(baseline_entry.get("concept.isolation_score"), 0.0)
        current_leakage = _safe_float(concept_entry.get("concept.leakage_score"), 0.0)
        baseline_leakage = _safe_float(baseline_entry.get("concept.leakage_score"), 0.0)
        isolation_delta = current_isolation - baseline_isolation
        leakage_delta = current_leakage - baseline_leakage
        if isolation_delta < -tolerances.concept_isolation_delta_epsilon:
            details.append(
                {
                    "metric": "concept.isolation_score",
                    "concept_id": concept_id,
                    "baseline": baseline_isolation,
                    "current": current_isolation,
                    "delta": isolation_delta,
                    "threshold": -tolerances.concept_isolation_delta_epsilon,
                    "reason": "isolation decreased beyond tolerance",
                }
            )
        if leakage_delta > tolerances.concept_leakage_delta_epsilon:
            details.append(
                {
                    "metric": "concept.leakage_score",
                    "concept_id": concept_id,
                    "baseline": baseline_leakage,
                    "current": current_leakage,
                    "delta": leakage_delta,
                    "threshold": tolerances.concept_leakage_delta_epsilon,
                    "reason": "leakage increased beyond tolerance",
                }
            )

    if details:
        reasons.append(f"{len(details)} concept-level tolerance violations detected")
    return ("PASS" if not reasons else "FAIL", reasons, details)


def evaluate_s5_status(
    *,
    current_paired_diff: Mapping[str, Any],
    baseline_paired_diff: Mapping[str, Any] | None,
    tolerances: Tolerances,
    phase: str,
) -> tuple[str, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    details: List[Dict[str, Any]] = []
    if _is_phase_c_or_later(phase) and not isinstance(baseline_paired_diff, Mapping):
        reasons.append("baseline paired_representation_diff is required in Phase C+")
        return "FAIL", reasons, details

    current_pair_count = int(_safe_float(current_paired_diff.get("pair_count"), 0.0))
    if current_pair_count <= 0:
        reasons.append("current paired_representation_diff has no pairs")
        return "FAIL", reasons, details

    baseline_pair_count = int(_safe_float((baseline_paired_diff or {}).get("pair_count"), 0.0))
    if baseline_pair_count <= 0:
        reasons.append("baseline paired_representation_diff has no pairs")
        return "FAIL", reasons, details

    current_asymmetry = _safe_float(current_paired_diff.get("paired.asymmetry_rate"), 0.0)
    baseline_asymmetry = _safe_float((baseline_paired_diff or {}).get("paired.asymmetry_rate"), 0.0)
    asymmetry_delta = current_asymmetry - baseline_asymmetry
    if asymmetry_delta > tolerances.paired_asymmetry_delta_epsilon:
        details.append(
            {
                "metric": "paired.asymmetry_rate",
                "baseline": baseline_asymmetry,
                "current": current_asymmetry,
                "delta": asymmetry_delta,
                "threshold": tolerances.paired_asymmetry_delta_epsilon,
                "reason": "asymmetry rate increased beyond tolerance",
            }
        )

    current_gap = _safe_float(current_paired_diff.get("paired.invariance.gap"), 0.0)
    baseline_gap = _safe_float((baseline_paired_diff or {}).get("paired.invariance.gap"), 0.0)
    gap_delta = current_gap - baseline_gap
    if gap_delta > tolerances.paired_invariance_gap_delta_epsilon:
        details.append(
            {
                "metric": "paired.invariance.gap",
                "baseline": baseline_gap,
                "current": current_gap,
                "delta": gap_delta,
                "threshold": tolerances.paired_invariance_gap_delta_epsilon,
                "reason": "invariance gap increased beyond tolerance",
            }
        )

    if details:
        reasons.append(f"{len(details)} paired-representation tolerance violations detected")
    return ("PASS" if not reasons else "FAIL", reasons, details)


def build_h100_packet_summary(
    path_map: Mapping[str, Any],
    context: GateContext,
    tolerances: Tolerances | None = None,
) -> Dict[str, Any]:
    h100_packet = _build_resume_consistency_packet_internal(
        path_map=path_map,
        context=context,
        require_checkpoint_ref_exists=True,
    )
    s8_status, s8_reasons = evaluate_s8_status(h100_packet, tolerances=tolerances)
    path_by_id = {
        str(path_entry.get("resume_path_id", "")): path_entry for path_entry in h100_packet.get("paths", [])
    }
    baseline_entry = path_by_id.get("uninterrupted", {})
    execute_entry = path_by_id.get("execute_crash", {})
    runtime_match = (
        str(execute_entry.get("runtime_lock_manifest_id", "")) == str(baseline_entry.get("runtime_lock_manifest_id", ""))
        and str(execute_entry.get("runtime_lock_manifest_sha256", ""))
        == str(baseline_entry.get("runtime_lock_manifest_sha256", ""))
    )
    if not runtime_match:
        s8_reasons.append("execute_crash runtime lock does not match uninterrupted path")
    packet_status = "PASS" if not s8_reasons and s8_status == "PASS" else "FAIL"
    unique_reasons = list(dict.fromkeys(reason for reason in s8_reasons if reason))
    return {
        "schema": "iris.regression.s8_h100_packet_summary/v1",
        "uninterrupted_artifact": str(path_map.get("uninterrupted", "")),
        "execute_crash_artifact": str(path_map.get("execute_crash", "")),
        "pre_commit_crash_artifact": str(path_map.get("pre_commit_crash", "")),
        "post_commit_crash_artifact": str(path_map.get("post_commit_crash", "")),
        "coverage_matrix": dict(h100_packet.get("coverage_matrix", {})),
        "runtime_lock_manifest_match_execute_vs_uninterrupted": bool(runtime_match),
        "metric_deltas_execute_vs_uninterrupted": dict(execute_entry.get("metric_deltas_vs_uninterrupted", {})),
        "failure_credit_l1_distance_execute_vs_uninterrupted": _safe_float(
            execute_entry.get("failure_credit_l1_distance"),
            0.0,
        ),
        "s8_status_for_h100_packet": packet_status,
        "block_reason": "; ".join(unique_reasons) if unique_reasons else None,
        "details": h100_packet,
    }

def build_summary_report(
    *,
    context: GateContext,
    suite_status: Mapping[str, str],
    violations: Sequence[Mapping[str, Any]],
    generated_at_utc: str,
    notes: Sequence[str],
    evidence_paths: Sequence[str],
) -> Dict[str, Any]:
    all_suites_pass = all(str(value).upper() == "PASS" for value in suite_status.values())
    all_violations = [dict(item) for item in violations]
    overall_status = "PASS" if all_suites_pass and not all_violations else "FAIL"
    termination = "Done" if overall_status == "PASS" else "Blocked"
    return {
        "schema": "iris.regression.summary_report/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "change_class": context.change_class,
        "generated_at_utc": generated_at_utc,
        "suite_status": dict(suite_status),
        "regression.status": overall_status,
        "regression.violations": all_violations,
        "notes": list(notes),
        "completion_checklist": {
            "mandatory_docs_consulted": list(context.mandatory_docs_consulted),
            "expected_failure_category_impact": "Targeted closure for Phase C gate blocking failures.",
            "technical_debt_guardrails_introduced": "none",
            "regression_evidence_paths": list(evidence_paths),
            "termination": termination,
        },
    }


def _build_markdown_report(
    *,
    summary_report: Mapping[str, Any],
    resume_packet: Mapping[str, Any],
    h100_packet: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Phase C Gate Report (Strict JAX)")
    lines.append("")
    lines.append("- Document Type: Design Note (Non-normative)")
    lines.append(f"- Generated At (UTC): {summary_report.get('generated_at_utc', '')}")
    lines.append(f"- Phase: {summary_report.get('phase', '')}")
    lines.append(f"- Baseline ID: {summary_report.get('baseline_id', '')}")
    lines.append(f"- Tolerance Profile ID: {summary_report.get('tolerance_profile_id', '')}")
    lines.append(f"- Change Class: {summary_report.get('change_class', '')}")
    lines.append(f"- Overall Regression Status: **{summary_report.get('regression.status', 'FAIL')}**")
    lines.append("")
    lines.append("## 1) Suite Status")
    for suite_name, status in sorted(dict(summary_report.get("suite_status", {})).items()):
        lines.append(f"- {suite_name}: {status}")
    lines.append("")
    lines.append("## 2) Blocking Violations")
    violations = summary_report.get("regression.violations", [])
    if isinstance(violations, list) and violations:
        for violation in violations:
            suite = str(violation.get("suite", "unknown"))
            reason = str(violation.get("reason", ""))
            metric = str(violation.get("metric", ""))
            lines.append(f"- [{suite}] {metric}: {reason}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## 3) S8 Crash-Class Coverage Matrix (Local Packet)")
    lines.append("| Path | Coverage | Status | Runtime Lock ID |")
    lines.append("| --- | --- | --- | --- |")
    path_by_id = {
        str(item.get("resume_path_id", "")): item for item in resume_packet.get("paths", []) if isinstance(item, Mapping)
    }
    coverage = resume_packet.get("coverage_matrix", {})
    for path_id in _REQUIRED_S8_PATHS:
        path_entry = path_by_id.get(path_id, {})
        coverage_text = "Yes" if bool(coverage.get(path_id, False)) else "No"
        status_text = str(path_entry.get("status", "FAIL"))
        runtime_lock_id = str(path_entry.get("runtime_lock_manifest_id", ""))
        lines.append(f"| {path_id} | {coverage_text} | {status_text} | {runtime_lock_id} |")

    lines.append("")
    lines.append("## 4) S8 Drift Diagnosis Summary (vs uninterrupted)")
    for path_id in _REQUIRED_S8_PATHS:
        path_entry = path_by_id.get(path_id, {})
        labels = path_entry.get("drift_diagnosis_labels", {})
        runtime_drift = bool(labels.get("runtime_drift", False))
        rng_drift = bool(labels.get("rng_drift", False))
        data_drift = bool(labels.get("data_slice_drift", False))
        optimizer_drift = bool(labels.get("optimizer_state_drift", False))
        l1_value = _safe_float(path_entry.get("failure_credit_l1_distance"), 0.0)
        lines.append(
            f"- {path_id}: runtime_drift={runtime_drift}, rng_drift={rng_drift}, "
            f"data_slice_drift={data_drift}, optimizer_state_drift={optimizer_drift}, "
            f"failure_credit_l1={l1_value}"
        )

    lines.append("")
    lines.append("## 5) H100 Packet Status")
    lines.append(f"- S8 status for H100 packet: **{h100_packet.get('s8_status_for_h100_packet', 'FAIL')}**")
    lines.append(
        "- Coverage: "
        + ", ".join(
            f"{path_id}={bool(dict(h100_packet.get('coverage_matrix', {})).get(path_id, False))}"
            for path_id in _REQUIRED_S8_PATHS
        )
    )
    if h100_packet.get("block_reason"):
        lines.append(f"- Block reason: {h100_packet['block_reason']}")

    lines.append("")
    lines.append("## 6) Notes")
    for note in summary_report.get("notes", []):
        lines.append(f"- {note}")

    lines.append("")
    lines.append("## 7) Completion Checklist")
    checklist = summary_report.get("completion_checklist", {})
    docs = checklist.get("mandatory_docs_consulted", [])
    lines.append("- Mandatory docs consulted: " + ", ".join(f"`{doc}`" for doc in docs))
    lines.append(f"- Change class: `{summary_report.get('change_class', '')}`")
    lines.append(
        "- Expected failure-category impact: "
        + str(checklist.get("expected_failure_category_impact", ""))
    )
    lines.append(
        "- Technical debt guardrails introduced: "
        + str(checklist.get("technical_debt_guardrails_introduced", "none"))
    )
    lines.append("- Termination: `" + str(checklist.get("termination", "Blocked")) + "`")
    return "\n".join(lines) + "\n"


def write_phase_c_gate_artifacts(
    *,
    output_dir: Path,
    summary_report: Mapping[str, Any],
    concept_breakdown: Mapping[str, Any],
    paired_representation_diff: Mapping[str, Any],
    failure_profile_diff: Mapping[str, Any],
    credit_routing_diff: Mapping[str, Any],
    resume_packet: Mapping[str, Any],
    s1_output: Mapping[str, Any],
    s2_output: Mapping[str, Any],
    s2_mounted_output: Mapping[str, Any],
    h100_packet: Mapping[str, Any],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_report.json"
    concept_path = output_dir / "concept_breakdown.json"
    paired_path = output_dir / "paired_representation_diff.json"
    failure_profile_path = output_dir / "failure_profile_diff.json"
    credit_routing_path = output_dir / "credit_routing_diff.json"
    resume_packet_path = output_dir / "resume_consistency_packet.json"
    s1_path = output_dir / "s1_output.json"
    s2_path = output_dir / "s2_output.json"
    s2_mounted_path = output_dir / "s2_mounted_output.json"
    h100_path = output_dir / "h100_packet_summary.json"
    report_path = output_dir / "phase_c_gate_report.md"

    _write_json(summary_path, summary_report)
    _write_json(concept_path, concept_breakdown)
    _write_json(paired_path, paired_representation_diff)
    _write_json(failure_profile_path, failure_profile_diff)
    _write_json(credit_routing_path, credit_routing_diff)

    resume_payload = dict(resume_packet)
    resume_payload["h100_packet"] = dict(h100_packet)
    _write_json(resume_packet_path, resume_payload)
    _write_json(s1_path, s1_output)
    _write_json(s2_path, s2_output)
    _write_json(s2_mounted_path, s2_mounted_output)
    _write_json(h100_path, h100_packet)

    markdown_report = _build_markdown_report(
        summary_report=summary_report,
        resume_packet=resume_packet,
        h100_packet=h100_packet,
    )
    report_path.write_text(markdown_report, encoding="utf-8")

    return {
        "summary_report.json": str(summary_path),
        "concept_breakdown.json": str(concept_path),
        "paired_representation_diff.json": str(paired_path),
        "failure_profile_diff.json": str(failure_profile_path),
        "credit_routing_diff.json": str(credit_routing_path),
        "resume_consistency_packet.json": str(resume_packet_path),
        "s1_output.json": str(s1_path),
        "s2_output.json": str(s2_path),
        "s2_mounted_output.json": str(s2_mounted_path),
        "h100_packet_summary.json": str(h100_path),
        "phase_c_gate_report.md": str(report_path),
    }
