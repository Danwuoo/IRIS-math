from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from ..arc import (
    ArcDiagnosticRunner,
    ArcEvalConfig,
    ArcExample,
    ArcInferenceRecord,
    ArcTask,
    build_rearc_pairs,
    group_tasks_by_concept,
    load_conceptarc_tasks,
    load_rearc_tasks,
    normalize_failure_histogram,
)
from .phase_c_gate import (
    GateContext,
    Tolerances,
    _is_phase_c_or_later,
    _safe_float,
    build_credit_routing_diff,
    build_h100_packet_summary,
    build_resume_consistency_packet,
    evaluate_s6_status,
    evaluate_s7_status,
    evaluate_s8_status,
    utc_now_iso,
)

_FAILURE_CODES: Tuple[str, ...] = (
    "F_REP",
    "F_PROC",
    "F_SEARCH",
    "F_MEM",
    "F_ABS",
    "F_EVAL",
)

_TECH_DEBT_NOTE = (
    "TEMPORARY TECHNICAL DEBT: max_reasoning_cycles hard cap. "
    "Removal criterion: remove after 3 consecutive full-runs "
    "show stable termination calibration."
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return dict(payload)
    return {"data": payload}


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(dict(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_failure_histogram(records: Iterable[ArcInferenceRecord], *, failures_only: bool) -> Dict[str, float]:
    histogram = {code: 0.0 for code in _FAILURE_CODES}
    for record in records:
        if failures_only and record.success:
            continue
        for code in _FAILURE_CODES:
            histogram[code] += float(record.failure_code_distribution.get(code, 0.0))
    return normalize_failure_histogram(histogram)


def _merge_failure_histograms(histograms: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    merged = {code: 0.0 for code in _FAILURE_CODES}
    for histogram in histograms:
        for code in _FAILURE_CODES:
            merged[code] += float(histogram.get(code, 0.0))
    return normalize_failure_histogram(merged)


def _per_code_delta(
    current_histogram: Mapping[str, float],
    baseline_histogram: Mapping[str, float],
) -> Dict[str, float]:
    return {
        code: float(current_histogram.get(code, 0.0)) - float(baseline_histogram.get(code, 0.0))
        for code in _FAILURE_CODES
    }


def _max_abs_delta(delta: Mapping[str, float]) -> float:
    return max((abs(float(value)) for value in delta.values()), default=0.0)


def _l1_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    return float(sum(abs(float(left.get(code, 0.0)) - float(right.get(code, 0.0))) for code in _FAILURE_CODES))


def _kl_divergence(
    current_histogram: Mapping[str, float],
    baseline_histogram: Mapping[str, float],
    epsilon: float = 1e-12,
) -> float:
    current = [max(float(current_histogram.get(code, 0.0)), epsilon) for code in _FAILURE_CODES]
    baseline = [max(float(baseline_histogram.get(code, 0.0)), epsilon) for code in _FAILURE_CODES]
    current_total = float(sum(current))
    baseline_total = float(sum(baseline))
    if current_total <= 0.0 or baseline_total <= 0.0:
        return float("inf")
    p = [value / current_total for value in current]
    q = [value / baseline_total for value in baseline]
    return float(sum(p_i * math.log(p_i / q_i) for p_i, q_i in zip(p, q)))


def _prepare_leakage_train_overrides(
    tasks: Sequence[ArcTask],
    *,
    seed: int,
) -> Dict[str, Tuple[ArcExample, ...]]:
    # Keep deterministic cross-concept mixing to make full-run regression hashes stable.
    rng = __import__("random").Random(seed)
    grouped = group_tasks_by_concept(tasks)
    by_task_id = {task.task_id: task for task in tasks}
    overrides: Dict[str, Tuple[ArcExample, ...]] = {}

    concept_ids = sorted(grouped.keys())
    for concept_id in concept_ids:
        candidate_examples: List[ArcExample] = []
        for other_concept in concept_ids:
            if other_concept == concept_id:
                continue
            for task in grouped[other_concept]:
                candidate_examples.extend(task.train_examples)
        for task in grouped[concept_id]:
            target_count = max(1, len(task.train_examples))
            if not candidate_examples:
                overrides[task.task_id] = tuple(task.train_examples)
                continue
            picked: List[ArcExample] = []
            for _ in range(target_count):
                picked.append(candidate_examples[rng.randrange(0, len(candidate_examples))])
            overrides[task.task_id] = tuple(picked)

    for task_id, task in by_task_id.items():
        overrides.setdefault(task_id, tuple(task.train_examples))
    return overrides


def _dominant_code(histogram: Mapping[str, float]) -> str:
    normalized = normalize_failure_histogram(histogram)
    return max(_FAILURE_CODES, key=lambda code: float(normalized.get(code, 0.0)))


def build_concept_breakdown_v2(
    *,
    concept_tasks: Sequence[ArcTask],
    isolation_records: Sequence[ArcInferenceRecord],
    leakage_records: Sequence[ArcInferenceRecord],
    context: GateContext,
    pairing_policy: str = "adjacent",
) -> Dict[str, Any]:
    isolation_by_concept: Dict[str, List[ArcInferenceRecord]] = {}
    leakage_by_concept: Dict[str, List[ArcInferenceRecord]] = {}
    for record in isolation_records:
        concept_id = str(record.concept_id or "UNKNOWN")
        isolation_by_concept.setdefault(concept_id, []).append(record)
    for record in leakage_records:
        concept_id = str(record.concept_id or "UNKNOWN")
        leakage_by_concept.setdefault(concept_id, []).append(record)

    concept_ids = sorted(
        {
            *(str(task.concept_id or "UNKNOWN") for task in concept_tasks),
            *isolation_by_concept.keys(),
            *leakage_by_concept.keys(),
        }
    )
    concepts: List[Dict[str, Any]] = []
    for concept_id in concept_ids:
        iso_records = isolation_by_concept.get(concept_id, [])
        leak_records = leakage_by_concept.get(concept_id, [])
        if not iso_records:
            continue
        iso_success_rate = float(sum(1.0 for record in iso_records if record.success) / float(len(iso_records)))
        leak_success_rate = (
            float(sum(1.0 for record in leak_records if record.success) / float(len(leak_records)))
            if leak_records
            else iso_success_rate
        )
        isolation_score = iso_success_rate
        leakage_score = max(0.0, isolation_score - leak_success_rate)
        combined_hist = _merge_failure_histograms(
            [
                _stable_failure_histogram(iso_records, failures_only=True),
                _stable_failure_histogram(leak_records, failures_only=True),
            ]
        )
        concepts.append(
            {
                "concept_id": concept_id,
                "sample_count": len(iso_records),
                "leakage_sample_count": len(leak_records),
                "concept.success_rate": iso_success_rate,
                "concept.isolation_score": isolation_score,
                "concept.leakage_score": leakage_score,
                "concept.failure_profile": combined_hist,
                "concept.failure_profile_dominant": _dominant_code(combined_hist),
            }
        )

    concept_count = len(concepts)
    aggregate_hist = _stable_failure_histogram(
        list(isolation_records) + list(leakage_records),
        failures_only=True,
    )
    aggregate = {
        "concept.success_rate.mean": (
            sum(float(item["concept.success_rate"]) for item in concepts) / float(concept_count)
            if concept_count
            else 0.0
        ),
        "concept.isolation_score.mean": (
            sum(float(item["concept.isolation_score"]) for item in concepts) / float(concept_count)
            if concept_count
            else 0.0
        ),
        "concept.leakage_score.mean": (
            sum(float(item["concept.leakage_score"]) for item in concepts) / float(concept_count)
            if concept_count
            else 0.0
        ),
        "failure_profile": aggregate_hist,
        "failure_profile_dominant": _dominant_code(aggregate_hist),
    }
    status = "PASS" if concept_count > 0 else "FAIL"
    report: Dict[str, Any] = {
        "schema": "iris.regression.concept_breakdown/v2",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "pairing_policy": pairing_policy,
        "status": status,
        "concept_count": concept_count,
        "aggregate": aggregate,
        "concepts": concepts,
    }
    if status != "PASS":
        report["reason"] = "No concept diagnostic records were produced."
    report["artifact_hash"] = _hash_payload(report)
    return report


def build_paired_representation_diff_v2(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    context: GateContext,
    pairing_policy: str = "adjacent",
) -> Dict[str, Any]:
    pair_count = len(pair_rows)
    asymmetry_rate = (
        sum(1.0 for row in pair_rows if bool(row.get("pair.asymmetry", False))) / float(pair_count)
        if pair_count
        else 1.0
    )
    invariance_gap = (
        sum(float(row.get("pair.invariance_gap", 0.0)) for row in pair_rows) / float(pair_count)
        if pair_count
        else 1.0
    )
    report: Dict[str, Any] = {
        "schema": "iris.regression.paired_representation_diff/v2",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "pairing_policy": pairing_policy,
        "status": "PASS" if pair_count > 0 else "FAIL",
        "pair_count": pair_count,
        "paired.asymmetry_rate": float(asymmetry_rate),
        "paired.invariance.gap": float(invariance_gap),
        "pairs": [dict(row) for row in pair_rows],
    }
    if pair_count <= 0:
        report["reason"] = "No paired diagnostic records were produced."
    report["artifact_hash"] = _hash_payload(report)
    return report


def build_failure_profile_diff_v2(
    *,
    current_histogram: Mapping[str, float],
    baseline_histogram: Mapping[str, float] | None,
    context: GateContext,
) -> Dict[str, Any]:
    current = normalize_failure_histogram(current_histogram)
    if not isinstance(baseline_histogram, Mapping):
        report: Dict[str, Any] = {
            "schema": "iris.regression.failure_profile_diff/v2",
            "phase": context.phase,
            "baseline_id": context.baseline_id,
            "tolerance_profile_id": context.tolerance_profile_id,
            "status": "FAIL",
            "reason": "baseline failure histogram is missing",
            "failure_profile.current_histogram": current,
            "failure_profile.baseline_histogram": {},
            "failure_profile.delta": {code: current.get(code, 0.0) for code in _FAILURE_CODES},
            "failure_profile.max_abs_delta": 1.0,
            "failure_profile.l1_distance": 1.0,
            "failure_profile.kl_divergence": float("inf"),
        }
        report["artifact_hash"] = _hash_payload(report)
        return report

    baseline = normalize_failure_histogram(baseline_histogram)
    delta = _per_code_delta(current, baseline)
    report = {
        "schema": "iris.regression.failure_profile_diff/v2",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "status": "PASS",
        "failure_profile.current_histogram": current,
        "failure_profile.baseline_histogram": baseline,
        "failure_profile.delta": delta,
        "failure_profile.max_abs_delta": _max_abs_delta(delta),
        "failure_profile.l1_distance": _l1_distance(current, baseline),
        "failure_profile.kl_divergence": _kl_divergence(current, baseline),
    }
    report["artifact_hash"] = _hash_payload(report)
    return report


def evaluate_s3_status_v2(
    *,
    failure_profile_diff: Mapping[str, Any],
    tolerances: Tolerances,
    phase: str,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if _is_phase_c_or_later(phase) and str(failure_profile_diff.get("status")) != "PASS":
        reasons.append(str(failure_profile_diff.get("reason", "failure_profile_diff artifact is unavailable")))
        return "FAIL", reasons
    l1_distance = _safe_float(failure_profile_diff.get("failure_profile.l1_distance"), float("inf"))
    kl_value = _safe_float(failure_profile_diff.get("failure_profile.kl_divergence"), float("inf"))
    if not math.isfinite(l1_distance):
        reasons.append("failure_profile.l1_distance is non-finite")
    elif l1_distance > tolerances.failure_profile_kl_epsilon:
        reasons.append(
            f"failure taxonomy L1 drift {l1_distance:.6g} exceeds tolerance "
            f"{tolerances.failure_profile_kl_epsilon:.6g}"
        )
    if not math.isfinite(kl_value):
        reasons.append("failure_profile.kl_divergence is non-finite")
    elif kl_value > tolerances.failure_profile_kl_epsilon:
        reasons.append(
            f"failure taxonomy KL drift {kl_value:.6g} exceeds tolerance "
            f"{tolerances.failure_profile_kl_epsilon:.6g}"
        )
    return ("PASS" if not reasons else "FAIL", reasons)


def evaluate_s4_status_v2(
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
        return "FAIL", reasons, details
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
    for concept_entry in current_concepts:
        concept_id = str(concept_entry.get("concept_id", ""))
        baseline_entry = baseline_by_id.get(concept_id)
        if baseline_entry is None:
            details.append(
                {
                    "metric": "concept.baseline_presence",
                    "concept_id": concept_id,
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


def evaluate_s5_status_v2(
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
    baseline_pair_count = int(_safe_float((baseline_paired_diff or {}).get("pair_count"), 0.0))
    if current_pair_count <= 0:
        reasons.append("current paired_representation_diff has no pairs")
        return "FAIL", reasons, details
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


def _build_summary_report_v2(
    *,
    context: GateContext,
    suite_status: Mapping[str, str],
    violations: Sequence[Mapping[str, Any]],
    notes: Sequence[str],
    evidence_paths: Sequence[str],
    max_reasoning_cycles: int,
) -> Dict[str, Any]:
    all_suites_pass = all(str(value).upper() == "PASS" for value in suite_status.values())
    normalized_violations = [dict(item) for item in violations]
    overall_status = "PASS" if all_suites_pass and not normalized_violations else "FAIL"
    termination = "Done" if overall_status == "PASS" else "Blocked"
    return {
        "schema": "iris.regression.summary_report/v1",
        "phase": context.phase,
        "baseline_id": context.baseline_id,
        "tolerance_profile_id": context.tolerance_profile_id,
        "change_class": context.change_class,
        "generated_at_utc": utc_now_iso(),
        "suite_status": dict(suite_status),
        "regression.status": overall_status,
        "regression.violations": normalized_violations,
        "notes": list(notes),
        "completion_checklist": {
            "mandatory_docs_consulted": list(context.mandatory_docs_consulted),
            "expected_failure_category_impact": "F_REP, F_PROC, F_SEARCH, F_EVAL visibility+attribution uplift.",
            "technical_debt_guardrails_introduced": (
                f"{_TECH_DEBT_NOTE} Current max_reasoning_cycles={int(max_reasoning_cycles)}."
            ),
            "regression_evidence_paths": list(evidence_paths),
            "termination": termination,
        },
    }


def _build_markdown_report_v2(
    *,
    summary_report: Mapping[str, Any],
    notes: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("# Phase D Gate Report")
    lines.append("")
    lines.append("- Document Type: Design Note (Non-normative)")
    lines.append(f"- Generated At (UTC): {summary_report.get('generated_at_utc', '')}")
    lines.append(f"- Phase: {summary_report.get('phase', '')}")
    lines.append(f"- Baseline ID: {summary_report.get('baseline_id', '')}")
    lines.append(f"- Tolerance Profile ID: {summary_report.get('tolerance_profile_id', '')}")
    lines.append(f"- Change Class: {summary_report.get('change_class', '')}")
    lines.append(f"- Regression Status: **{summary_report.get('regression.status', 'FAIL')}**")
    lines.append("")
    lines.append("## 1) Suite Status")
    for suite_name, status in sorted(dict(summary_report.get("suite_status", {})).items()):
        lines.append(f"- {suite_name}: {status}")
    lines.append("")
    lines.append("## 2) Violations")
    violations = summary_report.get("regression.violations", [])
    if isinstance(violations, list) and violations:
        for row in violations:
            lines.append(
                f"- [{row.get('suite', 'unknown')}] {row.get('metric', '')}: {row.get('reason', '')}"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## 3) Notes")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    checklist = dict(summary_report.get("completion_checklist", {}))
    lines.append("## 4) Completion Checklist")
    lines.append(
        "- Mandatory docs consulted: "
        + ", ".join(f"`{doc}`" for doc in checklist.get("mandatory_docs_consulted", []))
    )
    lines.append(f"- Change class: `{summary_report.get('change_class', '')}`")
    lines.append(
        "- Expected failure-category impact: "
        + str(checklist.get("expected_failure_category_impact", ""))
    )
    lines.append(
        "- Technical debt guardrails introduced: "
        + str(checklist.get("technical_debt_guardrails_introduced", "none"))
    )
    lines.append(f"- Termination: `{checklist.get('termination', 'Blocked')}`")
    return "\n".join(lines) + "\n"


def _load_baseline_artifacts(baseline_report_dir: Path | None) -> Dict[str, Dict[str, Any] | None]:
    baseline: Dict[str, Dict[str, Any] | None] = {
        "concept_breakdown": None,
        "paired_representation_diff": None,
        "failure_profile_diff": None,
    }
    if baseline_report_dir is None:
        return baseline
    base_dir = Path(baseline_report_dir)
    concept_path = base_dir / "concept_breakdown.json"
    paired_path = base_dir / "paired_representation_diff.json"
    failure_path = base_dir / "failure_profile_diff.json"
    if concept_path.exists():
        baseline["concept_breakdown"] = _read_json(concept_path)
    if paired_path.exists():
        baseline["paired_representation_diff"] = _read_json(paired_path)
    if failure_path.exists():
        baseline["failure_profile_diff"] = _read_json(failure_path)
    return baseline


def _collect_pair_rows(
    *,
    runner: ArcDiagnosticRunner,
    tasks_by_id: Mapping[str, ArcTask],
    pairing_policy: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pairs = build_rearc_pairs(tasks_by_id.values(), pairing_policy=pairing_policy)
    for pair in pairs:
        source_task = tasks_by_id[pair.task_id]
        left_task = ArcTask(
            task_id=f"{pair.task_id}/pair{pair.pair_index}/left",
            train_examples=tuple(source_task.train_examples),
            test_examples=(pair.left,),
            source_path=pair.source_path,
            source_name=source_task.source_name,
            concept_id=source_task.concept_id,
        )
        right_task = ArcTask(
            task_id=f"{pair.task_id}/pair{pair.pair_index}/right",
            train_examples=tuple(source_task.train_examples),
            test_examples=(pair.right,),
            source_path=pair.source_path,
            source_name=source_task.source_name,
            concept_id=source_task.concept_id,
        )
        left_record = runner.evaluate_case(task=left_task, case_index=0, mode="paired.left")
        right_record = runner.evaluate_case(task=right_task, case_index=0, mode="paired.right")
        rows.append(
            {
                "task_id": pair.task_id,
                "pair_index": int(pair.pair_index),
                "left.success": bool(left_record.success),
                "right.success": bool(right_record.success),
                "left.validity_score": float(left_record.validity_score),
                "right.validity_score": float(right_record.validity_score),
                "left.failure_code": str(left_record.failure_code),
                "right.failure_code": str(right_record.failure_code),
                "pair.invariance_gap": abs(
                    float(left_record.validity_score) - float(right_record.validity_score)
                ),
                "pair.asymmetry": bool(left_record.success != right_record.success),
            }
        )
    return rows


def run_phase_d_gate(
    *,
    context: GateContext,
    tolerances: Tolerances,
    output_dir: Path,
    model_run_dir: Path,
    conceptarc_corpus: Path,
    rearc_tasks: Path,
    baseline_report_dir: Path | None,
    phase_root: Path,
    h100_path_map: Mapping[str, Path],
    s1_status: str,
    s1_reasons: Sequence[str],
    s2_status: str,
    s2_reasons: Sequence[str],
    s1_output: Mapping[str, Any],
    s2_output: Mapping[str, Any],
    s2_mounted_output: Mapping[str, Any],
    max_reasoning_cycles: int,
    termination_threshold: float,
    seed: int,
    pairing_policy: str = "adjacent",
    freeze_baseline: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline_artifacts(baseline_report_dir)

    concept_tasks = load_conceptarc_tasks(conceptarc_corpus)
    rearc_task_list = load_rearc_tasks(rearc_tasks)
    rearc_tasks_by_id = {task.task_id: task for task in rearc_task_list}

    arc_config = ArcEvalConfig(
        model_run_dir=Path(model_run_dir),
        max_reasoning_cycles=max_reasoning_cycles,
        termination_threshold=termination_threshold,
        seed=seed,
    )
    runner = ArcDiagnosticRunner(config=arc_config)

    leakage_overrides = _prepare_leakage_train_overrides(concept_tasks, seed=seed)
    isolation_records: List[ArcInferenceRecord] = []
    leakage_records: List[ArcInferenceRecord] = []
    for task in concept_tasks:
        for case_index in range(len(task.test_examples)):
            isolation_records.append(
                runner.evaluate_case(
                    task=task,
                    case_index=case_index,
                    mode="concept.isolation",
                    train_examples_override=tuple(task.train_examples),
                )
            )
            leakage_records.append(
                runner.evaluate_case(
                    task=task,
                    case_index=case_index,
                    mode="concept.leakage",
                    train_examples_override=leakage_overrides.get(task.task_id),
                )
            )

    concept_breakdown = build_concept_breakdown_v2(
        concept_tasks=concept_tasks,
        isolation_records=isolation_records,
        leakage_records=leakage_records,
        context=context,
        pairing_policy=pairing_policy,
    )
    pair_rows = _collect_pair_rows(
        runner=runner,
        tasks_by_id=rearc_tasks_by_id,
        pairing_policy=pairing_policy,
    )
    paired_diff = build_paired_representation_diff_v2(
        pair_rows=pair_rows,
        context=context,
        pairing_policy=pairing_policy,
    )
    current_failure_hist = _merge_failure_histograms(
        [
            _stable_failure_histogram(isolation_records, failures_only=True),
            _stable_failure_histogram(leakage_records, failures_only=True),
        ]
    )
    baseline_initialized = False
    if freeze_baseline and baseline_report_dir is not None:
        if baseline.get("concept_breakdown") is None:
            baseline["concept_breakdown"] = concept_breakdown
            baseline_initialized = True
        if baseline.get("paired_representation_diff") is None:
            baseline["paired_representation_diff"] = paired_diff
            baseline_initialized = True

    baseline_failure_hist = None
    baseline_failure_payload = baseline.get("failure_profile_diff")
    if isinstance(baseline_failure_payload, Mapping):
        baseline_failure_hist = baseline_failure_payload.get("failure_profile.current_histogram")
        if not isinstance(baseline_failure_hist, Mapping):
            baseline_failure_hist = baseline_failure_payload.get("failure_profile.baseline_histogram")
    if freeze_baseline and baseline_report_dir is not None and not isinstance(baseline_failure_hist, Mapping):
        baseline_failure_hist = current_failure_hist
        baseline_initialized = True

    failure_profile_diff = build_failure_profile_diff_v2(
        current_histogram=current_failure_hist,
        baseline_histogram=baseline_failure_hist if isinstance(baseline_failure_hist, Mapping) else None,
        context=context,
    )

    local_s8_paths = {
        "uninterrupted": Path(phase_root) / "s8_uninterrupted",
        "execute_crash": Path(phase_root) / "s8_execute",
        "pre_commit_crash": Path(phase_root) / "s8_pre_commit",
        "post_commit_crash": Path(phase_root) / "s8_post_commit",
    }
    resume_packet = build_resume_consistency_packet(local_s8_paths, context)
    credit_routing_diff = build_credit_routing_diff(resume_packet, context)
    h100_packet = build_h100_packet_summary(h100_path_map, context, tolerances=tolerances)

    s3_status, s3_reasons = evaluate_s3_status_v2(
        failure_profile_diff=failure_profile_diff,
        tolerances=tolerances,
        phase=context.phase,
    )
    s4_status, s4_reasons, s4_details = evaluate_s4_status_v2(
        current_concept_breakdown=concept_breakdown,
        baseline_concept_breakdown=baseline.get("concept_breakdown"),
        tolerances=tolerances,
        phase=context.phase,
    )
    s5_status, s5_reasons, s5_details = evaluate_s5_status_v2(
        current_paired_diff=paired_diff,
        baseline_paired_diff=baseline.get("paired_representation_diff"),
        tolerances=tolerances,
        phase=context.phase,
    )
    s6_status, s6_reasons = evaluate_s6_status(resume_packet, credit_routing_diff, tolerances)
    s7_status, s7_reasons = evaluate_s7_status(resume_packet, tolerances)
    s8_status, s8_reasons = evaluate_s8_status(resume_packet, tolerances)

    suite_status = {
        "S1": str(s1_status).upper(),
        "S2": str(s2_status).upper(),
        "S3": s3_status,
        "S4": s4_status,
        "S5": s5_status,
        "S6": s6_status,
        "S7": s7_status,
        "S8": s8_status,
        "S8_h100_packet": str(h100_packet.get("s8_status_for_h100_packet", "FAIL")),
    }
    violations: List[Dict[str, Any]] = []
    if suite_status["S1"] != "PASS":
        violations.append(
            {
                "suite": "S1",
                "metric": "smoke runtime checks",
                "phase": context.phase,
                "suspected_level": "L0-L6",
                "reason": "; ".join(dict.fromkeys(str(item) for item in s1_reasons if str(item))),
            }
        )
    if suite_status["S2"] != "PASS":
        violations.append(
            {
                "suite": "S2",
                "metric": "structural contract checks",
                "phase": context.phase,
                "suspected_level": "State IR + L0-L6 interface layer",
                "reason": "; ".join(dict.fromkeys(str(item) for item in s2_reasons if str(item))),
            }
        )
    if s3_status != "PASS":
        violations.append(
            {
                "suite": "S3",
                "metric": "failure taxonomy histogram drift",
                "phase": context.phase,
                "suspected_level": "L6 routing / regression harness",
                "reason": "; ".join(s3_reasons),
            }
        )
    if s4_status != "PASS":
        violations.append(
            {
                "suite": "S4",
                "metric": "concept.success_rate / concept.isolation_score / concept.leakage_score",
                "phase": context.phase,
                "suspected_level": "L5",
                "reason": "; ".join(dict.fromkeys(s4_reasons)),
                "details": s4_details,
            }
        )
    if s5_status != "PASS":
        violations.append(
            {
                "suite": "S5",
                "metric": "paired.asymmetry_rate / paired.invariance.gap",
                "phase": context.phase,
                "suspected_level": "L0/L1/L2",
                "reason": "; ".join(dict.fromkeys(s5_reasons)),
                "details": s5_details,
            }
        )
    if s6_status != "PASS":
        violations.append(
            {
                "suite": "S6",
                "metric": "failure.credit stability",
                "phase": context.phase,
                "suspected_level": "L6",
                "reason": "; ".join(s6_reasons),
            }
        )
    if s7_status != "PASS":
        violations.append(
            {
                "suite": "S7",
                "metric": "pretraining diagnostics deltas",
                "phase": context.phase,
                "suspected_level": "L3/L6",
                "reason": "; ".join(s7_reasons),
            }
        )
    if s8_status != "PASS":
        violations.append(
            {
                "suite": "S8",
                "metric": "resume consistency drift / coverage",
                "phase": context.phase,
                "suspected_level": "L3/L6 + training transaction path",
                "reason": "; ".join(s8_reasons),
            }
        )
    if suite_status["S8_h100_packet"] != "PASS":
        violations.append(
            {
                "suite": "S8",
                "metric": "crash-class coverage (H100 packet)",
                "phase": context.phase,
                "suspected_level": "L3/L6 + training transaction path",
                "reason": str(h100_packet.get("block_reason", "H100 packet coverage incomplete.")),
            }
        )

    evidence_paths = [
        str(output_dir / "summary_report.json"),
        str(output_dir / "concept_breakdown.json"),
        str(output_dir / "paired_representation_diff.json"),
        str(output_dir / "failure_profile_diff.json"),
        str(output_dir / "credit_routing_diff.json"),
        str(output_dir / "resume_consistency_packet.json"),
        str(output_dir / "h100_packet_summary.json"),
        str(output_dir / "s1_output.json"),
        str(output_dir / "s2_output.json"),
        str(output_dir / "s2_mounted_output.json"),
    ]
    notes = [
        f"pairing_policy={pairing_policy}",
        f"max_reasoning_cycles={int(max_reasoning_cycles)}",
        f"termination_threshold={float(termination_threshold):.4f}",
        f"seed={int(seed)}",
        f"S8 local packet drift_clear={resume_packet.get('all_drift_labels_clear')}",
        f"S8 h100 packet status={h100_packet.get('s8_status_for_h100_packet')}",
        _TECH_DEBT_NOTE,
    ]
    if baseline_initialized:
        notes.append("phase-d-v1 baseline initialized from current run artifacts.")
    summary_report = _build_summary_report_v2(
        context=context,
        suite_status=suite_status,
        violations=violations,
        notes=notes,
        evidence_paths=evidence_paths,
        max_reasoning_cycles=max_reasoning_cycles,
    )

    _write_json(output_dir / "summary_report.json", summary_report)
    _write_json(output_dir / "concept_breakdown.json", concept_breakdown)
    _write_json(output_dir / "paired_representation_diff.json", paired_diff)
    _write_json(output_dir / "failure_profile_diff.json", failure_profile_diff)
    _write_json(output_dir / "credit_routing_diff.json", credit_routing_diff)
    _write_json(output_dir / "resume_consistency_packet.json", resume_packet)
    _write_json(output_dir / "h100_packet_summary.json", h100_packet)
    _write_json(output_dir / "s1_output.json", dict(s1_output))
    _write_json(output_dir / "s2_output.json", dict(s2_output))
    _write_json(output_dir / "s2_mounted_output.json", dict(s2_mounted_output))

    report_text = _build_markdown_report_v2(summary_report=summary_report, notes=notes)
    (output_dir / "phase_d_gate_report.md").write_text(report_text, encoding="utf-8")

    if freeze_baseline and baseline_report_dir is not None:
        baseline_dir = Path(baseline_report_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        _write_json(baseline_dir / "concept_breakdown.json", concept_breakdown)
        _write_json(baseline_dir / "paired_representation_diff.json", paired_diff)
        _write_json(baseline_dir / "failure_profile_diff.json", failure_profile_diff)

    return {
        "summary_report": summary_report,
        "suite_status": suite_status,
        "artifact_paths": {
            "summary_report.json": str(output_dir / "summary_report.json"),
            "concept_breakdown.json": str(output_dir / "concept_breakdown.json"),
            "paired_representation_diff.json": str(output_dir / "paired_representation_diff.json"),
            "failure_profile_diff.json": str(output_dir / "failure_profile_diff.json"),
            "credit_routing_diff.json": str(output_dir / "credit_routing_diff.json"),
            "resume_consistency_packet.json": str(output_dir / "resume_consistency_packet.json"),
            "h100_packet_summary.json": str(output_dir / "h100_packet_summary.json"),
            "phase_d_gate_report.md": str(output_dir / "phase_d_gate_report.md"),
        },
    }
