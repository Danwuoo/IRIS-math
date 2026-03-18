from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_QA_PROFILES: Dict[str, Dict[str, float]] = {
    "default": {
        "max_control_char_ratio": 0.02,
        "max_repetition_rate": 0.20,
        "min_avg_line_length": 20.0,
        "max_line_break_density": 0.08,
        "min_language_signal_ratio": 0.10,
    },
    "document_relaxed": {
        "max_control_char_ratio": 0.02,
        "max_repetition_rate": 0.35,
        "min_avg_line_length": 8.0,
        "max_line_break_density": 0.18,
        "min_language_signal_ratio": 0.08,
    },
}


def _control_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    control_count = len(_CONTROL_PATTERN.findall(text))
    return float(control_count) / float(len(text))


def _repetition_rate(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    counts: Dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    repeated = sum(max(0, count - 1) for count in counts.values())
    return float(repeated) / float(len(lines))


def _fragmentation_metrics(text: str) -> Dict[str, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    avg_line_length = (
        sum(len(line) for line in lines) / float(len(lines))
        if lines
        else 0.0
    )
    line_break_density = float(text.count("\n")) / float(max(len(text), 1))
    return {
        "avg_line_length": float(avg_line_length),
        "line_break_density": float(line_break_density),
    }


def _language_signal_ratio(text: str) -> float:
    if not text:
        return 0.0
    alphabetic = sum(1 for char in text if char.isalpha())
    printable = sum(1 for char in text if char.isprintable())
    if printable <= 0:
        return 0.0
    return float(alphabetic) / float(printable)


def _resolve_profile(profile: str | None) -> Dict[str, float]:
    normalized = str(profile or "default").strip().lower() or "default"
    return dict(_QA_PROFILES.get(normalized, _QA_PROFILES["default"]))


def evaluate_text_quality(text: str, *, profile: str | None = None) -> Dict[str, Any]:
    thresholds = _resolve_profile(profile)
    metrics: Dict[str, Any] = {}
    reasons: List[str] = []

    control_ratio = _control_char_ratio(text)
    metrics["control_char_ratio"] = control_ratio
    if control_ratio > float(thresholds["max_control_char_ratio"]):
        reasons.append(
            f"control/non-printable ratio {control_ratio:.4f} exceeds {thresholds['max_control_char_ratio']:.2f}"
        )

    repetition = _repetition_rate(text)
    metrics["repetition_rate"] = repetition
    if repetition > float(thresholds["max_repetition_rate"]):
        reasons.append(
            f"repetition rate {repetition:.4f} exceeds {thresholds['max_repetition_rate']:.2f}"
        )

    frag = _fragmentation_metrics(text)
    metrics.update(frag)
    if (
        frag["avg_line_length"] < float(thresholds["min_avg_line_length"])
        and frag["line_break_density"] > float(thresholds["max_line_break_density"])
    ):
        reasons.append(
            "severe fragmentation detected (short lines with high line-break density)"
        )

    language_signal = _language_signal_ratio(text)
    metrics["language_signal_ratio"] = language_signal
    if language_signal < float(thresholds["min_language_signal_ratio"]):
        reasons.append(
            f"language signal ratio {language_signal:.4f} below {thresholds['min_language_signal_ratio']:.2f}"
        )

    return {
        "pass": not reasons,
        "profile": str(profile or "default"),
        "metrics": metrics,
        "reasons": reasons,
    }


def enforce_qa_gate(text: str, *, profile: str | None = None) -> Dict[str, Any]:
    report = evaluate_text_quality(text, profile=profile)
    if not report["pass"]:
        raise ValueError("Data QA gate failed: " + "; ".join(report["reasons"]))
    return report


def merge_qa_reports(reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"source_reports": {}, "failed_sources": []}
    for source_id, payload in reports.items():
        source_report = dict(payload)
        merged["source_reports"][source_id] = source_report
        if not bool(source_report.get("pass", False)):
            merged["failed_sources"].append(str(source_id))
    merged["pass"] = len(merged["failed_sources"]) == 0
    return merged
