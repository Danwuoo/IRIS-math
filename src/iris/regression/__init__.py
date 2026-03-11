from importlib import import_module

from .phase_c_gate import (
    GateContext,
    Tolerances,
    build_concept_breakdown,
    build_credit_routing_diff,
    build_failure_profile_diff,
    build_h100_packet_summary,
    build_paired_representation_diff,
    build_resume_consistency_packet,
    build_summary_report,
    evaluate_s3_status,
    evaluate_s4_status,
    evaluate_s5_status,
    evaluate_s6_status,
    evaluate_s7_status,
    evaluate_s8_status,
    utc_now_iso,
    write_phase_c_gate_artifacts,
)

_OPTIONAL_EXPORTS = {
    "build_concept_breakdown_v2": ("iris.regression.phase_d_gate", "build_concept_breakdown_v2"),
    "build_failure_profile_diff_v2": ("iris.regression.phase_d_gate", "build_failure_profile_diff_v2"),
    "build_paired_representation_diff_v2": (
        "iris.regression.phase_d_gate",
        "build_paired_representation_diff_v2",
    ),
    "evaluate_s3_status_v2": ("iris.regression.phase_d_gate", "evaluate_s3_status_v2"),
    "evaluate_s4_status_v2": ("iris.regression.phase_d_gate", "evaluate_s4_status_v2"),
    "evaluate_s5_status_v2": ("iris.regression.phase_d_gate", "evaluate_s5_status_v2"),
    "run_phase_d_gate": ("iris.regression.phase_d_gate", "run_phase_d_gate"),
    "run_phase_e_gate": ("iris.regression.phase_e_gate", "run_phase_e_gate"),
}

__all__ = [
    "GateContext",
    "Tolerances",
    "build_concept_breakdown",
    "build_credit_routing_diff",
    "build_failure_profile_diff",
    "build_h100_packet_summary",
    "build_paired_representation_diff",
    "build_resume_consistency_packet",
    "build_summary_report",
    "evaluate_s3_status",
    "evaluate_s4_status",
    "evaluate_s5_status",
    "evaluate_s6_status",
    "evaluate_s7_status",
    "evaluate_s8_status",
    "utc_now_iso",
    "write_phase_c_gate_artifacts",
    "build_concept_breakdown_v2",
    "build_failure_profile_diff_v2",
    "build_paired_representation_diff_v2",
    "evaluate_s3_status_v2",
    "evaluate_s4_status_v2",
    "evaluate_s5_status_v2",
    "run_phase_d_gate",
    "run_phase_e_gate",
]


def __getattr__(name: str):
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module 'iris.regression' has no attribute {name!r}")

    module_name, attr_name = _OPTIONAL_EXPORTS[name]
    try:
        module = import_module(module_name)
    except Exception as error:  # pragma: no cover - optional dependency failure
        raise RuntimeError(
            "Phase D/E regression exports require the optional JAX-backed ARC compatibility stack."
        ) from error
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
