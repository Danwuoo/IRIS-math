from importlib import import_module

from .encoding import ArcEncodingConfig, encode_arc_case_to_state, infer_target_shape
from .loaders import group_tasks_by_concept, load_conceptarc_tasks, load_rearc_tasks
from .pairing import build_rearc_pairs
from .types import (
    FAILURE_CODES,
    ArcExample,
    ArcInferenceRecord,
    ArcPair,
    ArcTask,
    dominant_failure_code,
    failure_credit_to_code_distribution,
    normalize_failure_histogram,
)

_LAZY_EXPORTS = {
    "ArcDiagnosticRunner": ("iris.arc.inference", "ArcDiagnosticRunner"),
    "ArcEvalConfig": ("iris.arc.inference", "ArcEvalConfig"),
    "aggregate_failure_histogram": ("iris.arc.inference", "aggregate_failure_histogram"),
    "run_arc_diagnostic_eval": ("iris.arc.inference", "run_arc_diagnostic_eval"),
    "export_benchmark_submission": ("iris.arc.benchmark_bridge", "export_benchmark_submission"),
    "load_benchmark_tasks": ("iris.arc.benchmark_bridge", "load_benchmark_tasks"),
}

__all__ = [
    "ArcDiagnosticRunner",
    "ArcEncodingConfig",
    "ArcEvalConfig",
    "ArcExample",
    "ArcInferenceRecord",
    "ArcPair",
    "ArcTask",
    "FAILURE_CODES",
    "aggregate_failure_histogram",
    "export_benchmark_submission",
    "build_rearc_pairs",
    "dominant_failure_code",
    "encode_arc_case_to_state",
    "failure_credit_to_code_distribution",
    "group_tasks_by_concept",
    "infer_target_shape",
    "load_conceptarc_tasks",
    "load_benchmark_tasks",
    "load_rearc_tasks",
    "normalize_failure_histogram",
    "run_arc_diagnostic_eval",
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'iris.arc' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        module = import_module(module_name)
    except Exception as error:  # pragma: no cover - optional dependency failure
        raise RuntimeError(
            "ARC inference and benchmark bridge exports require the optional JAX runtime."
        ) from error
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
