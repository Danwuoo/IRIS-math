from .encoding import ArcEncodingConfig, encode_arc_case_to_state, infer_target_shape
from .benchmark_bridge import export_benchmark_submission, load_benchmark_tasks
from .inference import ArcDiagnosticRunner, ArcEvalConfig, aggregate_failure_histogram, run_arc_diagnostic_eval
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
