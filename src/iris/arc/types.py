from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

FAILURE_CODES: Tuple[str, ...] = (
    "F_REP",
    "F_PROC",
    "F_SEARCH",
    "F_MEM",
    "F_ABS",
    "F_EVAL",
)

FAILURE_CODE_LEVEL_MAP: Dict[str, Tuple[str, ...]] = {
    "F_REP": ("L0", "L1"),
    "F_PROC": ("L2",),
    "F_SEARCH": ("L3",),
    "F_MEM": ("L4",),
    "F_ABS": ("L5",),
    "F_EVAL": ("L6",),
}


def neutral_failure_histogram() -> Dict[str, float]:
    return {code: 0.0 for code in FAILURE_CODES}


def normalize_failure_histogram(histogram: Mapping[str, float]) -> Dict[str, float]:
    values = {code: float(histogram.get(code, 0.0)) for code in FAILURE_CODES}
    total = float(sum(values.values()))
    if total <= 0.0:
        return neutral_failure_histogram()
    return {code: values[code] / total for code in FAILURE_CODES}


def dominant_failure_code(histogram: Mapping[str, float]) -> str:
    normalized = normalize_failure_histogram(histogram)
    return max(FAILURE_CODES, key=lambda code: float(normalized.get(code, 0.0)))


def failure_credit_to_code_distribution(failure_credit: Mapping[str, float]) -> Dict[str, float]:
    distribution: Dict[str, float] = {}
    for code, levels in FAILURE_CODE_LEVEL_MAP.items():
        distribution[code] = float(sum(float(failure_credit.get(level, 0.0)) for level in levels))
    return normalize_failure_histogram(distribution)


def validate_grid(grid: Sequence[Sequence[int]], *, field_name: str) -> List[List[int]]:
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a 2D integer grid.")
    rows: List[List[int]] = []
    expected_width: int | None = None
    for row_idx, row in enumerate(grid):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"{field_name}[{row_idx}] must be a list of ints.")
        converted_row: List[int] = []
        for col_idx, value in enumerate(row):
            if not isinstance(value, int):
                raise ValueError(f"{field_name}[{row_idx}][{col_idx}] must be int.")
            if value < 0 or value > 9:
                raise ValueError(f"{field_name}[{row_idx}][{col_idx}] must be in [0, 9].")
            converted_row.append(int(value))
        if expected_width is None:
            expected_width = len(converted_row)
            if expected_width <= 0:
                raise ValueError(f"{field_name} rows must be non-empty.")
        elif len(converted_row) != expected_width:
            raise ValueError(f"{field_name} rows must have consistent widths.")
        rows.append(converted_row)
    if not rows:
        raise ValueError(f"{field_name} must contain at least one row.")
    return rows


def grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols


@dataclass(frozen=True)
class ArcExample:
    input_grid: List[List[int]]
    output_grid: List[List[int]]

    @property
    def input_shape(self) -> Tuple[int, int]:
        return grid_shape(self.input_grid)

    @property
    def output_shape(self) -> Tuple[int, int]:
        return grid_shape(self.output_grid)


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train_examples: Tuple[ArcExample, ...]
    test_examples: Tuple[ArcExample, ...]
    source_path: str
    source_name: str
    concept_id: str | None = None

    @property
    def all_examples(self) -> Tuple[ArcExample, ...]:
        return tuple(self.train_examples) + tuple(self.test_examples)


@dataclass(frozen=True)
class ArcPair:
    task_id: str
    pair_index: int
    left: ArcExample
    right: ArcExample
    source_path: str
    pairing_policy: str = "adjacent"


@dataclass(frozen=True)
class ArcInferenceRecord:
    task_id: str
    source_name: str
    concept_id: str | None
    mode: str
    case_index: int
    success: bool
    validity_score: float
    confidence: float
    failure_credit: Dict[str, float]
    failure_code_distribution: Dict[str, float]
    failure_code: str
    input_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    predicted_shape: Tuple[int, int]
    predicted_grid: List[List[int]] = field(repr=False)

