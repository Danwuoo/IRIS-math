from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..levels import LEVEL_IDS, apply_level_stack_params
from ..runtime import assert_jax_runtime
from ..schema import STATE_IR_TOKEN_ORDER
from ..train.checkpoint import load_checkpoint
from ..train.journal import last_applied_event, load_journal
from ..trunk import build_typed_sequence, forward_with_params
from .decoding import ArcDecoder
from .encoding import ArcEncodingConfig, encode_arc_case_to_state, infer_target_shape
from .types import (
    ArcExample,
    ArcInferenceRecord,
    ArcTask,
    FAILURE_CODES,
    dominant_failure_code,
    failure_credit_to_code_distribution,
    neutral_failure_histogram,
    normalize_failure_histogram,
)


def _tree_to_jax(tree: object) -> object:
    return jax.tree_util.tree_map(lambda value: jnp.asarray(np.asarray(value, dtype=np.float32)), tree)


def _state_section_lengths_tuple(state: object) -> tuple[int, ...]:
    lengths = state.section_lengths()
    return tuple(int(lengths[token_type]) for token_type in STATE_IR_TOKEN_ORDER)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _grid_exact_match(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> bool:
    if len(left) != len(right):
        return False
    if not left:
        return False
    for row_left, row_right in zip(left, right):
        if len(row_left) != len(row_right):
            return False
        if any(int(a) != int(b) for a, b in zip(row_left, row_right)):
            return False
    return True


def _cell_accuracy(input_grid: Sequence[Sequence[int]], output_grid: Sequence[Sequence[int]]) -> float:
    max_rows = max(len(input_grid), len(output_grid))
    max_cols = max(
        max((len(row) for row in input_grid), default=0),
        max((len(row) for row in output_grid), default=0),
    )
    if max_rows == 0 or max_cols == 0:
        return 0.0
    matches = 0
    total = max_rows * max_cols
    for row_idx in range(max_rows):
        for col_idx in range(max_cols):
            left_value = 0
            right_value = 0
            if row_idx < len(input_grid) and col_idx < len(input_grid[row_idx]):
                left_value = int(input_grid[row_idx][col_idx])
            if row_idx < len(output_grid) and col_idx < len(output_grid[row_idx]):
                right_value = int(output_grid[row_idx][col_idx])
            if left_value == right_value:
                matches += 1
    return float(matches) / float(total)


@dataclass(frozen=True)
class ArcEvalConfig:
    model_run_dir: Path
    max_reasoning_cycles: int = 3
    termination_threshold: float = 0.7
    seed: int = 17
    device: str = "cpu"


class ArcDiagnosticRunner:
    def __init__(
        self,
        *,
        config: ArcEvalConfig,
        encoding_config: ArcEncodingConfig | None = None,
        level_alpha: float = 0.1,
    ) -> None:
        self.config = config
        self.encoding_config = encoding_config or ArcEncodingConfig()
        self.level_alpha = float(level_alpha)
        self._load_model()

    def _load_model(self) -> None:
        assert_jax_runtime(device=self.config.device, require_gpu=False)
        run_dir = Path(self.config.model_run_dir)
        journal_path = run_dir / "segment_journal.jsonl"
        events = load_journal(journal_path)
        applied_event = last_applied_event(events)
        if applied_event is None:
            raise RuntimeError(f"No APPLIED event found in journal: {journal_path}")
        checkpoint_path = Path(str(applied_event["checkpoint_ref"]))
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / checkpoint_path
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint path does not exist: {applied_event['checkpoint_ref']}")

        checkpoint = load_checkpoint(checkpoint_path)
        model_state = checkpoint.get("model_state", {})
        if model_state.get("schema") != "iris.model_state/v2":
            raise RuntimeError("Unsupported model_state schema. Expected iris.model_state/v2.")
        self.hidden_dim = int(model_state["hidden_dim"])
        self.trunk_params = _tree_to_jax(model_state["trunk"])
        self.level_params = _tree_to_jax(model_state["levels"])
        self.decoder = ArcDecoder.from_model_params(model_state["trunk"])

    def _run_reasoning_cycles(self, state) -> tuple[np.ndarray, Dict[str, float], float]:
        failure_credit = {level_id: 1.0 / len(LEVEL_IDS) for level_id in LEVEL_IDS}
        sequence_np = state.to_canonical_sequence()
        confidence = 0.0
        max_cycles = max(int(self.config.max_reasoning_cycles), 1)
        for _ in range(max_cycles):
            base_sequence = jnp.asarray(sequence_np, dtype=jnp.float32)
            level_sequence, _, l6_credit_arr = apply_level_stack_params(
                self.level_params,
                base_sequence,
                alpha=self.level_alpha,
            )
            typed_sequence = build_typed_sequence(
                sequence=level_sequence,
                section_lengths=_state_section_lengths_tuple(state),
                type_embeddings=self.trunk_params["type_embeddings"],
            )
            pred_sequence, trunk_control = forward_with_params(self.trunk_params, typed_sequence)
            sequence_np = np.asarray(pred_sequence, dtype=np.float32)
            state = state.with_updated_sequence(sequence_np)
            failure_credit = {
                level_id: float(np.asarray(l6_credit_arr[index]))
                for index, level_id in enumerate(LEVEL_IDS)
            }
            termination_conf = _sigmoid(float(np.asarray(trunk_control[7])))
            confidence = max(confidence, termination_conf)
            if termination_conf >= float(self.config.termination_threshold):
                break
        return sequence_np, failure_credit, confidence

    def evaluate_case(
        self,
        *,
        task: ArcTask,
        case_index: int,
        mode: str,
        train_examples_override: Sequence[ArcExample] | None = None,
    ) -> ArcInferenceRecord:
        if case_index < 0 or case_index >= len(task.test_examples):
            raise IndexError(f"case_index out of range for task '{task.task_id}': {case_index}")
        test_example = task.test_examples[case_index]
        state = encode_arc_case_to_state(
            task=task,
            test_example=test_example,
            hidden_dim=self.hidden_dim,
            config=self.encoding_config,
            train_examples_override=train_examples_override,
        )
        final_sequence, failure_credit, termination_conf = self._run_reasoning_cycles(state)
        target_shape = infer_target_shape(task, test_example)
        predicted_grid, decode_conf = self.decoder.decode_grid(
            final_sequence,
            output_shape=target_shape,
        )
        validity_score = _cell_accuracy(predicted_grid, test_example.output_grid)
        success = _grid_exact_match(predicted_grid, test_example.output_grid)
        confidence = float(0.6 * termination_conf + 0.4 * decode_conf)
        failure_distribution = failure_credit_to_code_distribution(failure_credit)
        failure_code = dominant_failure_code(failure_distribution)
        return ArcInferenceRecord(
            task_id=task.task_id,
            source_name=task.source_name,
            concept_id=task.concept_id,
            mode=mode,
            case_index=case_index,
            success=bool(success),
            validity_score=float(validity_score),
            confidence=float(confidence),
            failure_credit={level_id: float(failure_credit.get(level_id, 0.0)) for level_id in LEVEL_IDS},
            failure_code_distribution=failure_distribution,
            failure_code=failure_code,
            input_shape=test_example.input_shape,
            target_shape=test_example.output_shape,
            predicted_shape=(len(predicted_grid), len(predicted_grid[0]) if predicted_grid else 0),
            predicted_grid=predicted_grid,
        )


def run_arc_diagnostic_eval(
    *,
    tasks: Sequence[ArcTask],
    config: ArcEvalConfig,
    mode: str,
    train_override_resolver: Callable[[ArcTask, int], Sequence[ArcExample] | None] | None = None,
) -> List[ArcInferenceRecord]:
    runner = ArcDiagnosticRunner(config=config)
    records: List[ArcInferenceRecord] = []
    for task in tasks:
        for case_index in range(len(task.test_examples)):
            train_override = None
            if train_override_resolver is not None:
                train_override = train_override_resolver(task, case_index)
            records.append(
                runner.evaluate_case(
                    task=task,
                    case_index=case_index,
                    mode=mode,
                    train_examples_override=train_override,
                )
            )
    return records


def aggregate_failure_histogram(
    records: Iterable[ArcInferenceRecord],
    *,
    failures_only: bool = True,
) -> Dict[str, float]:
    histogram = neutral_failure_histogram()
    for record in records:
        if failures_only and record.success:
            continue
        for code in FAILURE_CODES:
            histogram[code] += float(record.failure_code_distribution.get(code, 0.0))
    return normalize_failure_histogram(histogram)

