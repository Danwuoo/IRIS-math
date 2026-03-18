from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..governance import stable_hash
from .access import DatasetAccessError, LoadDatasetFn, open_streaming_source
from .contracts import DatasetSourceSpec
from .filters import prepare_clean_text
from .p1_manifest import P1StreamingManifest, P1StreamingSource, manifest_sha256
from .state_builder import text_to_state_ir
from .token_accounting import (
    TokenizerHandle,
    count_tokens,
    truncate_text_to_tokens,
)

_LEVEL_IDS: Tuple[str, ...] = tuple(f"L{index}" for index in range(7))
_POOL_SCALE = {"A": 1.00, "B": 0.95, "C": 1.05, "D": 0.90, "E": 1.10}
_SOURCE_KIND_SCALE = {"hf": 1.00, "synthetic": 1.08}
_PROJECTION_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


@dataclass(frozen=True)
class P1BatchPlan:
    batch_index: int
    source_id: str
    pool_id: str
    target_tokens: int
    sample_key: str


@dataclass(frozen=True)
class P1SegmentPlan:
    segment_id: int
    dataset_slice_id: str
    plan_hash: str
    steps: Tuple[P1BatchPlan, ...]
    total_target_tokens: int
    requested_streaming_mode: str
    effective_streaming_mode: str


@dataclass(frozen=True)
class P1SourceManifest:
    manifest_id: str
    manifest_sha256: str
    commit_posture: str
    requested_streaming_mode: str
    effective_streaming_mode: str
    source_modes: Dict[str, str]


@dataclass(frozen=True)
class P1Batch:
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    aux_targets: np.ndarray
    aux_mask: np.ndarray
    token_count: int
    source_id: str
    pool_id: str
    effective_mode: str
    sample_keys: Tuple[str, ...]
    state_irs: Tuple[Any, ...]
    metadata: Tuple[Dict[str, Any], ...]


def _coerce_dataset_spec(source: P1StreamingSource) -> DatasetSourceSpec:
    return DatasetSourceSpec(
        source_id=source.source_id,
        hf_path=str(source.hf_path or ""),
        hf_name=source.hf_name,
        split=source.split,
        revision=str(source.revision or ""),
        text_field=source.payload_field,
        ratio_total=float(source.token_weight),
        required=bool(source.required),
        local_snapshot_pattern=source.local_snapshot_pattern,
        metadata=dict(source.metadata),
    )


def _stable_seed(payload: Mapping[str, Any]) -> int:
    return int(stable_hash(dict(payload))[:16], 16)


def _allocate_step_counts(weights: Mapping[str, float], total_steps: int) -> Dict[str, int]:
    if total_steps <= 0:
        return {str(source_id): 0 for source_id in weights.keys()}
    expected = {
        str(source_id): float(weight) * float(total_steps)
        for source_id, weight in weights.items()
    }
    counts = {source_id: int(value) for source_id, value in expected.items()}
    allocated = int(sum(counts.values()))
    remainder = max(int(total_steps) - allocated, 0)
    ranked = sorted(
        expected.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0]),
        reverse=True,
    )
    for index in range(remainder):
        source_id = ranked[index % len(ranked)][0]
        counts[source_id] = int(counts.get(source_id, 0)) + 1
    return counts


def deterministic_p1_sampling_key(
    *,
    run_id: str,
    dataset_slice_id: str,
    segment_id: int,
    batch_index: int,
    example_index: int,
    data_seed: int,
) -> str:
    return stable_hash(
        {
            "run_id": str(run_id),
            "dataset_slice_id": str(dataset_slice_id),
            "segment_id": int(segment_id),
            "batch_index": int(batch_index),
            "example_index": int(example_index),
            "data_seed": int(data_seed),
        }
    )


def _projection_matrix(level_index: int, input_dim: int, output_dim: int) -> np.ndarray:
    cache_key = (int(level_index), int(input_dim), int(output_dim))
    cached = _PROJECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(_stable_seed({"level_index": level_index, "input_dim": input_dim, "output_dim": output_dim}))
    scale = 1.0 / max(float(input_dim), 1.0)
    matrix = rng.normal(loc=0.0, scale=scale, size=(int(input_dim), int(output_dim))).astype(np.float32)
    _PROJECTION_CACHE[cache_key] = matrix
    return matrix


def _mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    weights = attention_mask.astype(np.float32)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return np.mean(hidden_states.astype(np.float32), axis=0)
    return np.sum(hidden_states.astype(np.float32) * weights[:, None], axis=0) / denom


def _derive_level_targets(
    *,
    text: str,
    tokenizer: Any,
    pool_id: str,
    source_kind: str,
    aux_target_dim: int,
    state_target_hidden_dim: int,
) -> Tuple[np.ndarray, Any]:
    state = text_to_state_ir(
        text=text,
        tokenizer=tokenizer,
        hidden_dim=max(int(state_target_hidden_dim), int(aux_target_dim), 8),
        max_input_tokens=256,
    )
    sequence = np.asarray(state.to_canonical_sequence(), dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence.reshape(1, -1)
    if sequence.shape[0] <= 0:
        sequence = np.zeros((1, max(int(state_target_hidden_dim), int(aux_target_dim), 8)), dtype=np.float32)
    positions = np.linspace(0, sequence.shape[0] - 1, num=len(_LEVEL_IDS), dtype=int)
    scale = float(_POOL_SCALE.get(pool_id, 1.0)) * float(_SOURCE_KIND_SCALE.get(source_kind, 1.0))
    targets: List[np.ndarray] = []
    for level_index, row_index in enumerate(positions):
        row = sequence[int(row_index)]
        projection = _projection_matrix(level_index, row.shape[-1], int(aux_target_dim))
        projected = np.tanh(np.matmul(row, projection) * scale).astype(np.float32)
        targets.append(projected)
    return np.stack(targets, axis=0).astype(np.float32), state


def _normalize_effective_mode(modes: Sequence[str]) -> str:
    normalized = sorted({str(mode).strip() for mode in modes if str(mode).strip()})
    if not normalized:
        return "unavailable"
    if len(normalized) == 1:
        return normalized[0]
    return "mixed"


class P1StreamingProvider:
    def __init__(
        self,
        *,
        manifest: P1StreamingManifest,
        tokenizer_handle: TokenizerHandle,
        run_id: str,
        data_seed: int,
        streaming_mode: str,
        cache_root: Path | None,
        snapshot_root: Path | None,
        sequence_pack_tokens: int,
        micro_batch_size: int,
        aux_target_dim: int,
        state_target_hidden_dim: int,
        loader: LoadDatasetFn | None = None,
        max_offset_records: int = 256,
        max_records_scan: int = 4096,
        max_source_read_retries: int = 3,
    ) -> None:
        self.manifest = manifest
        self.tokenizer_handle = tokenizer_handle
        self.run_id = str(run_id)
        self.data_seed = int(data_seed)
        self.streaming_mode = str(streaming_mode).strip().lower() or manifest.default_streaming_mode
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.snapshot_root = Path(snapshot_root) if snapshot_root is not None else None
        self.sequence_pack_tokens = int(max(sequence_pack_tokens, 8))
        self.micro_batch_size = int(max(micro_batch_size, 1))
        self.aux_target_dim = int(max(aux_target_dim, 1))
        self.state_target_hidden_dim = int(max(state_target_hidden_dim, self.aux_target_dim))
        self.loader = loader
        self.max_offset_records = int(max(max_offset_records, 1))
        self.max_records_scan = int(max(max_records_scan, 64))
        self.max_source_read_retries = int(max(max_source_read_retries, 0))

        self._source_by_id = {source.source_id: source for source in self.manifest.sources}
        self._effective_mode_by_source: Dict[str, str] = {}
        self._available_source_ids: List[str] = []
        self._preflight_sources()

    def _preflight_sources(self) -> None:
        for source in self.manifest.sources:
            if source.source_kind == "synthetic":
                self._effective_mode_by_source[source.source_id] = "synthetic"
                self._available_source_ids.append(source.source_id)
                continue
            dataset_spec = _coerce_dataset_spec(source)
            try:
                opened = open_streaming_source(
                    dataset_spec,
                    streaming_mode=self.streaming_mode,
                    snapshot_root=self.snapshot_root,
                    loader=self.loader,
                )
            except Exception as error:
                if source.required:
                    raise DatasetAccessError(
                        f"Required P1 source '{source.source_id}' is unavailable: {error}"
                    ) from error
                continue
            self._effective_mode_by_source[source.source_id] = opened.effective_mode
            self._available_source_ids.append(source.source_id)

    @property
    def source_manifest(self) -> P1SourceManifest:
        effective_mode = _normalize_effective_mode(tuple(self._effective_mode_by_source.values()))
        return P1SourceManifest(
            manifest_id=self.manifest.manifest_id,
            manifest_sha256=manifest_sha256(self.manifest),
            commit_posture=self.manifest.commit_posture,
            requested_streaming_mode=self.streaming_mode,
            effective_streaming_mode=effective_mode,
            source_modes=dict(sorted(self._effective_mode_by_source.items())),
        )

    def _available_sources(self) -> Tuple[P1StreamingSource, ...]:
        return tuple(
            self._source_by_id[source_id]
            for source_id in self._available_source_ids
            if source_id in self._source_by_id
        )

    def build_segment_plan(
        self,
        *,
        segment_id: int,
        optimizer_steps: int,
        gradient_accumulation_steps: int,
    ) -> P1SegmentPlan:
        total_batches = int(max(optimizer_steps, 1)) * int(max(gradient_accumulation_steps, 1))
        sources = self._available_sources()
        if not sources:
            raise DatasetAccessError("No P1 sources are available for sampling.")
        total_weight = float(sum(float(source.token_weight) for source in sources))
        if total_weight <= 0.0:
            raise DatasetAccessError("P1 token weights are invalid after availability filtering.")
        weights = {
            source.source_id: float(source.token_weight) / total_weight
            for source in sources
        }
        step_counts = _allocate_step_counts(weights, total_batches)
        sequence: List[str] = []
        for source_id in sorted(step_counts.keys()):
            sequence.extend([source_id] * int(step_counts[source_id]))
        rng = random.Random(
            _stable_seed(
                {
                    "manifest_sha256": manifest_sha256(self.manifest),
                    "segment_id": int(segment_id),
                    "optimizer_steps": int(optimizer_steps),
                    "gradient_accumulation_steps": int(gradient_accumulation_steps),
                    "data_seed": int(self.data_seed),
                    "tokenizer_fingerprint": self.tokenizer_handle.fingerprint,
                }
            )
        )
        rng.shuffle(sequence)
        step_payload: List[Dict[str, Any]] = []
        steps: List[P1BatchPlan] = []
        for batch_index, source_id in enumerate(sequence):
            source = self._source_by_id[source_id]
            sample_key = stable_hash(
                {
                    "manifest_sha256": manifest_sha256(self.manifest),
                    "segment_id": int(segment_id),
                    "batch_index": int(batch_index),
                    "source_id": source_id,
                    "data_seed": int(self.data_seed),
                }
            )
            step = P1BatchPlan(
                batch_index=int(batch_index),
                source_id=source_id,
                pool_id=source.pool_id,
                target_tokens=self.sequence_pack_tokens,
                sample_key=sample_key,
            )
            steps.append(step)
            step_payload.append(
                {
                    "batch_index": step.batch_index,
                    "source_id": step.source_id,
                    "pool_id": step.pool_id,
                    "target_tokens": step.target_tokens,
                    "sample_key": step.sample_key,
                }
            )
        plan_hash = stable_hash(
            {
                "manifest_sha256": manifest_sha256(self.manifest),
                "segment_id": int(segment_id),
                "data_seed": int(self.data_seed),
                "requested_streaming_mode": self.streaming_mode,
                "effective_streaming_mode": self.source_manifest.effective_streaming_mode,
                "tokenizer_fingerprint": self.tokenizer_handle.fingerprint,
                "steps": step_payload,
            }
        )
        dataset_slice_id = "p1-slice-" + stable_hash(
            {
                "manifest_sha256": manifest_sha256(self.manifest),
                "plan_hash": plan_hash,
                "segment_id": int(segment_id),
                "data_seed": int(self.data_seed),
            }
        )[:16]
        return P1SegmentPlan(
            segment_id=int(segment_id),
            dataset_slice_id=dataset_slice_id,
            plan_hash=plan_hash,
            steps=tuple(steps),
            total_target_tokens=len(steps) * self.sequence_pack_tokens * self.micro_batch_size,
            requested_streaming_mode=self.streaming_mode,
            effective_streaming_mode=self.source_manifest.effective_streaming_mode,
        )

    def _open_source_iterable(self, source_id: str) -> Tuple[P1StreamingSource, Any]:
        source = self._source_by_id[source_id]
        if source.source_kind == "synthetic":
            return source, None
        dataset_spec = _coerce_dataset_spec(source)
        opened = open_streaming_source(
            dataset_spec,
            streaming_mode=self.streaming_mode,
            snapshot_root=self.snapshot_root,
            loader=self.loader,
        )
        return source, opened

    def _reopen_source_iterable(self, source: P1StreamingSource) -> Any:
        dataset_spec = _coerce_dataset_spec(source)
        return open_streaming_source(
            dataset_spec,
            streaming_mode=self.streaming_mode,
            snapshot_root=self.snapshot_root,
            loader=self.loader,
        )

    def _synthetic_text(
        self,
        *,
        source: P1StreamingSource,
        sample_key: str,
        target_tokens: int,
    ) -> str:
        digest = sample_key[:16]
        fragments: List[str] = []
        while count_tokens(self.tokenizer_handle.tokenizer, "\n".join(fragments) or "seed") < int(target_tokens):
            index = len(fragments)
            fragments.append(
                (
                    f"Synthetic IR-aligned sample {index} for source {source.source_id} in pool {source.pool_id} "
                    f"uses digest {digest} and preserves verifier-centered structure, recovery traces, "
                    f"document grounding cues, and explicit governed replay semantics."
                )
            )
        merged = "\n".join(fragments)
        return truncate_text_to_tokens(self.tokenizer_handle.tokenizer, merged, int(target_tokens))

    def _sample_text_record(
        self,
        *,
        source: P1StreamingSource,
        opened: Any,
        sample_key: str,
        target_tokens: int,
    ) -> Tuple[str, int, str]:
        if source.source_kind == "synthetic":
            text = self._synthetic_text(source=source, sample_key=sample_key, target_tokens=target_tokens)
            return text, 1, "synthetic"

        dataset_spec = _coerce_dataset_spec(source)
        offset = int(stable_hash({"sample_key": sample_key, "source_id": source.source_id})[:8], 16)
        offset = offset % self.max_offset_records

        last_read_error: Exception | None = None
        open_state = opened
        total_scanned = 0
        total_records_used = 0
        rejection_counts = {
            "non_mapping": 0,
            "clean_text_rejected": 0,
            "tokenizer_empty": 0,
            "iterator_error": 0,
        }

        for attempt_index in range(self.max_source_read_retries + 1):
            iterator = iter(open_state.iterable)
            skipped = 0
            empty_cycles = 0
            while skipped < offset:
                try:
                    next(iterator)
                except StopIteration:
                    empty_cycles += 1
                    if skipped <= 0 and empty_cycles > max(self.max_source_read_retries + 1, 2):
                        break
                    iterator = iter(open_state.iterable)
                    continue
                except Exception as error:
                    last_read_error = error
                    rejection_counts["iterator_error"] += 1
                    open_state = self._reopen_source_iterable(source)
                    iterator = iter(open_state.iterable)
                    skipped = 0
                    empty_cycles = 0
                    continue
                empty_cycles = 0
                skipped += 1

            remaining_tokens = int(target_tokens)
            texts: List[str] = []
            records_used = 0
            scanned = 0
            while remaining_tokens > 0 and scanned < self.max_records_scan:
                scanned += 1
                total_scanned += 1
                try:
                    record = next(iterator)
                except StopIteration:
                    iterator = iter(open_state.iterable)
                    continue
                except Exception as error:
                    last_read_error = error
                    rejection_counts["iterator_error"] += 1
                    open_state = self._reopen_source_iterable(source)
                    iterator = iter(open_state.iterable)
                    continue
                if not isinstance(record, Mapping):
                    rejection_counts["non_mapping"] += 1
                    continue
                clean_text = prepare_clean_text(dataset_spec, record)
                if not clean_text:
                    rejection_counts["clean_text_rejected"] += 1
                    continue
                clipped = truncate_text_to_tokens(self.tokenizer_handle.tokenizer, clean_text, remaining_tokens)
                token_count = count_tokens(self.tokenizer_handle.tokenizer, clipped)
                if token_count <= 0:
                    rejection_counts["tokenizer_empty"] += 1
                    continue
                texts.append(clipped)
                records_used += 1
                total_records_used += 1
                remaining_tokens -= token_count

            if texts:
                merged = truncate_text_to_tokens(
                    self.tokenizer_handle.tokenizer,
                    "\n\n".join(texts),
                    int(target_tokens),
                )
                return merged, records_used, str(open_state.effective_mode)
            if attempt_index < self.max_source_read_retries:
                open_state = self._reopen_source_iterable(source)

        error_message = (
            f"Failed to sample usable P1 text for source '{source.source_id}' within "
            f"{self.max_records_scan} scans per attempt. total_scanned={total_scanned}; "
            f"records_used={total_records_used}; rejection_counts={json.dumps(rejection_counts, sort_keys=True)}"
        )
        if last_read_error is not None:
            error_message += f"; last_iterator_error={last_read_error}"
        raise DatasetAccessError(error_message)

    def sample_batch(
        self,
        *,
        segment_id: int,
        dataset_slice_id: str,
        batch_plan: P1BatchPlan,
    ) -> P1Batch:
        source, opened = self._open_source_iterable(batch_plan.source_id)
        tokenizer = self.tokenizer_handle.tokenizer
        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        input_rows: List[np.ndarray] = []
        label_rows: List[np.ndarray] = []
        mask_rows: List[np.ndarray] = []
        aux_rows: List[np.ndarray] = []
        aux_mask_rows: List[np.ndarray] = []
        states: List[Any] = []
        metadata_rows: List[Dict[str, Any]] = []
        sample_keys: List[str] = []
        token_count_total = 0

        for example_index in range(self.micro_batch_size):
            sample_key = deterministic_p1_sampling_key(
                run_id=self.run_id,
                dataset_slice_id=dataset_slice_id,
                segment_id=segment_id,
                batch_index=batch_plan.batch_index,
                example_index=example_index,
                data_seed=self.data_seed,
            )
            text, records_used, effective_mode = self._sample_text_record(
                source=source,
                opened=opened,
                sample_key=sample_key,
                target_tokens=batch_plan.target_tokens,
            )
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids = [int(token_id) for token_id in token_ids[: batch_plan.target_tokens]]
            if len(token_ids) < batch_plan.target_tokens:
                token_ids = token_ids + [pad_token_id] * (batch_plan.target_tokens - len(token_ids))
            input_ids = np.asarray(token_ids[:-1], dtype=np.int32)
            labels = np.asarray(token_ids[1:], dtype=np.int32)
            attention_mask = np.asarray(
                [1 if token_id != pad_token_id else 0 for token_id in token_ids[:-1]],
                dtype=np.int32,
            )
            aux_targets, state = _derive_level_targets(
                text=text,
                tokenizer=tokenizer,
                pool_id=source.pool_id,
                source_kind=source.source_kind,
                aux_target_dim=self.aux_target_dim,
                state_target_hidden_dim=self.state_target_hidden_dim,
            )
            aux_mask = np.ones((len(_LEVEL_IDS),), dtype=np.float32)
            benchmark_refs = list(source.benchmark_family_refs)
            metadata_rows.append(
                {
                    "source_id": source.source_id,
                    "pool_id": source.pool_id,
                    "pool_role": source.pool_role,
                    "source_kind": source.source_kind,
                    "effective_mode": effective_mode,
                    "requested_streaming_mode": self.streaming_mode,
                    "streaming_manifest_id": self.manifest.manifest_id,
                    "streaming_manifest_sha256": manifest_sha256(self.manifest),
                    "dataset_slice_id": dataset_slice_id,
                    "sample_key": sample_key,
                    "records_used": int(records_used),
                    "benchmark_family_refs": benchmark_refs,
                    "admission_posture": source.admission_posture,
                    "payload_field": source.payload_field,
                    "filter_signature": source.filter_signature,
                    "heuristic_target_source": "text_to_state_ir/bootstrap_projection_v1",
                    "technical_debt": (
                        "TEMPORARY TECHNICAL DEBT: heuristic State-IR targets remain active until canonical "
                        "projection or verifier-backed labels cover the committed P1 train-visible slice."
                    ),
                }
            )
            input_rows.append(input_ids)
            label_rows.append(labels)
            mask_rows.append(attention_mask)
            aux_rows.append(aux_targets)
            aux_mask_rows.append(aux_mask)
            states.append(state)
            sample_keys.append(sample_key)
            token_count_total += int(np.sum(attention_mask))

        return P1Batch(
            input_ids=np.stack(input_rows, axis=0).astype(np.int32),
            labels=np.stack(label_rows, axis=0).astype(np.int32),
            attention_mask=np.stack(mask_rows, axis=0).astype(np.int32),
            aux_targets=np.stack(aux_rows, axis=0).astype(np.float32),
            aux_mask=np.stack(aux_mask_rows, axis=0).astype(np.float32),
            token_count=int(token_count_total),
            source_id=source.source_id,
            pool_id=source.pool_id,
            effective_mode=self._effective_mode_by_source.get(source.source_id, "synthetic"),
            sample_keys=tuple(sample_keys),
            state_irs=tuple(states),
            metadata=tuple(metadata_rows),
        )
