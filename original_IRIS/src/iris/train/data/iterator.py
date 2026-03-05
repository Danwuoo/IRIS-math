from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from .access import DatasetAccessError, LoadDatasetFn, open_streaming_source
from .contracts import PureLMProfile, sources_manifest_sha256
from .filters import prepare_clean_text
from .planner import MicroStepPlan, SegmentPlan, build_pure_lm_segment_plan
from .token_accounting import TokenizerHandle, count_tokens, truncate_text_to_tokens


@dataclass(frozen=True)
class TextBatch:
    source_id: str
    text: str
    token_count: int
    records_used: int
    effective_mode: str
    sampling_key: str


@dataclass(frozen=True)
class SourceManifest:
    profile_id: str
    sources_manifest_sha256: str
    effective_mode: str


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_sampling_key(
    *,
    run_id: str,
    dataset_slice_id: str,
    segment_id: int,
    micro_step_idx: int,
    data_seed: int,
) -> str:
    return _stable_hash(
        {
            "run_id": str(run_id),
            "dataset_slice_id": str(dataset_slice_id),
            "segment_id": int(segment_id),
            "micro_step_idx": int(micro_step_idx),
            "data_seed": int(data_seed),
        }
    )


class PureLMStreamingProvider:
    def __init__(
        self,
        *,
        profile: PureLMProfile,
        tokenizer_handle: TokenizerHandle,
        run_id: str,
        data_seed: int,
        streaming_mode: str,
        snapshot_root: Path | None,
        loader: LoadDatasetFn | None = None,
        max_offset_records: int = 256,
        max_records_scan: int = 4096,
    ) -> None:
        self.profile = profile
        self.tokenizer_handle = tokenizer_handle
        self.run_id = str(run_id)
        self.data_seed = int(data_seed)
        self.streaming_mode = str(streaming_mode)
        self.snapshot_root = Path(snapshot_root) if snapshot_root is not None else None
        self.loader = loader
        self.max_offset_records = int(max(max_offset_records, 1))
        self.max_records_scan = int(max(max_records_scan, 64))

        self._effective_mode_by_source: Dict[str, str] = {}
        self._preflight_sources()

    def _preflight_sources(self) -> None:
        for source in self.profile.sources:
            try:
                opened = open_streaming_source(
                    source,
                    streaming_mode=self.streaming_mode,
                    snapshot_root=self.snapshot_root,
                    loader=self.loader,
                )
            except Exception as error:
                if source.required:
                    raise DatasetAccessError(
                        f"Required source '{source.source_id}' is unavailable: {error}"
                    ) from error
                continue
            self._effective_mode_by_source[source.source_id] = opened.effective_mode

    @property
    def sources_manifest(self) -> SourceManifest:
        modes = set(self._effective_mode_by_source.values())
        if len(modes) == 1:
            effective_mode = next(iter(modes))
        elif not modes:
            effective_mode = "unavailable"
        else:
            effective_mode = "mixed"
        return SourceManifest(
            profile_id=self.profile.profile_id,
            sources_manifest_sha256=sources_manifest_sha256(self.profile),
            effective_mode=effective_mode,
        )

    def build_segment_plan(
        self,
        *,
        segment_id: int,
        micro_steps: int,
        tokens_per_micro_step: int,
    ) -> SegmentPlan:
        return build_pure_lm_segment_plan(
            profile=self.profile,
            segment_id=segment_id,
            micro_steps=micro_steps,
            tokens_per_micro_step=tokens_per_micro_step,
            data_seed=self.data_seed,
            tokenizer_fingerprint=self.tokenizer_handle.fingerprint,
        )

    def _open_source_iterable(self, source_id: str):
        source = self.profile.source_by_id[source_id]
        opened = open_streaming_source(
            source,
            streaming_mode=self.streaming_mode,
            snapshot_root=self.snapshot_root,
            loader=self.loader,
        )
        return source, opened

    def sample_micro_step_text(
        self,
        *,
        segment_id: int,
        dataset_slice_id: str,
        micro_step_plan: MicroStepPlan,
    ) -> TextBatch:
        sampling_key = deterministic_sampling_key(
            run_id=self.run_id,
            dataset_slice_id=dataset_slice_id,
            segment_id=segment_id,
            micro_step_idx=micro_step_plan.micro_step_idx,
            data_seed=self.data_seed,
        )
        source, opened = self._open_source_iterable(micro_step_plan.source_id)
        iterator = iter(opened.iterable)

        offset_seed = _stable_hash(
            {
                "sampling_key": sampling_key,
                "source_id": source.source_id,
                "sample_key": micro_step_plan.sample_key,
            }
        )
        offset = int(offset_seed[:8], 16) % self.max_offset_records

        skipped = 0
        while skipped < offset:
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(opened.iterable)
                skipped = 0
                continue
            skipped += 1

        remaining_tokens = int(micro_step_plan.target_tokens)
        texts = []
        records_used = 0
        scanned = 0
        while remaining_tokens > 0 and scanned < self.max_records_scan:
            scanned += 1
            try:
                record = next(iterator)
            except StopIteration:
                iterator = iter(opened.iterable)
                continue
            if not isinstance(record, Mapping):
                continue
            clean_text = prepare_clean_text(source, record)
            if not clean_text:
                continue
            clipped = truncate_text_to_tokens(
                self.tokenizer_handle.tokenizer,
                clean_text,
                remaining_tokens,
            )
            token_count = count_tokens(self.tokenizer_handle.tokenizer, clipped)
            if token_count <= 0:
                continue
            texts.append(clipped)
            records_used += 1
            remaining_tokens -= token_count

        if not texts:
            raise DatasetAccessError(
                f"Failed to sample usable text for source '{source.source_id}' within {self.max_records_scan} scans."
            )

        merged = "\n\n".join(texts)
        merged = truncate_text_to_tokens(
            self.tokenizer_handle.tokenizer,
            merged,
            int(micro_step_plan.target_tokens),
        )
        merged_token_count = count_tokens(self.tokenizer_handle.tokenizer, merged)
        if merged_token_count <= 0:
            raise DatasetAccessError(
                f"Token accounting underflow for source '{source.source_id}' (target={micro_step_plan.target_tokens})."
            )

        return TextBatch(
            source_id=source.source_id,
            text=merged,
            token_count=merged_token_count,
            records_used=records_used,
            effective_mode=opened.effective_mode,
            sampling_key=sampling_key,
        )
