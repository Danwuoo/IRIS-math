from __future__ import annotations

from iris.train.data.p1_iterator import P1BatchPlan, P1StreamingProvider
from iris.train.data.p1_manifest import P1StreamingManifest, P1StreamingSource
from iris.train.data.token_accounting import TokenizerHandle


class _DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        if not text:
            return []
        return list(range(1, len(text.split()) + 1))

    def decode(self, token_ids, clean_up_tokenization_spaces: bool = False):
        return " ".join(f"tok-{token_id}" for token_id in token_ids)


class _FailOnceIterable:
    def __init__(self, state: dict[str, int], payload: dict[str, str]) -> None:
        self._state = state
        self._payload = payload

    def __iter__(self):
        if self._state["failures_remaining"] > 0:
            self._state["failures_remaining"] -= 1
            raise OSError("bad file descriptor")
        yield dict(self._payload)


def test_streaming_provider_recovers_from_iterator_error() -> None:
    manifest = P1StreamingManifest(
        schema="p1_streaming_manifest/v1",
        manifest_id="p1-stream-retry-test",
        profile_id="P1",
        phase="E",
        commit_posture="committed",
        default_streaming_mode="auto",
        sources=(
            P1StreamingSource(
                source_id="fineweb_edu_math_core",
                source_kind="hf",
                pool_id="A",
                pool_role="core",
                token_weight=0.20,
                record_weight=0.25,
                hf_path="HuggingFaceFW/fineweb-edu",
                hf_name="sample-10BT",
                split="train",
                revision="main",
                payload_field="text",
            ),
        ),
    )
    loader_state = {"calls": 0, "failures_remaining": 1}

    def loader(dataset_or_builder: str, **kwargs):
        loader_state["calls"] += 1
        return _FailOnceIterable(
            loader_state,
            {
                "text": (
                    "This mathematical exposition contains enough prose and structure "
                    "to survive the cleaner after the stream is reopened."
                ),
                "url": "https://example.org/math/reopened-stream",
            },
        )

    provider = P1StreamingProvider(
        manifest=manifest,
        tokenizer_handle=TokenizerHandle(
            id_or_path="dummy",
            tokenizer=_DummyTokenizer(),
            fingerprint="dummy-tokenizer",
        ),
        run_id="retry",
        data_seed=17,
        streaming_mode="auto",
        cache_root=None,
        snapshot_root=None,
        snapshot_fallback_root=None,
        sequence_pack_tokens=32,
        micro_batch_size=1,
        aux_target_dim=8,
        state_target_hidden_dim=8,
        loader=loader,
        max_records_scan=16,
        max_source_read_retries=2,
    )

    batch = provider.sample_batch(
        segment_id=0,
        dataset_slice_id="retry-slice",
        batch_plan=P1BatchPlan(
            batch_index=0,
            source_id="fineweb_edu_math_core",
            pool_id="A",
            target_tokens=32,
            sample_key="retry-key",
        ),
    )

    assert batch.source_id == "fineweb_edu_math_core"
    assert batch.token_count > 0
    assert batch.metadata[0]["records_used"] >= 1
    assert loader_state["calls"] >= 3


def test_prefetched_batches_preserve_serial_order_for_synthetic_sources() -> None:
    manifest = P1StreamingManifest(
        schema="p1_streaming_manifest/v1",
        manifest_id="p1-prefetch-test",
        profile_id="P1",
        phase="E",
        commit_posture="committed",
        default_streaming_mode="auto",
        sources=(
            P1StreamingSource(
                source_id="synth_a",
                source_kind="synthetic",
                pool_id="A",
                pool_role="core",
                token_weight=0.5,
                record_weight=0.5,
                revision="synthetic-v1",
            ),
            P1StreamingSource(
                source_id="synth_b",
                source_kind="synthetic",
                pool_id="B",
                pool_role="core",
                token_weight=0.5,
                record_weight=0.5,
                revision="synthetic-v1",
            ),
        ),
    )
    provider = P1StreamingProvider(
        manifest=manifest,
        tokenizer_handle=TokenizerHandle(
            id_or_path="dummy",
            tokenizer=_DummyTokenizer(),
            fingerprint="dummy-tokenizer",
        ),
        run_id="prefetch",
        data_seed=23,
        streaming_mode="auto",
        cache_root=None,
        snapshot_root=None,
        snapshot_fallback_root=None,
        sequence_pack_tokens=24,
        micro_batch_size=1,
        aux_target_dim=8,
        state_target_hidden_dim=8,
    )
    plan = provider.build_segment_plan(
        segment_id=0,
        optimizer_steps=2,
        gradient_accumulation_steps=2,
    )

    serial = [
        provider.sample_batch(
            segment_id=0,
            dataset_slice_id=plan.dataset_slice_id,
            batch_plan=batch_plan,
        )
        for batch_plan in plan.steps
    ]
    prefetched = list(
        provider.iter_prefetched_batches(
            segment_id=0,
            dataset_slice_id=plan.dataset_slice_id,
            batch_plans=plan.steps,
            max_workers=4,
            prefetch_batches=4,
        )
    )

    assert [batch.source_id for batch in prefetched] == [batch.source_id for batch in serial]
    assert [batch.sample_keys for batch in prefetched] == [batch.sample_keys for batch in serial]
    assert [batch.token_count for batch in prefetched] == [batch.token_count for batch in serial]
