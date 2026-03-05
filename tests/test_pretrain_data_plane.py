from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.train.data.access import DatasetAccessError, open_streaming_source
from iris.train.data.contracts import ProfileValidationError, load_default_pure_lm_profile, load_profile
from iris.train.data.planner import build_pure_lm_segment_plan
from iris.train.data.token_accounting import TokenizerError, validate_tokenizer_required


def test_pure_lm_profile_has_nine_sources_and_ratio_sum() -> None:
    profile = load_default_pure_lm_profile()
    assert len(profile.sources) == 9
    assert abs(sum(source.ratio_total for source in profile.sources) - profile.pure_lm_ratio_total) < 1e-9


def test_forbidden_source_is_rejected(tmp_path: Path) -> None:
    payload = {
        "profile_id": "bad-profile",
        "version": "test",
        "pure_lm_ratio_total": 0.9,
        "tokenizer_required": True,
        "sources": [
            {
                "source_id": f"source_{idx}",
                "hf_path": "bigcode/the-stack-v2-dedup" if idx == 0 else "HuggingFaceFW/fineweb-edu",
                "hf_name": None,
                "split": "train",
                "revision": "main",
                "text_field": "text",
                "ratio_total": 0.1,
                "required": True,
            }
            for idx in range(9)
        ],
    }
    profile_path = tmp_path / "bad_profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileValidationError):
        load_profile(profile_path)


def test_tokenizer_is_required_for_streaming_modes() -> None:
    with pytest.raises(TokenizerError):
        validate_tokenizer_required("pure_lm_streaming", None)
    with pytest.raises(TokenizerError):
        validate_tokenizer_required("hybrid_mixture", "")
    validate_tokenizer_required("synthetic", None)


def test_deterministic_dataset_slice_reproducibility() -> None:
    profile = load_default_pure_lm_profile()
    plan_a = build_pure_lm_segment_plan(
        profile=profile,
        segment_id=3,
        micro_steps=8,
        tokens_per_micro_step=128,
        data_seed=17,
        tokenizer_fingerprint="tok-fp",
    )
    plan_b = build_pure_lm_segment_plan(
        profile=profile,
        segment_id=3,
        micro_steps=8,
        tokens_per_micro_step=128,
        data_seed=17,
        tokenizer_fingerprint="tok-fp",
    )
    plan_c = build_pure_lm_segment_plan(
        profile=profile,
        segment_id=4,
        micro_steps=8,
        tokens_per_micro_step=128,
        data_seed=17,
        tokenizer_fingerprint="tok-fp",
    )

    assert plan_a.dataset_slice_id == plan_b.dataset_slice_id
    assert plan_a.plan_hash == plan_b.plan_hash
    assert plan_a.dataset_slice_id != plan_c.dataset_slice_id


def test_missing_required_local_snapshot_fails(tmp_path: Path) -> None:
    profile = load_default_pure_lm_profile()
    source = profile.sources[0]
    with pytest.raises(DatasetAccessError):
        open_streaming_source(
            source,
            streaming_mode="local_snapshot",
            snapshot_root=tmp_path / "does-not-exist-local-snapshot",
        )
