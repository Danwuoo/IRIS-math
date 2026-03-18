from __future__ import annotations

from dataclasses import replace

from iris.train import (
    IRIS3BConfig,
    cycle_memory_profile_candidates,
    default_iris3b_config,
    iris3b_config_from_mapping,
    kaggle_emergency_iris3b_config,
    kaggle_safe_iris3b_config,
    kaggle_safer_iris3b_config,
    kaggle_survival_iris3b_config,
    select_cycle_iris3b_config,
)


def test_default_iris3b_config_matches_3b_band_and_level_schedule() -> None:
    config = default_iris3b_config()

    assert config.schema == "iris.iris3b_config/v1"
    assert config.hidden_size == 2560
    assert config.num_layers == 36
    assert config.num_attention_heads == 20
    assert config.ffn_hidden_size == 6912
    assert config.estimated_parameter_count() == 2_984_630_896
    assert 2_800_000_000 <= config.estimated_parameter_count() <= 3_200_000_000
    assert [(item.level_id, item.block_index) for item in config.level_schedule] == [
        ("L0", 0),
        ("L1", 6),
        ("L2", 12),
        ("L3", 18),
        ("L4", 24),
        ("L5", 30),
        ("L6", 36),
    ]


def test_iris3b_config_round_trips_from_payload() -> None:
    payload = default_iris3b_config().to_payload()

    restored = iris3b_config_from_mapping(payload)

    assert restored == default_iris3b_config()
    assert restored.to_payload()["config_sha256"] == default_iris3b_config().sha256


def test_kaggle_safe_config_preserves_3b_band_but_reduces_context_length() -> None:
    config = kaggle_safe_iris3b_config()

    assert config.config_id == "p1-iris3b-kaggle-safe-v1"
    assert config.hidden_size == 2560
    assert config.num_layers == 36
    assert config.estimated_parameter_count() == default_iris3b_config().estimated_parameter_count()
    assert config.max_sequence_length == 1024
    assert config.sequence_pack_tokens == 1025


def test_lower_kaggle_fallback_profiles_reduce_context_progressively() -> None:
    safer = kaggle_safer_iris3b_config()
    emergency = kaggle_emergency_iris3b_config()
    survival = kaggle_survival_iris3b_config()

    assert safer.estimated_parameter_count() == default_iris3b_config().estimated_parameter_count()
    assert safer.max_sequence_length == 768
    assert safer.sequence_pack_tokens == 769
    assert emergency.estimated_parameter_count() == default_iris3b_config().estimated_parameter_count()
    assert emergency.max_sequence_length == 512
    assert emergency.sequence_pack_tokens == 513
    assert emergency.param_dtype == "bfloat16"
    assert survival.estimated_parameter_count() == default_iris3b_config().estimated_parameter_count()
    assert survival.max_sequence_length == 384
    assert survival.sequence_pack_tokens == 385
    assert survival.param_dtype == "bfloat16"


def test_select_cycle_config_uses_kaggle_safe_profile_only_when_requested() -> None:
    auto_default = select_cycle_iris3b_config(memory_profile="auto", kaggle_runtime=False)
    auto_kaggle = select_cycle_iris3b_config(memory_profile="auto", kaggle_runtime=True)
    explicit_default = select_cycle_iris3b_config(memory_profile="default", kaggle_runtime=True)
    explicit_safer = select_cycle_iris3b_config(memory_profile="kaggle_safer", kaggle_runtime=True)
    explicit_emergency = select_cycle_iris3b_config(memory_profile="kaggle_emergency", kaggle_runtime=True)
    explicit_survival = select_cycle_iris3b_config(memory_profile="kaggle_survival", kaggle_runtime=True)

    assert auto_default == default_iris3b_config()
    assert auto_kaggle == kaggle_safe_iris3b_config()
    assert explicit_default == default_iris3b_config()
    assert explicit_safer == kaggle_safer_iris3b_config()
    assert explicit_emergency == kaggle_emergency_iris3b_config()
    assert explicit_survival == kaggle_survival_iris3b_config()


def test_select_cycle_config_applies_memory_profile_overrides_to_explicit_config() -> None:
    explicit = replace(
        default_iris3b_config(),
        config_id="custom-explicit",
        max_sequence_length=1536,
        sequence_pack_tokens=1537,
        param_dtype="float32",
    ).validate()

    preserved = select_cycle_iris3b_config(
        explicit_config=explicit,
        memory_profile="default",
        kaggle_runtime=True,
    )
    auto_kaggle = select_cycle_iris3b_config(
        explicit_config=explicit,
        memory_profile="auto",
        kaggle_runtime=True,
    )
    emergency = select_cycle_iris3b_config(
        explicit_config=explicit,
        memory_profile="kaggle_emergency",
        kaggle_runtime=True,
    )

    assert preserved == explicit
    assert auto_kaggle.max_sequence_length == 1024
    assert auto_kaggle.sequence_pack_tokens == 1025
    assert auto_kaggle.hidden_size == explicit.hidden_size
    assert emergency.max_sequence_length == 512
    assert emergency.sequence_pack_tokens == 513
    assert emergency.param_dtype == "bfloat16"


def test_cycle_memory_profile_candidates_retry_for_auto_kaggle() -> None:
    assert cycle_memory_profile_candidates(memory_profile="auto", kaggle_runtime=False) == (
        "auto",
    )
    assert cycle_memory_profile_candidates(memory_profile="auto", kaggle_runtime=True) == (
        "kaggle_safe",
        "kaggle_safer",
        "kaggle_emergency",
        "kaggle_survival",
    )
