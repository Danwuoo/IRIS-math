from __future__ import annotations

from iris.train import IRIS3BConfig, default_iris3b_config, iris3b_config_from_mapping


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
