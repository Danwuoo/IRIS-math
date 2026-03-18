from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Tuple

from .governance import stable_hash


@dataclass(frozen=True)
class InterleavedLevelPlacement:
    level_id: str
    block_index: int


@dataclass(frozen=True)
class IRIS3BConfig:
    schema: str = "iris.iris3b_config/v1"
    config_id: str = "p1-iris3b-v1"
    profile_id: str = "P1"
    phase: str = "E"
    vocab_size: int = 50_176
    hidden_size: int = 2_560
    num_layers: int = 36
    num_attention_heads: int = 20
    ffn_hidden_size: int = 6_912
    max_sequence_length: int = 2_048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1.0e-6
    dropout_rate: float = 0.0
    tie_word_embeddings: bool = True
    use_remat: bool = True
    dtype: str = "bfloat16"
    param_dtype: str = "float32"
    initializer_range: float = 0.02
    aux_target_dim: int = 16
    level_schedule: Tuple[InterleavedLevelPlacement, ...] = field(
        default_factory=lambda: (
            InterleavedLevelPlacement("L0", 0),
            InterleavedLevelPlacement("L1", 6),
            InterleavedLevelPlacement("L2", 12),
            InterleavedLevelPlacement("L3", 18),
            InterleavedLevelPlacement("L4", 24),
            InterleavedLevelPlacement("L5", 30),
            InterleavedLevelPlacement("L6", 36),
        )
    )
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    segment_steps: int = 25
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    sequence_pack_tokens: int = 2_049
    state_target_hidden_dim: int = 32
    synthetic_target_weight: float = 0.1

    def validate(self) -> "IRIS3BConfig":
        if self.schema != "iris.iris3b_config/v1":
            raise ValueError("schema must be iris.iris3b_config/v1.")
        if self.hidden_size <= 0 or self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_attention_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.ffn_hidden_size <= self.hidden_size:
            raise ValueError("ffn_hidden_size must be greater than hidden_size.")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        if self.vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256.")
        if self.sequence_pack_tokens <= 8:
            raise ValueError("sequence_pack_tokens must be > 8.")
        if len(self.level_schedule) != 7:
            raise ValueError("level_schedule must contain exactly 7 placements for L0-L6.")
        placements = {placement.level_id: placement.block_index for placement in self.level_schedule}
        if set(placements.keys()) != {f"L{index}" for index in range(7)}:
            raise ValueError("level_schedule must define each level exactly once.")
        if placements["L0"] != 0:
            raise ValueError("L0 must be placed at block_index 0.")
        if placements["L6"] != self.num_layers:
            raise ValueError("L6 must be placed at the final block boundary.")
        block_indices = [placement.block_index for placement in self.level_schedule]
        if sorted(block_indices) != block_indices:
            raise ValueError("level_schedule block indices must be non-decreasing.")
        if any(index < 0 or index > self.num_layers for index in block_indices):
            raise ValueError("level_schedule block indices must lie inside [0, num_layers].")
        return self

    @property
    def head_dim(self) -> int:
        return int(self.hidden_size // self.num_attention_heads)

    @property
    def sha256(self) -> str:
        payload = {
            "config_id": self.config_id,
            "profile_id": self.profile_id,
            "phase": self.phase,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "ffn_hidden_size": self.ffn_hidden_size,
            "max_sequence_length": self.max_sequence_length,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout_rate": self.dropout_rate,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_remat": self.use_remat,
            "dtype": self.dtype,
            "param_dtype": self.param_dtype,
            "aux_target_dim": self.aux_target_dim,
            "level_schedule": [
                {"level_id": placement.level_id, "block_index": placement.block_index}
                for placement in self.level_schedule
            ],
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "segment_steps": self.segment_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "sequence_pack_tokens": self.sequence_pack_tokens,
            "state_target_hidden_dim": self.state_target_hidden_dim,
            "synthetic_target_weight": self.synthetic_target_weight,
        }
        return stable_hash(payload)

    def estimated_parameter_count(self) -> int:
        embedding_params = self.vocab_size * self.hidden_size
        attention_params_per_block = (4 * self.hidden_size * self.hidden_size) + (4 * self.hidden_size)
        mlp_params_per_block = (
            3 * self.hidden_size * self.ffn_hidden_size
            + (2 * self.ffn_hidden_size)
            + self.hidden_size
        )
        norm_params_per_block = 2 * self.hidden_size
        decoder_params = self.num_layers * (
            attention_params_per_block + mlp_params_per_block + norm_params_per_block
        )
        level_head_params = len(self.level_schedule) * (
            self.hidden_size * self.aux_target_dim + self.aux_target_dim
        )
        final_norm_params = self.hidden_size
        lm_head_params = 0 if self.tie_word_embeddings else embedding_params
        return int(
            embedding_params
            + decoder_params
            + level_head_params
            + final_norm_params
            + lm_head_params
        )

    def to_payload(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "config_id": self.config_id,
            "profile_id": self.profile_id,
            "phase": self.phase,
            "vocab_size": int(self.vocab_size),
            "hidden_size": int(self.hidden_size),
            "num_layers": int(self.num_layers),
            "num_attention_heads": int(self.num_attention_heads),
            "ffn_hidden_size": int(self.ffn_hidden_size),
            "max_sequence_length": int(self.max_sequence_length),
            "rope_theta": float(self.rope_theta),
            "rms_norm_eps": float(self.rms_norm_eps),
            "dropout_rate": float(self.dropout_rate),
            "tie_word_embeddings": bool(self.tie_word_embeddings),
            "use_remat": bool(self.use_remat),
            "dtype": self.dtype,
            "param_dtype": self.param_dtype,
            "initializer_range": float(self.initializer_range),
            "aux_target_dim": int(self.aux_target_dim),
            "level_schedule": [
                {"level_id": placement.level_id, "block_index": int(placement.block_index)}
                for placement in self.level_schedule
            ],
            "micro_batch_size": int(self.micro_batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "segment_steps": int(self.segment_steps),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "warmup_steps": int(self.warmup_steps),
            "sequence_pack_tokens": int(self.sequence_pack_tokens),
            "state_target_hidden_dim": int(self.state_target_hidden_dim),
            "synthetic_target_weight": float(self.synthetic_target_weight),
            "estimated_parameter_count": int(self.estimated_parameter_count()),
            "config_sha256": self.sha256,
        }


def default_iris3b_config() -> IRIS3BConfig:
    return IRIS3BConfig().validate()


def kaggle_safe_iris3b_config(base: IRIS3BConfig | None = None) -> IRIS3BConfig:
    cfg = (base or default_iris3b_config()).validate()
    return replace(
        cfg,
        config_id="p1-iris3b-kaggle-safe-v1",
        max_sequence_length=1_024,
        sequence_pack_tokens=1_025,
    ).validate()


def kaggle_safer_iris3b_config(base: IRIS3BConfig | None = None) -> IRIS3BConfig:
    cfg = (base or default_iris3b_config()).validate()
    return replace(
        cfg,
        config_id="p1-iris3b-kaggle-safer-v1",
        max_sequence_length=768,
        sequence_pack_tokens=769,
    ).validate()


def kaggle_emergency_iris3b_config(base: IRIS3BConfig | None = None) -> IRIS3BConfig:
    cfg = (base or default_iris3b_config()).validate()
    return replace(
        cfg,
        config_id="p1-iris3b-kaggle-emergency-v1",
        max_sequence_length=512,
        sequence_pack_tokens=513,
        param_dtype="bfloat16",
    ).validate()


def kaggle_survival_iris3b_config(base: IRIS3BConfig | None = None) -> IRIS3BConfig:
    cfg = (base or default_iris3b_config()).validate()
    return replace(
        cfg,
        config_id="p1-iris3b-kaggle-survival-v1",
        max_sequence_length=384,
        sequence_pack_tokens=385,
        param_dtype="bfloat16",
    ).validate()


def cycle_memory_profile_candidates(
    *,
    memory_profile: str = "auto",
    kaggle_runtime: bool = False,
) -> Tuple[str, ...]:
    normalized = str(memory_profile or "auto").strip().lower() or "auto"
    # TEMPORARY TECHNICAL DEBT: keep auto-selected Kaggle P1 runs from failing
    # hard on first-step device OOM by walking a bounded shorter-context ladder.
    # Remove once the governed 3B P1 stack fits the default Kaggle path reliably
    # without deterministic retry. Intended replacement: a stable single-profile
    # cycle config selected without runtime fallback on Kaggle-class devices.
    if normalized == "auto" and kaggle_runtime:
        return ("kaggle_safe", "kaggle_safer", "kaggle_emergency", "kaggle_survival")
    return (normalized,)


def select_cycle_iris3b_config(
    *,
    explicit_config: IRIS3BConfig | None = None,
    memory_profile: str = "auto",
    kaggle_runtime: bool = False,
) -> IRIS3BConfig:
    base_config = explicit_config.validate() if explicit_config is not None else default_iris3b_config()
    normalized = str(memory_profile or "auto").strip().lower() or "auto"
    if normalized == "default":
        return base_config
    if normalized == "kaggle_safe":
        return kaggle_safe_iris3b_config(base_config)
    if normalized == "kaggle_safer":
        return kaggle_safer_iris3b_config(base_config)
    if normalized == "kaggle_emergency":
        return kaggle_emergency_iris3b_config(base_config)
    if normalized == "kaggle_survival":
        return kaggle_survival_iris3b_config(base_config)
    if normalized == "auto":
        if kaggle_runtime:
            return kaggle_safe_iris3b_config(base_config)
        return base_config
    raise ValueError(
        "memory_profile must be one of auto|default|kaggle_safe|kaggle_safer|kaggle_emergency|kaggle_survival."
    )


def iris3b_config_from_mapping(payload: Mapping[str, Any]) -> IRIS3BConfig:
    raw_schedule = payload.get("level_schedule", [])
    schedule = tuple(
        InterleavedLevelPlacement(
            level_id=str(item.get("level_id", "")).strip(),
            block_index=int(item.get("block_index", -1)),
        )
        for item in raw_schedule
        if isinstance(item, Mapping)
    )
    config = IRIS3BConfig(
        schema=str(payload.get("schema", "iris.iris3b_config/v1")),
        config_id=str(payload.get("config_id", "p1-iris3b-v1")),
        profile_id=str(payload.get("profile_id", "P1")),
        phase=str(payload.get("phase", "E")),
        vocab_size=int(payload.get("vocab_size", 50_176)),
        hidden_size=int(payload.get("hidden_size", 2_560)),
        num_layers=int(payload.get("num_layers", 36)),
        num_attention_heads=int(payload.get("num_attention_heads", 20)),
        ffn_hidden_size=int(payload.get("ffn_hidden_size", 6_912)),
        max_sequence_length=int(payload.get("max_sequence_length", 2_048)),
        rope_theta=float(payload.get("rope_theta", 10_000.0)),
        rms_norm_eps=float(payload.get("rms_norm_eps", 1.0e-6)),
        dropout_rate=float(payload.get("dropout_rate", 0.0)),
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
        use_remat=bool(payload.get("use_remat", True)),
        dtype=str(payload.get("dtype", "bfloat16")),
        param_dtype=str(payload.get("param_dtype", "float32")),
        initializer_range=float(payload.get("initializer_range", 0.02)),
        aux_target_dim=int(payload.get("aux_target_dim", 16)),
        level_schedule=schedule or IRIS3BConfig().level_schedule,
        micro_batch_size=int(payload.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 8)),
        segment_steps=int(payload.get("segment_steps", 25)),
        learning_rate=float(payload.get("learning_rate", 3.0e-4)),
        weight_decay=float(payload.get("weight_decay", 0.1)),
        warmup_steps=int(payload.get("warmup_steps", 100)),
        sequence_pack_tokens=int(payload.get("sequence_pack_tokens", 2_049)),
        state_target_hidden_dim=int(payload.get("state_target_hidden_dim", 32)),
        synthetic_target_weight=float(payload.get("synthetic_target_weight", 0.1)),
    )
    return config.validate()
