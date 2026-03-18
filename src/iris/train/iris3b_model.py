from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping


def _require_jax_stack() -> tuple[Any, Any, Any]:
    try:
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
    except Exception as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "IRIS 3B Flax model requires the optional jax/flax runtime."
        ) from error
    return nn, jax, jnp


from .iris3b_config import IRIS3BConfig


def dtype_from_name(name: str) -> Any:
    _, _, jnp = _require_jax_stack()
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if normalized in {"fp16", "float16"}:
        return jnp.float16
    if normalized in {"fp32", "float32"}:
        return jnp.float32
    raise ValueError(f"Unsupported dtype name: {name!r}")


def _rotate_half(x: Any) -> Any:
    _, _, jnp = _require_jax_stack()
    half = x.shape[-1] // 2
    left = x[..., :half]
    right = x[..., half:]
    return jnp.concatenate((-right, left), axis=-1)


def _apply_rope(x: Any, positions: Any, theta: float) -> Any:
    _, _, jnp = _require_jax_stack()
    head_dim = int(x.shape[-1])
    inv_freq = 1.0 / (float(theta) ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)))
    sinusoid = positions[..., None].astype(jnp.float32) * inv_freq[None, None, :]
    sin = jnp.repeat(jnp.sin(sinusoid), repeats=2, axis=-1)
    cos = jnp.repeat(jnp.cos(sinusoid), repeats=2, axis=-1)
    sin = sin[:, :, None, :]
    cos = cos[:, :, None, :]
    return (x * cos) + (_rotate_half(x) * sin)


def _mean_pool(hidden_states: Any, attention_mask: Any | None) -> Any:
    _, _, jnp = _require_jax_stack()
    if attention_mask is None:
        return jnp.mean(hidden_states, axis=1)
    weights = attention_mask.astype(jnp.float32)
    denom = jnp.clip(jnp.sum(weights, axis=1, keepdims=True), a_min=1.0)
    return jnp.sum(hidden_states * weights[..., None], axis=1) / denom


def _build_modules() -> Dict[str, Any]:
    nn, jax, jnp = _require_jax_stack()

    class _RMSNorm(nn.Module):
        hidden_size: int
        eps: float
        dtype: Any
        param_dtype: Any

        @nn.compact
        def __call__(self, x: Any) -> Any:
            scale = self.param("scale", nn.initializers.ones, (self.hidden_size,), self.param_dtype)
            variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            normalized = x * jax.lax.rsqrt(variance + self.eps)
            return (normalized * scale.astype(jnp.float32)).astype(self.dtype)

    class _SelfAttention(nn.Module):
        config: IRIS3BConfig

        @nn.compact
        def __call__(self, hidden_states: Any, attention_mask: Any | None) -> Any:
            cfg = self.config
            dtype = dtype_from_name(cfg.dtype)
            param_dtype = dtype_from_name(cfg.param_dtype)
            batch_size, sequence_length, _ = hidden_states.shape
            qkv = nn.Dense(
                cfg.hidden_size * 3,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name="qkv",
            )(hidden_states)
            qkv = qkv.reshape(batch_size, sequence_length, 3, cfg.num_attention_heads, cfg.head_dim)
            query = qkv[:, :, 0]
            key = qkv[:, :, 1]
            value = qkv[:, :, 2]
            positions = jnp.arange(sequence_length, dtype=jnp.float32)[None, :].repeat(batch_size, axis=0)
            query = _apply_rope(query, positions, cfg.rope_theta)
            key = _apply_rope(key, positions, cfg.rope_theta)

            scores = jnp.einsum("bthd,bshd->bhts", query, key).astype(jnp.float32)
            scores = scores / jnp.sqrt(float(cfg.head_dim))
            causal_mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))[None, None, :, :]
            if attention_mask is not None:
                key_mask = attention_mask.astype(bool)[:, None, None, :]
                query_mask = attention_mask.astype(bool)[:, None, :, None]
                combined_mask = causal_mask & key_mask & query_mask
            else:
                combined_mask = causal_mask
            scores = jnp.where(combined_mask, scores, jnp.full_like(scores, -1.0e9))
            weights = jax.nn.softmax(scores, axis=-1).astype(dtype)
            attn_output = jnp.einsum("bhts,bshd->bthd", weights, value).reshape(
                batch_size,
                sequence_length,
                cfg.hidden_size,
            )
            return nn.Dense(
                cfg.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name="out_proj",
            )(attn_output)

    class _SwiGLUMLP(nn.Module):
        config: IRIS3BConfig

        @nn.compact
        def __call__(self, hidden_states: Any) -> Any:
            cfg = self.config
            dtype = dtype_from_name(cfg.dtype)
            param_dtype = dtype_from_name(cfg.param_dtype)
            gate = nn.Dense(
                cfg.ffn_hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name="gate_proj",
            )(hidden_states)
            up = nn.Dense(
                cfg.ffn_hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name="up_proj",
            )(hidden_states)
            activated = nn.silu(gate) * up
            return nn.Dense(
                cfg.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name="down_proj",
            )(activated)

    class _DecoderBlock(nn.Module):
        config: IRIS3BConfig

        @nn.compact
        def __call__(self, hidden_states: Any, attention_mask: Any | None) -> Any:
            cfg = self.config
            dtype = dtype_from_name(cfg.dtype)
            hidden_states = hidden_states + _SelfAttention(cfg, name="self_attn")(
                _RMSNorm(
                    hidden_size=cfg.hidden_size,
                    eps=cfg.rms_norm_eps,
                    dtype=dtype,
                    param_dtype=dtype_from_name(cfg.param_dtype),
                    name="input_norm",
                )(hidden_states),
                attention_mask,
            )
            hidden_states = hidden_states + _SwiGLUMLP(cfg, name="mlp")(
                _RMSNorm(
                    hidden_size=cfg.hidden_size,
                    eps=cfg.rms_norm_eps,
                    dtype=dtype,
                    param_dtype=dtype_from_name(cfg.param_dtype),
                    name="post_attn_norm",
                )(hidden_states)
            )
            return hidden_states

    class _LevelHead(nn.Module):
        config: IRIS3BConfig
        level_id: str

        @nn.compact
        def __call__(self, hidden_states: Any, attention_mask: Any | None) -> Any:
            cfg = self.config
            dtype = dtype_from_name(cfg.dtype)
            param_dtype = dtype_from_name(cfg.param_dtype)
            pooled = _mean_pool(hidden_states, attention_mask)
            hidden = nn.Dense(
                cfg.state_target_hidden_dim,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name=f"{self.level_id.lower()}_dense",
            )(pooled)
            hidden = nn.tanh(hidden)
            return nn.Dense(
                cfg.aux_target_dim,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(cfg.initializer_range),
                name=f"{self.level_id.lower()}_out",
            )(hidden)

    class _IRIS3BForCausalLM(nn.Module):
        config: IRIS3BConfig

        @nn.compact
        def __call__(
            self,
            input_ids: Any,
            attention_mask: Any | None = None,
            *,
            deterministic: bool = True,
        ) -> Mapping[str, Any]:
            del deterministic
            cfg = self.config.validate()
            dtype = dtype_from_name(cfg.dtype)
            param_dtype = dtype_from_name(cfg.param_dtype)
            embeddings = nn.Embed(
                num_embeddings=cfg.vocab_size,
                features=cfg.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=nn.initializers.normal(cfg.initializer_range),
                name="token_embeddings",
            )
            hidden_states = embeddings(input_ids)
            schedule_map: Dict[int, list[str]] = {}
            for placement in cfg.level_schedule:
                schedule_map.setdefault(int(placement.block_index), []).append(str(placement.level_id))
            level_outputs = []
            if 0 in schedule_map:
                for level_id in schedule_map[0]:
                    level_outputs.append(
                        _LevelHead(cfg, level_id, name=f"{level_id.lower()}_head")(
                            hidden_states,
                            attention_mask,
                        )
                    )
            block_cls = _DecoderBlock
            if cfg.use_remat:
                block_cls = nn.remat(_DecoderBlock)
            for layer_index in range(cfg.num_layers):
                hidden_states = block_cls(cfg, name=f"layers_{layer_index}")(
                    hidden_states,
                    attention_mask,
                )
                boundary = layer_index + 1
                if boundary in schedule_map:
                    for level_id in schedule_map[boundary]:
                        level_outputs.append(
                            _LevelHead(
                                cfg,
                                level_id,
                                name=f"{level_id.lower()}_head",
                            )(hidden_states, attention_mask)
                        )
            hidden_states = _RMSNorm(
                hidden_size=cfg.hidden_size,
                eps=cfg.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                name="final_norm",
            )(hidden_states)
            if cfg.tie_word_embeddings:
                logits = embeddings.attend(hidden_states)
            else:
                logits = nn.Dense(
                    cfg.vocab_size,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=nn.initializers.normal(cfg.initializer_range),
                    name="lm_head",
                )(hidden_states)
            level_logits = jnp.stack(level_outputs, axis=1).astype(jnp.float32)
            return {
                "logits": logits.astype(jnp.float32),
                "level_logits": level_logits,
                "hidden_states": hidden_states,
            }

    return {
        "RMSNorm": _RMSNorm,
        "SelfAttention": _SelfAttention,
        "SwiGLUMLP": _SwiGLUMLP,
        "DecoderBlock": _DecoderBlock,
        "LevelHead": _LevelHead,
        "IRIS3BForCausalLM": _IRIS3BForCausalLM,
    }


def __getattr__(name: str) -> Any:
    if name not in {
        "RMSNorm",
        "SelfAttention",
        "SwiGLUMLP",
        "DecoderBlock",
        "LevelHead",
        "IRIS3BForCausalLM",
    }:
        raise AttributeError(name)
    modules = _build_modules()
    value = modules[name]
    globals()[name] = value
    return value


def init_iris3b_params(
    config: IRIS3BConfig,
    *,
    seed: int = 0,
    batch_size: int = 1,
) -> Mapping[str, Any]:
    _, jax, jnp = _require_jax_stack()
    model_cls = __getattr__("IRIS3BForCausalLM")
    model = model_cls(config.validate())
    dummy_input_ids = jnp.zeros((int(batch_size), int(config.max_sequence_length)), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones_like(dummy_input_ids)
    variables = model.init(
        jax.random.PRNGKey(int(seed)),
        dummy_input_ids,
        dummy_attention_mask,
        deterministic=True,
    )
    return variables["params"]


def level_schedule_map(config: IRIS3BConfig) -> Dict[str, int]:
    return {
        str(placement.level_id): int(placement.block_index)
        for placement in config.validate().level_schedule
    }


def small_test_config(base: IRIS3BConfig | None = None) -> IRIS3BConfig:
    cfg = (base or IRIS3BConfig()).validate()
    return replace(
        cfg,
        config_id="p1-iris3b-test",
        vocab_size=512,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        ffn_hidden_size=256,
        max_sequence_length=32,
        aux_target_dim=8,
        state_target_hidden_dim=16,
        sequence_pack_tokens=33,
        dtype="float32",
        param_dtype="float32",
        use_remat=False,
        level_schedule=tuple(
            type(cfg.level_schedule[0])(level_id=f"L{index}", block_index=position)
            for index, position in enumerate((0, 1, 2, 3, 4, 4, 4))
        ),
        segment_steps=2,
        gradient_accumulation_steps=2,
    ).validate()
