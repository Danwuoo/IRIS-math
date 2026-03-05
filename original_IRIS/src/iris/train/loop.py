from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization as flax_serialization

from ..levels import LEVEL_IDS, apply_level_stack_params, init_level_stack_params
from ..metrics import append_jsonl, build_canonical_metrics, neutral_failure_credit
from ..runtime import assert_jax_runtime
from ..schema import STATE_IR_TOKEN_ORDER
from ..trunk import build_typed_sequence, forward_with_params, init_trunk_params
from .checkpoint import load_checkpoint, save_checkpoint_atomic
from .data.contracts import load_default_pure_lm_profile, load_profile
from .data.iterator import PureLMStreamingProvider, deterministic_sampling_key
from .data.planner import build_hybrid_schedule
from .data.state_builder import text_to_state_ir
from .data.token_accounting import (
    TokenLedger,
    TokenizerError,
    load_tokenizer_handle,
    validate_tokenizer_required,
)
from .journal import APPLIED, PENDING, append_journal_event, journal_head_hash, load_journal, resolve_resume_pointer
from .synthetic import dataset_slice_id_for_segment, generate_synthetic_state


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _git_code_version_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _build_runtime_lock_manifest(phase: str) -> Dict[str, Any]:
    env_keys = [
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DISABLE_JIT",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ]
    return {
        "schema": "iris.runtime_lock_manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "phase": phase,
        "host": {
            "os": platform.platform(),
            "kernel": platform.release(),
            "gpu": "unknown",
            "nvidia_driver": "unknown",
            "cuda_runtime": "unknown",
            "cudnn": "unknown",
        },
        "python": {
            "version": sys.version.split(" ")[0],
            "packages": [
                {"name": "jax", "version": _safe_version("jax"), "hash": "n/a"},
                {"name": "jaxlib", "version": _safe_version("jaxlib"), "hash": "n/a"},
                {"name": "flax", "version": _safe_version("flax"), "hash": "n/a"},
                {"name": "optax", "version": _safe_version("optax"), "hash": "n/a"},
                {"name": "orbax-checkpoint", "version": _safe_version("orbax-checkpoint"), "hash": "n/a"},
                {"name": "numpy", "version": _safe_version("numpy"), "hash": "n/a"},
            ],
        },
        "jax": {
            "jax": _safe_version("jax"),
            "jaxlib": _safe_version("jaxlib"),
            "jaxlib_build": "unknown",
            "xla_flags": str(os.environ.get("XLA_FLAGS", "")),
            "env": {key: str(os.environ.get(key, "")) for key in env_keys},
        },
    }


def _load_pinned_runtime_lock_manifest(path: Path, phase: str) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Pinned runtime lock manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Pinned runtime lock manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pinned runtime lock manifest must be a JSON object: {path}")
    if payload.get("schema") != "iris.runtime_lock_manifest/v1":
        raise RuntimeError(
            "Pinned runtime lock manifest schema mismatch. "
            "Expected iris.runtime_lock_manifest/v1."
        )
    required_fields = ("schema", "created_at", "phase", "host", "python", "jax")
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise RuntimeError(
            "Pinned runtime lock manifest missing required fields: "
            + ", ".join(missing_fields)
        )
    manifest_phase = str(payload.get("phase", "")).strip()
    if manifest_phase and manifest_phase != str(phase):
        raise RuntimeError(
            f"Pinned runtime lock manifest phase mismatch. Expected '{phase}', got '{manifest_phase}'."
        )
    return payload


def _write_runtime_lock_manifest(
    output_dir: Path,
    phase: str,
    runtime_lock_manifest_path: Optional[Path] = None,
) -> Dict[str, str]:
    if runtime_lock_manifest_path is None:
        manifest = _build_runtime_lock_manifest(phase=phase)
    else:
        manifest = _load_pinned_runtime_lock_manifest(Path(runtime_lock_manifest_path), phase=phase)
    manifest_path = output_dir / "runtime_lock_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    manifest_id = manifest_sha[:12]
    return {
        "runtime_lock_manifest_id": manifest_id,
        "runtime_lock_manifest_sha256": manifest_sha,
        "runtime_lock_manifest_path": str(manifest_path),
    }


def _should_crash(config: "ToyTrainConfig", point: str, segment_id: int) -> bool:
    return config.crash_point == point and config.crash_segment == segment_id


def _initial_optimizer_state() -> Dict[str, Any]:
    return {"step": 0, "last_grad_norm": 0.0, "last_loss": 0.0}


def _initial_rng_state() -> Dict[str, Any]:
    return {"rng.model.train": 0, "rng.control.train": 0, "rng.data.train": 0}


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)


def _tree_to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.asarray(np.asarray(value, dtype=np.float32)), tree)


def _serialize_opt_state(opt_state: Any) -> Any:
    state_dict = flax_serialization.to_state_dict(opt_state)
    return jax.tree_util.tree_map(lambda value: np.asarray(value), state_dict)


def _restore_opt_state(serialized_state: Any, template_opt_state: Any) -> Any:
    return flax_serialization.from_state_dict(template_opt_state, serialized_state)


def _grad_l2_norm(grad_tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(grad_tree)
    if not leaves:
        return 0.0
    squared_sum = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        squared_sum = squared_sum + jnp.sum(jnp.square(leaf))
    return float(jnp.sqrt(squared_sum))


def _state_section_lengths_tuple(state: Any) -> tuple[int, ...]:
    lengths = state.section_lengths()
    return tuple(int(lengths[token_type]) for token_type in STATE_IR_TOKEN_ORDER)


def _loss_and_aux(
    model_params: Dict[str, Any],
    state: Any,
    *,
    level_alpha: float,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    base_sequence = jnp.asarray(state.to_canonical_sequence(), dtype=jnp.float32)
    level_sequence, level_controls, l6_credit = apply_level_stack_params(
        model_params["levels"],
        base_sequence,
        alpha=level_alpha,
    )
    typed_sequence = build_typed_sequence(
        sequence=level_sequence,
        section_lengths=_state_section_lengths_tuple(state),
        type_embeddings=model_params["trunk"]["type_embeddings"],
    )
    pred_sequence, trunk_control = forward_with_params(model_params["trunk"], typed_sequence)
    target = jnp.tanh(0.5 * typed_sequence)
    loss_main = jnp.mean(jnp.square(pred_sequence - target))
    all_controls = [trunk_control] + [level_controls[level_id] for level_id in LEVEL_IDS]
    loss_ctrl = 1e-4 * jnp.mean(jnp.square(jnp.concatenate(all_controls, axis=0)))
    loss = loss_main + loss_ctrl
    return loss, {
        "pred_sequence": pred_sequence,
        "trunk_control": trunk_control,
        "l6_credit": l6_credit,
    }


@dataclass
class ToyTrainConfig:
    output_dir: Path = Path("artifacts/toy_train")
    run_id: str = "toy-run"
    segments: int = 2
    micro_steps: int = 4
    hidden_dim: int = 16
    learning_rate: float = 0.05
    data_seed: int = 17
    device: str = "cpu"
    phase: str = "C"
    baseline_id: str = "toy-baseline"
    tolerance_profile_id: str = "toy-default"
    backend: str = "jax"
    strict_jax: bool = True
    level_impl: str = "mounted"
    level_alpha: float = 0.1
    crash_point: str = "none"
    crash_segment: int = -1
    resume_path_id: str = "uninterrupted"
    runtime_lock_manifest_path: Optional[Path] = None
    data_source: str = "synthetic"
    pure_lm_profile: Optional[Path] = None
    tokenizer_id_or_path: Optional[str] = None
    streaming_mode: str = "auto"
    snapshot_root: Optional[Path] = None
    tokens_per_micro_step: int = 128
    hybrid_pure_ratio: float = 0.9

    def as_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["output_dir"] = str(self.output_dir)
        if payload.get("runtime_lock_manifest_path") is not None:
            payload["runtime_lock_manifest_path"] = str(payload["runtime_lock_manifest_path"])
        if payload.get("pure_lm_profile") is not None:
            payload["pure_lm_profile"] = str(payload["pure_lm_profile"])
        if payload.get("snapshot_root") is not None:
            payload["snapshot_root"] = str(payload["snapshot_root"])
        return payload


def run_toy_training(config: ToyTrainConfig) -> Dict[str, Any]:
    if config.backend != "jax":
        raise RuntimeError("Strict JAX mode requires --backend jax.")
    if config.level_impl != "mounted":
        raise RuntimeError("Phase C.1 main path requires --level-impl mounted.")
    data_source = str(config.data_source).strip().lower()
    if data_source not in {"synthetic", "pure_lm_streaming", "hybrid_mixture"}:
        raise RuntimeError(
            "data_source must be one of synthetic|pure_lm_streaming|hybrid_mixture."
        )
    try:
        validate_tokenizer_required(data_source, config.tokenizer_id_or_path)
    except TokenizerError as error:
        raise RuntimeError(str(error)) from error

    assert_jax_runtime(
        device=config.device,
        require_gpu=bool(config.strict_jax and str(config.device).lower() == "gpu"),
    )

    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    journal_path = output_dir / "segment_journal.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    config_hash = _stable_hash(config.as_dict())
    code_version_hash = _git_code_version_hash()
    runtime_lock = _write_runtime_lock_manifest(
        output_dir=output_dir,
        phase=config.phase,
        runtime_lock_manifest_path=config.runtime_lock_manifest_path,
    )
    segment_micro_steps = max(int(config.micro_steps), 1)
    tokens_per_micro_step = max(int(config.tokens_per_micro_step), 1)

    pretrain_provider = None
    tokenizer_handle = None
    data_profile_id = ""
    data_sources_manifest_sha256 = ""
    data_tokenizer_fingerprint = ""
    data_streaming_mode_effective = "synthetic"
    data_profile_hash = ""
    token_ledger = TokenLedger()

    if data_source in {"pure_lm_streaming", "hybrid_mixture"}:
        try:
            profile = (
                load_profile(Path(config.pure_lm_profile))
                if config.pure_lm_profile is not None
                else load_default_pure_lm_profile()
            )
        except Exception as error:
            raise RuntimeError(f"Failed to load Pure LM profile: {error}") from error
        try:
            tokenizer_handle = load_tokenizer_handle(str(config.tokenizer_id_or_path))
        except TokenizerError as error:
            raise RuntimeError(str(error)) from error
        pretrain_provider = PureLMStreamingProvider(
            profile=profile,
            tokenizer_handle=tokenizer_handle,
            run_id=config.run_id,
            data_seed=config.data_seed,
            streaming_mode=config.streaming_mode,
            snapshot_root=config.snapshot_root,
        )
        manifest = pretrain_provider.sources_manifest
        data_profile_id = manifest.profile_id
        data_sources_manifest_sha256 = manifest.sources_manifest_sha256
        data_tokenizer_fingerprint = tokenizer_handle.fingerprint
        data_streaming_mode_effective = manifest.effective_mode
        data_profile_hash = _stable_hash(profile.stable_payload())

    events = load_journal(journal_path)
    next_segment, last_applied, pending_event = resolve_resume_pointer(events)

    model_params = {
        "trunk": init_trunk_params(hidden_dim=config.hidden_dim, seed=0),
        "levels": init_level_stack_params(hidden_dim=config.hidden_dim, seed=0),
    }
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(model_params)
    optimizer_state = _initial_optimizer_state()
    rng_state = _initial_rng_state()

    if last_applied is not None:
        checkpoint_payload = load_checkpoint(Path(last_applied["checkpoint_ref"]))
        model_payload = checkpoint_payload["model_state"]
        if model_payload.get("schema") != "iris.model_state/v2":
            raise RuntimeError(
                "Unsupported checkpoint model schema. Expected iris.model_state/v2."
            )
        model_params = {
            "trunk": _tree_to_jax(model_payload["trunk"]),
            "levels": _tree_to_jax(model_payload["levels"]),
        }
        optimizer_state = dict(checkpoint_payload["optimizer_state"])
        opt_state = _restore_opt_state(optimizer_state["opt_state"], opt_state)
        rng_state = dict(checkpoint_payload["rng_state"])

    effective_resume_path = config.resume_path_id
    if pending_event is not None and config.resume_path_id == "uninterrupted":
        effective_resume_path = "execute_crash"

    start_segment = next_segment
    end_segment = start_segment + max(int(config.segments), 0)
    for segment_id in range(start_segment, end_segment):
        pure_segment_plan = None
        pure_step_plan_by_idx = {}
        if pretrain_provider is not None:
            pure_segment_plan = pretrain_provider.build_segment_plan(
                segment_id=segment_id,
                micro_steps=segment_micro_steps,
                tokens_per_micro_step=tokens_per_micro_step,
            )
            dataset_slice_id = pure_segment_plan.dataset_slice_id
            data_plan_hash = pure_segment_plan.plan_hash
            pure_step_plan_by_idx = {
                step.micro_step_idx: step for step in pure_segment_plan.steps
            }
        else:
            dataset_slice_id = dataset_slice_id_for_segment(segment_id)
            data_plan_hash = _stable_hash(
                {
                    "data_source": data_source,
                    "segment_id": segment_id,
                    "micro_steps": segment_micro_steps,
                    "data_seed": config.data_seed,
                }
            )

        hybrid_schedule = None
        if data_source == "hybrid_mixture":
            hybrid_schedule = build_hybrid_schedule(
                segment_id=segment_id,
                micro_steps=segment_micro_steps,
                pure_ratio=float(config.hybrid_pure_ratio),
                data_seed=config.data_seed,
            )

        rng_hash_pre = _stable_hash(rng_state)
        pending_record = append_journal_event(
            journal_path,
            {
                "run_id": config.run_id,
                "segment_id": segment_id,
                "status": PENDING,
                "optimizer_step_id": int(optimizer_state["step"]) + 1,
                "dataset_slice_id": dataset_slice_id,
                "rng_hash_pre": rng_hash_pre,
                "rng_hash_post": None,
                "loss_hash": "pending",
                "grad_stats_hash": "pending",
                "code_version_hash": code_version_hash,
                "config_hash": config_hash,
                "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
                "data.profile_id": data_profile_id,
                "data.sources_manifest_sha256": data_sources_manifest_sha256,
                "data.tokenizer_fingerprint": data_tokenizer_fingerprint,
                "data.streaming_mode_effective": data_streaming_mode_effective,
                "data.plan_hash": data_plan_hash,
                "data_source": data_source,
                "checkpoint_ref": None,
            },
        )

        losses = []
        grad_accum = jax.tree_util.tree_map(jnp.zeros_like, model_params)
        last_state = None
        last_data_source_id = "synthetic_ir_aligned"
        last_sampling_key = deterministic_sampling_key(
            run_id=config.run_id,
            dataset_slice_id=dataset_slice_id,
            segment_id=segment_id,
            micro_step_idx=0,
            data_seed=config.data_seed,
        )
        last_data_source_effective_mode = (
            data_streaming_mode_effective if data_streaming_mode_effective else "synthetic"
        )
        for micro_step_idx in range(segment_micro_steps):
            if _should_crash(config, "execute", segment_id) and micro_step_idx == 0:
                raise RuntimeError(
                    f"Injected crash at execute for segment_id={segment_id} (resume to replay)."
                )
            if data_source == "synthetic":
                state = generate_synthetic_state(
                    run_id=config.run_id,
                    dataset_slice_id=dataset_slice_id,
                    segment_id=segment_id,
                    micro_step_idx=micro_step_idx,
                    hidden_dim=config.hidden_dim,
                    data_seed=config.data_seed,
                )
                token_ledger.add("synthetic_ir_aligned", tokens_per_micro_step)
                last_data_source_id = "synthetic_ir_aligned"
                last_sampling_key = deterministic_sampling_key(
                    run_id=config.run_id,
                    dataset_slice_id=dataset_slice_id,
                    segment_id=segment_id,
                    micro_step_idx=micro_step_idx,
                    data_seed=config.data_seed,
                )
                last_data_source_effective_mode = "synthetic"
            elif data_source == "pure_lm_streaming":
                if pretrain_provider is None or tokenizer_handle is None:
                    raise RuntimeError("Pure LM streaming provider is not initialized.")
                micro_plan = pure_step_plan_by_idx.get(micro_step_idx)
                if micro_plan is None:
                    raise RuntimeError(
                        f"Missing micro-step plan for segment={segment_id}, micro_step_idx={micro_step_idx}."
                    )
                text_batch = pretrain_provider.sample_micro_step_text(
                    segment_id=segment_id,
                    dataset_slice_id=dataset_slice_id,
                    micro_step_plan=micro_plan,
                )
                state = text_to_state_ir(
                    text=text_batch.text,
                    tokenizer=tokenizer_handle.tokenizer,
                    hidden_dim=config.hidden_dim,
                )
                token_ledger.add(text_batch.source_id, text_batch.token_count)
                last_data_source_id = text_batch.source_id
                last_sampling_key = text_batch.sampling_key
                last_data_source_effective_mode = text_batch.effective_mode
            else:
                if hybrid_schedule is None:
                    raise RuntimeError("Hybrid schedule was not initialized.")
                use_pure_stream = hybrid_schedule.pure_step_flags[micro_step_idx]
                if use_pure_stream:
                    if pretrain_provider is None or tokenizer_handle is None:
                        raise RuntimeError("Pure LM streaming provider is not initialized.")
                    micro_plan = pure_step_plan_by_idx.get(micro_step_idx)
                    if micro_plan is None:
                        raise RuntimeError(
                            f"Missing micro-step plan for segment={segment_id}, micro_step_idx={micro_step_idx}."
                        )
                    text_batch = pretrain_provider.sample_micro_step_text(
                        segment_id=segment_id,
                        dataset_slice_id=dataset_slice_id,
                        micro_step_plan=micro_plan,
                    )
                    state = text_to_state_ir(
                        text=text_batch.text,
                        tokenizer=tokenizer_handle.tokenizer,
                        hidden_dim=config.hidden_dim,
                    )
                    token_ledger.add(text_batch.source_id, text_batch.token_count)
                    last_data_source_id = text_batch.source_id
                    last_sampling_key = text_batch.sampling_key
                    last_data_source_effective_mode = text_batch.effective_mode
                else:
                    state = generate_synthetic_state(
                        run_id=config.run_id,
                        dataset_slice_id=f"{dataset_slice_id}-synthetic",
                        segment_id=segment_id,
                        micro_step_idx=micro_step_idx,
                        hidden_dim=config.hidden_dim,
                        data_seed=config.data_seed,
                    )
                    token_ledger.add("synthetic_ir_aligned", tokens_per_micro_step)
                    last_data_source_id = "synthetic_ir_aligned"
                    last_sampling_key = deterministic_sampling_key(
                        run_id=config.run_id,
                        dataset_slice_id=dataset_slice_id,
                        segment_id=segment_id,
                        micro_step_idx=micro_step_idx,
                        data_seed=config.data_seed,
                    )
                    last_data_source_effective_mode = "synthetic"
            (loss, _), grads = jax.value_and_grad(_loss_and_aux, has_aux=True)(
                model_params,
                state,
                level_alpha=float(config.level_alpha),
            )
            losses.append(float(loss))
            grad_accum = jax.tree_util.tree_map(lambda acc, grad: acc + grad, grad_accum, grads)
            last_state = state

        mean_loss = float(np.mean(np.asarray(losses, dtype=np.float64)))
        micro_step_count = float(segment_micro_steps)
        mean_grads = jax.tree_util.tree_map(lambda grad: grad / micro_step_count, grad_accum)

        if _should_crash(config, "pre_commit", segment_id):
            raise RuntimeError(
                f"Injected crash at pre_commit for segment_id={segment_id} (resume to replay)."
            )

        updates, next_opt_state = optimizer.update(mean_grads, opt_state, params=model_params)
        next_model_params = optax.apply_updates(model_params, updates)
        mean_grad_norm = _grad_l2_norm(mean_grads)
        model_params = next_model_params
        opt_state = next_opt_state
        optimizer_state["step"] = int(optimizer_state["step"]) + 1
        optimizer_state["last_grad_norm"] = mean_grad_norm
        optimizer_state["last_loss"] = mean_loss
        optimizer_state["opt_state"] = _serialize_opt_state(opt_state)
        rng_state["rng.model.train"] = int(rng_state["rng.model.train"]) + 1
        rng_state["rng.control.train"] = int(rng_state["rng.control.train"]) + 1
        rng_state["rng.data.train"] = int(rng_state["rng.data.train"]) + 1
        rng_hash_post = _stable_hash(rng_state)

        current_events = load_journal(journal_path)
        checkpoint_payload = {
            "model_state": {
                "schema": "iris.model_state/v2",
                "backend": "jax",
                "hidden_dim": int(config.hidden_dim),
                "trunk": _tree_to_numpy(model_params["trunk"]),
                "levels": _tree_to_numpy(model_params["levels"]),
            },
            "optimizer_state": optimizer_state,
            "rng_state": rng_state,
            "behavior_state": {"macro_entries": 0, "memory_entries": 0},
            "segment_id_last_applied": segment_id,
            "optimizer_step_id_last_applied": optimizer_state["step"],
            "dataset_slice_id_last_applied": dataset_slice_id,
            "runtime_lock_manifest_id": runtime_lock["runtime_lock_manifest_id"],
            "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
            "journal_head_event_id": pending_record["event_id"],
            "journal_head_hash": journal_head_hash(current_events),
            "code_version_hash": code_version_hash,
            "config_hash": config_hash,
            "dataset_plan_hash": data_plan_hash,
            "data_provenance": {
                "data_source": data_source,
                "profile_id": data_profile_id,
                "profile_hash": data_profile_hash,
                "sources_manifest_sha256": data_sources_manifest_sha256,
                "tokenizer_fingerprint": data_tokenizer_fingerprint,
                "streaming_mode_effective": data_streaming_mode_effective,
                "token_ledger": token_ledger.as_dict(),
            },
        }
        checkpoint_ref = save_checkpoint_atomic(
            checkpoint_dir=checkpoints_dir,
            segment_id=segment_id,
            payload=checkpoint_payload,
        )

        if _should_crash(config, "post_commit", segment_id):
            raise RuntimeError(
                f"Injected crash at post_commit for segment_id={segment_id} (resume to replay)."
            )

        append_journal_event(
            journal_path,
            {
                "run_id": config.run_id,
                "segment_id": segment_id,
                "status": APPLIED,
                "optimizer_step_id": optimizer_state["step"],
                "dataset_slice_id": dataset_slice_id,
                "rng_hash_pre": rng_hash_pre,
                "rng_hash_post": rng_hash_post,
                "loss_hash": _stable_hash({"mean_loss": mean_loss}),
                "grad_stats_hash": _stable_hash({"mean_grad_norm": mean_grad_norm}),
                "code_version_hash": code_version_hash,
                "config_hash": config_hash,
                "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
                "data.profile_id": data_profile_id,
                "data.sources_manifest_sha256": data_sources_manifest_sha256,
                "data.tokenizer_fingerprint": data_tokenizer_fingerprint,
                "data.streaming_mode_effective": data_streaming_mode_effective,
                "data.plan_hash": data_plan_hash,
                "data.source_last": last_data_source_id,
                "data.sample_key_last": last_sampling_key,
                "data_source": data_source,
                "checkpoint_ref": str(checkpoint_ref),
            },
        )

        if last_state is not None:
            base_sequence = jnp.asarray(last_state.to_canonical_sequence(), dtype=jnp.float32)
            level_sequence, _, l6_credit_arr = apply_level_stack_params(
                model_params["levels"],
                base_sequence,
                alpha=float(config.level_alpha),
            )
            typed_sequence = build_typed_sequence(
                sequence=level_sequence,
                section_lengths=_state_section_lengths_tuple(last_state),
                type_embeddings=model_params["trunk"]["type_embeddings"],
            )
            pred_sequence, _ = forward_with_params(model_params["trunk"], typed_sequence)
            final_sequence = np.asarray(pred_sequence, dtype=np.float32)
            state_out = last_state.with_updated_sequence(final_sequence)
            l6_credit = {
                level_id: float(np.asarray(l6_credit_arr[index]))
                for index, level_id in enumerate(LEVEL_IDS)
            }
            metrics = build_canonical_metrics(
                state=state_out,
                failure_credit=l6_credit or neutral_failure_credit(),
                task_validity_score=max(0.0, 1.0 - mean_loss),
                task_confidence=max(0.0, 1.0 / (1.0 + mean_grad_norm)),
                extra={
                    "cost.total_steps": optimizer_state["step"],
                    "cost.program_proposals": 0,
                    "cost.rollout_steps": 0,
                    "cost.retrieval_calls": 0,
                    "phase": config.phase,
                    "baseline_id": config.baseline_id,
                    "tolerance_profile_id": config.tolerance_profile_id,
                    "segment_id": segment_id,
                    "optimizer_step_id": optimizer_state["step"],
                    "dataset_slice_id": dataset_slice_id,
                    "data_seed": config.data_seed,
                    "journal_status": APPLIED,
                    "journal_head_hash": journal_head_hash(load_journal(journal_path)),
                    "rng_hash_pre": rng_hash_pre,
                    "rng_hash_post": rng_hash_post,
                    "resume_path_id": effective_resume_path,
                    "runtime_lock_manifest_id": runtime_lock["runtime_lock_manifest_id"],
                    "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
                    "code_version_hash": code_version_hash,
                    "config_hash": config_hash,
                    "trunk.backend": "jax",
                    "data_source": data_source,
                    "data.profile_id": data_profile_id,
                    "data.profile_hash": data_profile_hash,
                    "data.sources_manifest_sha256": data_sources_manifest_sha256,
                    "data.tokenizer_fingerprint": data_tokenizer_fingerprint,
                    "data.streaming_mode_effective": data_streaming_mode_effective,
                    "data.plan_hash": data_plan_hash,
                    "data.source_last": last_data_source_id,
                    "data.sample_key_last": last_sampling_key,
                    "data.source_effective_mode_last": last_data_source_effective_mode,
                    "data.token_ledger": token_ledger.as_dict(),
                },
            )
            append_jsonl(metrics_path, metrics)

    final_events = load_journal(journal_path)
    return {
        "run_id": config.run_id,
        "status": "Done",
        "start_segment": start_segment,
        "end_segment_exclusive": end_segment,
        "last_event_status": final_events[-1]["status"] if final_events else "NONE",
        "journal_path": str(journal_path),
        "metrics_path": str(metrics_path),
        "checkpoints_dir": str(checkpoints_dir),
        "data_source": data_source,
        "data_profile_id": data_profile_id,
        "data_sources_manifest_sha256": data_sources_manifest_sha256,
        "data_tokenizer_fingerprint": data_tokenizer_fingerprint,
        "data_streaming_mode_effective": data_streaming_mode_effective,
    }
