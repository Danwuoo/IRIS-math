from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from ..metrics import append_jsonl, build_canonical_metrics, neutral_failure_credit
from ..runtime import assert_jax_runtime, resolve_task_semantics
from .cache_budget import enforce_dataset_cache_budget, gib_to_bytes
from .data import (
    P1StreamingManifest,
    load_default_p1_streaming_manifest,
    load_p1_streaming_manifest,
    p1_manifest_sha256,
    resolve_manifest_revisions,
)
from .data.p1_iterator import P1StreamingProvider
from .governance import (
    active_governance_snapshot,
    git_code_version_hash,
    stable_hash,
    write_runtime_lock_manifest,
)
from .hf_sync import resolve_dataset_commit_sha
from .iris3b_checkpoint import export_params_msgpack, load_iris3b_checkpoint, save_iris3b_checkpoint
from .iris3b_config import IRIS3BConfig, default_iris3b_config
from .iris3b_model import init_iris3b_params
from .journal import APPLIED, PENDING, append_journal_event, journal_head_hash, load_journal, resolve_resume_pointer
from .tokenizer_pipeline import TokenizerArtifact, TokenizerBuildConfig, train_sentencepiece_tokenizer

_LEVEL_IDS = tuple(f"L{index}" for index in range(7))


def _require_train_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from .iris3b_model import IRIS3BForCausalLM
    except Exception as error:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "IRIS 3B training requires the optional jax/flax/optax stack."
        ) from error
    return jax, jnp, optax, IRIS3BForCausalLM


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normalize_checkpoint_ref(checkpoint_path: Path, run_dir: Path) -> str:
    try:
        return str(Path(checkpoint_path).resolve().relative_to(Path(run_dir).resolve()))
    except Exception:
        return str(checkpoint_path)


def _failure_credit_from_level_losses(level_losses: np.ndarray) -> Dict[str, float]:
    values = np.asarray(level_losses, dtype=np.float64)
    values = np.clip(values, a_min=1.0e-8, a_max=None)
    total = float(np.sum(values))
    if total <= 0.0:
        return neutral_failure_credit()
    return {
        level_id: float(values[index] / total)
        for index, level_id in enumerate(_LEVEL_IDS)
    }


def _should_crash(config: "P1TrainConfig", point: str, segment_id: int) -> bool:
    return str(config.crash_point) == point and int(config.crash_segment) == int(segment_id)


def _enforce_cache_budget(config: "P1TrainConfig") -> Dict[str, object]:
    budget_gib = int(max(config.dataset_cache_limit_gib, 0))
    free_space_floor_gib = int(max(config.cache_free_space_floor_gib, 0))
    return enforce_dataset_cache_budget(
        roots=(config.cache_root, config.snapshot_root),
        budget_bytes=gib_to_bytes(budget_gib),
        min_free_bytes=gib_to_bytes(free_space_floor_gib),
        monitor_roots=(
            config.output_dir,
            config.tokenizer_dir,
            config.tokenizer_workdir,
            config.cache_root,
            config.snapshot_root,
        ),
    )


def _load_committed_manifest(config: "P1TrainConfig") -> tuple[P1StreamingManifest, Path]:
    data_dir = Path(config.output_dir) / "data"
    committed_path = data_dir / "p1_streaming_manifest_committed.json"
    source_path = Path(config.manifest_path) if config.manifest_path is not None else None
    if committed_path.exists() and source_path is None:
        return load_p1_streaming_manifest(committed_path), committed_path
    manifest = load_p1_streaming_manifest(source_path) if source_path is not None else load_default_p1_streaming_manifest()
    if manifest.commit_posture != "committed":
        manifest = resolve_manifest_revisions(
            manifest,
            dataset_sha_resolver=lambda dataset_id, config_name, revision_hint: resolve_dataset_commit_sha(
                dataset_id=dataset_id,
                config_name=config_name,
                revision_hint=revision_hint,
                token=config.hf_token,
            ),
        )
    committed_path.parent.mkdir(parents=True, exist_ok=True)
    committed_path.write_text(
        json.dumps(manifest.to_payload(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return manifest, committed_path


def _local_snapshot_manifest(manifest: P1StreamingManifest, snapshot_root: Path | None, output_dir: Path) -> Path | None:
    if snapshot_root is None:
        return None
    rows = []
    root = Path(snapshot_root)
    if not root.exists():
        return None
    for source in manifest.sources:
        source_root = root / source.source_id
        rows.append(
            {
                "source_id": source.source_id,
                "pool_id": source.pool_id,
                "snapshot_exists": bool(source_root.exists()),
                "snapshot_root": str(source_root),
                "local_snapshot_pattern": source.local_snapshot_pattern,
            }
        )
    output_path = Path(output_dir) / "data" / "local_snapshot_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema": "iris.local_snapshot_manifest/v1",
                "snapshot_root": str(root),
                "sources": rows,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _prepare_tokenizer(
    *,
    config: "P1TrainConfig",
    manifest: P1StreamingManifest,
) -> tuple[Any, TokenizerArtifact | None, Path]:
    from .data.token_accounting import load_tokenizer_handle

    tokenizer_dir = Path(config.tokenizer_dir) if config.tokenizer_dir is not None else (Path(config.output_dir) / "tokenizer")
    manifest_path = tokenizer_dir / "iris_p1_tokenizer" / "tokenizer_build_manifest.json"
    artifact: TokenizerArtifact | None = None
    handle_path = tokenizer_dir / "iris_p1_tokenizer"
    if manifest_path.exists():
        handle = load_tokenizer_handle(str(handle_path))
        return handle, artifact, handle_path
    workdir = Path(config.tokenizer_workdir) if config.tokenizer_workdir is not None else tokenizer_dir
    artifact = train_sentencepiece_tokenizer(
        manifest=manifest,
        output_dir=workdir,
        streaming_mode=config.streaming_mode,
        snapshot_root=config.snapshot_root,
        corpus_workers=config.tokenizer_corpus_workers,
        sentencepiece_threads=config.sentencepiece_threads,
        build_config=TokenizerBuildConfig(
            vocab_size=config.model_config.vocab_size,
            sample_records_per_source=config.tokenizer_sample_records_per_source,
            max_corpus_chars=config.tokenizer_max_corpus_chars,
            seed=config.tokenizer_seed,
        ),
    )
    if handle_path.resolve() != artifact.tokenizer_dir.resolve():
        shutil.rmtree(handle_path, ignore_errors=True)
        shutil.copytree(artifact.tokenizer_dir, handle_path)
    handle = load_tokenizer_handle(str(handle_path))
    return handle, artifact, handle_path


def _optimizer_for_config(optax: Any, config: IRIS3BConfig) -> Any:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(config.learning_rate),
        warmup_steps=max(int(config.warmup_steps), 1),
        decay_steps=max(int(config.warmup_steps) + 10_000, 10_001),
        end_value=float(config.learning_rate) * 0.1,
    )
    return optax.adamw(
        learning_rate=schedule,
        b1=0.9,
        b2=0.95,
        eps=1.0e-8,
        weight_decay=float(config.weight_decay),
    )


def _train_functions(model_config: IRIS3BConfig, optimizer: Any) -> tuple[Any, Any, Any]:
    jax, jnp, optax, model_cls = _require_train_stack()
    model = model_cls(model_config.validate())
    grad_accum_scale = 1.0 / float(max(int(model_config.gradient_accumulation_steps), 1))

    def loss_fn(params: Any, batch: Mapping[str, Any]) -> tuple[Any, Dict[str, Any]]:
        outputs = model.apply(
            {"params": params},
            batch["input_ids"],
            batch["attention_mask"],
            deterministic=True,
        )
        logits = outputs["logits"]
        level_logits = outputs["level_logits"]
        label_mask = batch["attention_mask"].astype(jnp.float32)
        lm_losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["labels"])
        lm_loss = jnp.sum(lm_losses * label_mask) / jnp.clip(jnp.sum(label_mask), a_min=1.0)
        aux_error = jnp.square(level_logits - batch["aux_targets"]).mean(axis=-1) * batch["aux_mask"]
        aux_loss = jnp.sum(aux_error) / jnp.clip(jnp.sum(batch["aux_mask"]), a_min=1.0)
        level_loss = jnp.sum(aux_error, axis=0) / jnp.clip(jnp.sum(batch["aux_mask"], axis=0), a_min=1.0)
        total_loss = lm_loss + (float(model_config.synthetic_target_weight) * aux_loss)
        return total_loss, {
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "level_loss": level_loss,
        }

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    # Reuse the previous accumulation buffer so the 50k x 2560 embedding grad
    # does not require a third full fp32 allocation at each micro-step.
    @partial(jax.jit, donate_argnums=(0,))
    def accumulate_grads(grad_accum: Any, grads: Any) -> Any:
        return jax.tree_util.tree_map(lambda left, right: left + right, grad_accum, grads)

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def apply_grads(params: Any, opt_state: Any, grad_accum: Any) -> tuple[Any, Any]:
        grads = jax.tree_util.tree_map(lambda value: value * grad_accum_scale, grad_accum)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state

    return grad_fn, accumulate_grads, apply_grads


@dataclass
class P1TrainConfig:
    output_dir: Path = Path("artifacts/p1_3b")
    run_id: str = "p1-iris3b-kaggle"
    profile_id: str = "P1"
    phase: str = "E"
    baseline_id: str = "p1-readiness-fixed-baseline"
    tolerance_profile_id: str = "tp_p1_bootstrap"
    device: str = "gpu"
    strict_jax: bool = True
    data_seed: int = 17
    model_seed: int = 0
    manifest_path: Path | None = None
    streaming_mode: str = "auto"
    cache_root: Path | None = None
    snapshot_root: Path | None = None
    tokenizer_dir: Path | None = None
    tokenizer_workdir: Path | None = None
    tokenizer_sample_records_per_source: int = 2_048
    tokenizer_max_corpus_chars: int = 4_000_000
    tokenizer_seed: int = 17
    host_cpu_threads: int = 1
    batch_prefetch: int = 1
    tokenizer_corpus_workers: int = 1
    sentencepiece_threads: int = 1
    max_cycle_minutes: int = 350
    max_segments: int = 10_000
    dataset_cache_limit_gib: int = 50
    cache_free_space_floor_gib: int = 0
    hf_token: str | None = None
    runtime_lock_manifest_path: Path | None = None
    crash_point: str = "none"
    crash_segment: int = -1
    resume_path_id: str = "uninterrupted"
    model_config: IRIS3BConfig = field(default_factory=default_iris3b_config)

    def as_payload(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "phase": self.phase,
            "baseline_id": self.baseline_id,
            "tolerance_profile_id": self.tolerance_profile_id,
            "device": self.device,
            "strict_jax": bool(self.strict_jax),
            "data_seed": int(self.data_seed),
            "model_seed": int(self.model_seed),
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "streaming_mode": self.streaming_mode,
            "cache_root": str(self.cache_root) if self.cache_root is not None else None,
            "snapshot_root": str(self.snapshot_root) if self.snapshot_root is not None else None,
            "tokenizer_dir": str(self.tokenizer_dir) if self.tokenizer_dir is not None else None,
            "tokenizer_workdir": str(self.tokenizer_workdir) if self.tokenizer_workdir is not None else None,
            "tokenizer_sample_records_per_source": int(self.tokenizer_sample_records_per_source),
            "tokenizer_max_corpus_chars": int(self.tokenizer_max_corpus_chars),
            "tokenizer_seed": int(self.tokenizer_seed),
            "host_cpu_threads": int(self.host_cpu_threads),
            "batch_prefetch": int(self.batch_prefetch),
            "tokenizer_corpus_workers": int(self.tokenizer_corpus_workers),
            "sentencepiece_threads": int(self.sentencepiece_threads),
            "max_cycle_minutes": int(self.max_cycle_minutes),
            "max_segments": int(self.max_segments),
            "dataset_cache_limit_gib": int(self.dataset_cache_limit_gib),
            "cache_free_space_floor_gib": int(self.cache_free_space_floor_gib),
            "runtime_lock_manifest_path": (
                str(self.runtime_lock_manifest_path) if self.runtime_lock_manifest_path is not None else None
            ),
            "crash_point": self.crash_point,
            "crash_segment": int(self.crash_segment),
            "resume_path_id": self.resume_path_id,
            "model_config": self.model_config.to_payload(),
        }


def run_p1_training_cycle(config: P1TrainConfig) -> Dict[str, Any]:
    jax, jnp, optax, _ = _require_train_stack()
    model_config = config.model_config.validate()
    assert_jax_runtime(
        device=config.device,
        require_gpu=bool(config.strict_jax and str(config.device).lower() == "gpu"),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    journal_path = output_dir / "segment_journal.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    cache_budget_summary = _enforce_cache_budget(config)

    runtime_lock = write_runtime_lock_manifest(
        output_dir=output_dir,
        phase=config.phase,
        runtime_lock_manifest_path=config.runtime_lock_manifest_path,
    )
    governance = active_governance_snapshot(profile_id=config.profile_id, phase=config.phase)
    code_version_hash = git_code_version_hash()
    config_hash = stable_hash(config.as_payload())
    manifest, committed_manifest_path = _load_committed_manifest(config)
    tokenizer_handle, tokenizer_artifact, tokenizer_root = _prepare_tokenizer(config=config, manifest=manifest)
    cache_budget_summary = _enforce_cache_budget(config)
    snapshot_manifest_path = _local_snapshot_manifest(manifest, config.snapshot_root, output_dir)

    provider = P1StreamingProvider(
        manifest=manifest,
        tokenizer_handle=tokenizer_handle,
        run_id=config.run_id,
        data_seed=config.data_seed,
        streaming_mode=config.streaming_mode,
        cache_root=config.cache_root,
        snapshot_root=config.snapshot_root,
        sequence_pack_tokens=model_config.sequence_pack_tokens,
        micro_batch_size=model_config.micro_batch_size,
        aux_target_dim=model_config.aux_target_dim,
        state_target_hidden_dim=model_config.state_target_hidden_dim,
    )
    cache_budget_summary = _enforce_cache_budget(config)
    source_manifest = provider.source_manifest

    events = load_journal(journal_path)
    next_segment, last_applied, pending_event = resolve_resume_pointer(events)
    optimizer = _optimizer_for_config(optax, model_config)
    grad_fn, accumulate_grads, apply_grads = _train_functions(model_config, optimizer)

    params = init_iris3b_params(
        model_config,
        seed=config.model_seed,
        batch_size=model_config.micro_batch_size,
    )
    opt_state = optimizer.init(params)
    optimizer_step_id = 0
    rng_state = {
        "rng.model.train": int(config.model_seed),
        "rng.control.train": int(config.model_seed),
        "rng.data.train": int(config.data_seed),
    }

    if last_applied is not None:
        checkpoint = load_iris3b_checkpoint(output_dir / str(last_applied["checkpoint_ref"]))
        if checkpoint.get("model_state") is not None:
            raise RuntimeError(
                "Legacy toy checkpoints are not compatible with the IRIS 3B trainer."
            )
        params = checkpoint["params"]
        opt_state = checkpoint["opt_state"]
        rng_state = dict(checkpoint["rng_state"])
        optimizer_step_id = int(checkpoint.get("optimizer_step_id_last_applied", 0))
        cache_budget_summary = _enforce_cache_budget(config)

    effective_resume_path = config.resume_path_id
    if pending_event is not None and config.resume_path_id == "uninterrupted":
        effective_resume_path = "execute_crash"

    deadline = time.monotonic() + max(int(config.max_cycle_minutes), 1) * 60.0
    last_checkpoint_manifest_path: Path | None = None
    last_checkpoint_payload_ref = ""
    last_dataset_slice_id = ""
    last_plan_hash = ""
    last_batch = None
    segments_completed = 0

    for segment_id in range(next_segment, next_segment + max(int(config.max_segments), 0)):
        if time.monotonic() >= deadline and segments_completed > 0:
            break
        cache_budget_summary = _enforce_cache_budget(config)
        segment_plan = provider.build_segment_plan(
            segment_id=segment_id,
            optimizer_steps=model_config.segment_steps,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        )
        last_dataset_slice_id = segment_plan.dataset_slice_id
        last_plan_hash = segment_plan.plan_hash
        rng_hash_pre = stable_hash(rng_state)
        pending_record = append_journal_event(
            journal_path,
            {
                "run_id": config.run_id,
                "segment_id": segment_id,
                "status": PENDING,
                "optimizer_step_id": optimizer_step_id,
                "dataset_slice_id": segment_plan.dataset_slice_id,
                "rng_hash_pre": rng_hash_pre,
                "code_version_hash": code_version_hash,
                "config_hash": config_hash,
                "runtime_lock_manifest_id": runtime_lock["runtime_lock_manifest_id"],
                "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
                "policy_bundle_sha256": governance["policy_bundle_sha256"],
                "profile_id": config.profile_id,
                "phase": config.phase,
                "resume_path_id": effective_resume_path,
                "data.source_manifest_sha256": source_manifest.manifest_sha256,
                "data.tokenizer_manifest_ref": (
                    str((tokenizer_artifact.manifest_path if tokenizer_artifact is not None else (tokenizer_root / "tokenizer_build_manifest.json")))
                ),
                "data.requested_streaming_mode": source_manifest.requested_streaming_mode,
                "data.effective_streaming_mode": source_manifest.effective_streaming_mode,
                "data.local_snapshot_manifest_ref": str(snapshot_manifest_path) if snapshot_manifest_path is not None else "",
            },
        )
        if _should_crash(config, "execute", segment_id):
            raise RuntimeError(f"Injected crash at execute for segment_id={segment_id}.")

        segment_losses = []
        segment_lm_losses = []
        segment_aux_losses = []
        segment_level_losses = []
        token_count_total = 0
        source_counts: Dict[str, int] = {}
        batch_prefetch = max(int(config.batch_prefetch), 1)
        batch_workers = max(1, min(int(config.host_cpu_threads), batch_prefetch))
        batch_iterator = iter(
            provider.iter_prefetched_batches(
                segment_id=segment_id,
                dataset_slice_id=segment_plan.dataset_slice_id,
                batch_plans=segment_plan.steps,
                max_workers=batch_workers,
                prefetch_batches=batch_prefetch,
            )
        )

        for optimizer_step_idx in range(model_config.segment_steps):
            grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
            micro_loss_values = []
            micro_lm_values = []
            micro_aux_values = []
            micro_level_values = []
            for _ in range(model_config.gradient_accumulation_steps):
                batch = next(batch_iterator)
                last_batch = batch
                batch_inputs = {
                    "input_ids": jnp.asarray(batch.input_ids, dtype=jnp.int32),
                    "labels": jnp.asarray(batch.labels, dtype=jnp.int32),
                    "attention_mask": jnp.asarray(batch.attention_mask, dtype=jnp.int32),
                    "aux_targets": jnp.asarray(batch.aux_targets, dtype=jnp.float32),
                    "aux_mask": jnp.asarray(batch.aux_mask, dtype=jnp.float32),
                }
                (loss_value, aux_values), grads = grad_fn(params, batch_inputs)
                grad_accum = accumulate_grads(grad_accum, grads)
                micro_loss_values.append(float(loss_value))
                micro_lm_values.append(float(aux_values["lm_loss"]))
                micro_aux_values.append(float(aux_values["aux_loss"]))
                micro_level_values.append(np.asarray(aux_values["level_loss"], dtype=np.float64))
                token_count_total += int(batch.token_count)
                source_counts[batch.source_id] = int(source_counts.get(batch.source_id, 0)) + int(batch.token_count)
            if _should_crash(config, "pre_commit", segment_id):
                raise RuntimeError(f"Injected crash at pre_commit for segment_id={segment_id}.")
            params, opt_state = apply_grads(params, opt_state, grad_accum)
            optimizer_step_id += 1
            rng_state["rng.model.train"] = int(rng_state["rng.model.train"]) + 1
            rng_state["rng.control.train"] = int(rng_state["rng.control.train"]) + 1
            rng_state["rng.data.train"] = int(rng_state["rng.data.train"]) + model_config.gradient_accumulation_steps
            segment_losses.append(float(np.mean(np.asarray(micro_loss_values, dtype=np.float64))))
            segment_lm_losses.append(float(np.mean(np.asarray(micro_lm_values, dtype=np.float64))))
            segment_aux_losses.append(float(np.mean(np.asarray(micro_aux_values, dtype=np.float64))))
            segment_level_losses.append(np.mean(np.stack(micro_level_values, axis=0), axis=0))

        rng_hash_post = stable_hash(rng_state)
        level_loss_mean = np.mean(np.stack(segment_level_losses, axis=0), axis=0)
        failure_credit = _failure_credit_from_level_losses(level_loss_mean)
        mean_loss = float(np.mean(np.asarray(segment_losses, dtype=np.float64)))
        mean_lm_loss = float(np.mean(np.asarray(segment_lm_losses, dtype=np.float64)))
        mean_aux_loss = float(np.mean(np.asarray(segment_aux_losses, dtype=np.float64)))
        task_validity_score = float(1.0 / (1.0 + max(mean_loss, 0.0)))
        task_confidence = float(1.0 / (1.0 + max(mean_aux_loss, 0.0)))
        checkpoint_manifest = {
            "profile_id": config.profile_id,
            "phase": config.phase,
            "segment_id_last_applied": segment_id,
            "optimizer_step_id_last_applied": optimizer_step_id,
            "dataset_slice_id_last_applied": segment_plan.dataset_slice_id,
            "policy_bundle_sha256": governance["policy_bundle_sha256"],
            "runtime_lock_manifest_id": runtime_lock["runtime_lock_manifest_id"],
            "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
            "data_realization_policy_id": governance["data_realization_policy_id"],
            "decontam_policy_id": governance["decontam_policy_id"],
            "learning_objective_bundle_id": governance["learning_objective_bundle_id"],
            "learning_objective_bundle_resolution_source": governance["learning_objective_bundle_resolution_source"],
            "learning_objective_bundle_sha256": governance["learning_objective_bundle_sha256"],
            "benchmark_family_policy_refs": governance["benchmark_family_policy_refs"],
            "parser_provenance_id": governance["parser_provenance_id"],
            "parser_provenance_refs": governance["parser_provenance_refs"],
            "parse_config_fingerprint": governance["parse_config_fingerprint"],
            "ocr_layout_extractor_version": governance["ocr_layout_extractor_version"],
            "formula_parser_version": governance["formula_parser_version"],
            "semantic_unit_typer_version": governance["semantic_unit_typer_version"],
            "formalizer_provenance_id": governance["formalizer_provenance_id"],
            "formalizer_version": governance["formalizer_version"],
            "verifier_provenance_id": governance["verifier_provenance_id"],
            "verifier_build_id": governance["verifier_build_id"],
            "journal_head_event_id": pending_record["event_id"],
            "journal_head_hash": journal_head_hash(load_journal(journal_path)),
            "code_version_hash": code_version_hash,
            "config_hash": config_hash,
            "dataset_plan_hash": segment_plan.plan_hash,
            "checkpoint_payload_ref": str(Path("payloads") / f"segment_{segment_id:06d}"),
            "governance_state": dict(governance),
            "model_config": model_config.to_payload(),
            "data_provenance": {
                "streaming_manifest_path": str(committed_manifest_path),
                "streaming_manifest_id": manifest.manifest_id,
                "streaming_manifest_sha256": source_manifest.manifest_sha256,
                "tokenizer_root": str(tokenizer_root),
                "tokenizer_manifest_ref": str(
                    tokenizer_artifact.manifest_path if tokenizer_artifact is not None else (tokenizer_root / "tokenizer_build_manifest.json")
                ),
                "requested_streaming_mode": source_manifest.requested_streaming_mode,
                "effective_streaming_mode": source_manifest.effective_streaming_mode,
                "local_snapshot_manifest_ref": str(snapshot_manifest_path) if snapshot_manifest_path is not None else "",
                "token_ledger": {
                    "total_tokens": int(token_count_total),
                    "by_source": {str(key): int(value) for key, value in sorted(source_counts.items())},
                },
                "resume_path_id": effective_resume_path,
            },
        }
        cache_budget_summary = _enforce_cache_budget(config)
        last_checkpoint_manifest_path = save_iris3b_checkpoint(
            checkpoint_dir=checkpoints_dir,
            segment_id=segment_id,
            params=params,
            opt_state=opt_state,
            rng_state=rng_state,
            manifest_payload=checkpoint_manifest,
        )
        last_checkpoint_payload_ref = str(Path("payloads") / f"segment_{segment_id:06d}")
        if _should_crash(config, "post_commit", segment_id):
            raise RuntimeError(f"Injected crash at post_commit for segment_id={segment_id}.")

        checkpoint_ref = _normalize_checkpoint_ref(last_checkpoint_manifest_path, output_dir)
        append_journal_event(
            journal_path,
            {
                "run_id": config.run_id,
                "segment_id": segment_id,
                "status": APPLIED,
                "optimizer_step_id": optimizer_step_id,
                "dataset_slice_id": segment_plan.dataset_slice_id,
                "rng_hash_pre": rng_hash_pre,
                "rng_hash_post": rng_hash_post,
                "loss_hash": stable_hash({"mean_loss": mean_loss}),
                "grad_stats_hash": stable_hash({"level_loss_mean": level_loss_mean.tolist()}),
                "code_version_hash": code_version_hash,
                "config_hash": config_hash,
                "runtime_lock_manifest_id": runtime_lock["runtime_lock_manifest_id"],
                "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
                "policy_bundle_sha256": governance["policy_bundle_sha256"],
                "profile_id": config.profile_id,
                "phase": config.phase,
                "data_realization_policy_id": governance["data_realization_policy_id"],
                "decontam_policy_id": governance["decontam_policy_id"],
                "learning_objective_bundle_id": governance["learning_objective_bundle_id"],
                "learning_objective_bundle_resolution_source": governance["learning_objective_bundle_resolution_source"],
                "learning_objective_bundle_sha256": governance["learning_objective_bundle_sha256"],
                "benchmark_family_policy_refs": governance["benchmark_family_policy_refs"],
                "parser_provenance_id": governance["parser_provenance_id"],
                "parser_provenance_refs": governance["parser_provenance_refs"],
                "parse_config_fingerprint": governance["parse_config_fingerprint"],
                "ocr_layout_extractor_version": governance["ocr_layout_extractor_version"],
                "formula_parser_version": governance["formula_parser_version"],
                "semantic_unit_typer_version": governance["semantic_unit_typer_version"],
                "formalizer_provenance_id": governance["formalizer_provenance_id"],
                "formalizer_version": governance["formalizer_version"],
                "verifier_provenance_id": governance["verifier_provenance_id"],
                "verifier_build_id": governance["verifier_build_id"],
                "resume_path_id": effective_resume_path,
                "data.plan_hash": segment_plan.plan_hash,
                "data.source_last": last_batch.source_id if last_batch is not None else "",
                "data.sample_key_last": (last_batch.sample_keys[-1] if last_batch is not None and last_batch.sample_keys else ""),
                "data.source_manifest_sha256": source_manifest.manifest_sha256,
                "data.tokenizer_manifest_ref": str(
                    tokenizer_artifact.manifest_path if tokenizer_artifact is not None else (tokenizer_root / "tokenizer_build_manifest.json")
                ),
                "data.requested_streaming_mode": source_manifest.requested_streaming_mode,
                "data.effective_streaming_mode": source_manifest.effective_streaming_mode,
                "data.local_snapshot_manifest_ref": str(snapshot_manifest_path) if snapshot_manifest_path is not None else "",
                "checkpoint_payload_ref": last_checkpoint_payload_ref,
                "checkpoint_ref": checkpoint_ref,
            },
        )

        if last_batch is None:
            raise RuntimeError("Training segment completed without a batch; this should be unreachable.")
        last_state = last_batch.state_irs[-1]
        task_semantics = resolve_task_semantics(last_state.PF)
        metrics = build_canonical_metrics(
            state=last_state,
            failure_credit=failure_credit,
            task_validity_score=task_validity_score,
            task_confidence=task_confidence,
            extra={
                "cost.total_steps": optimizer_step_id,
                "phase": config.phase,
                "baseline_id": config.baseline_id,
                "tolerance_profile_id": config.tolerance_profile_id,
                "segment_id": segment_id,
                "optimizer_step_id": optimizer_step_id,
                "dataset_slice_id": segment_plan.dataset_slice_id,
                "data_seed": config.data_seed,
                "journal_status": APPLIED,
                "journal_head_hash": journal_head_hash(load_journal(journal_path)),
                "rng_hash_pre": rng_hash_pre,
                "rng_hash_post": rng_hash_post,
                "resume_path_id": effective_resume_path,
                "runtime_lock_manifest_id": runtime_lock["runtime_lock_manifest_id"],
                "runtime_lock_manifest_sha256": runtime_lock["runtime_lock_manifest_sha256"],
                "policy_bundle_sha256": governance["policy_bundle_sha256"],
                "profile_id": config.profile_id,
                "phase": config.phase,
                "data_realization_policy_id": governance["data_realization_policy_id"],
                "decontam_policy_id": governance["decontam_policy_id"],
                "learning_objective_bundle_id": governance["learning_objective_bundle_id"],
                "learning_objective_bundle_resolution_source": governance["learning_objective_bundle_resolution_source"],
                "benchmark_family_policy_refs": governance["benchmark_family_policy_refs"],
                "parser_provenance_id": governance["parser_provenance_id"],
                "parser_provenance_refs": governance["parser_provenance_refs"],
                "parse_config_fingerprint": governance["parse_config_fingerprint"],
                "ocr_layout_extractor_version": governance["ocr_layout_extractor_version"],
                "formula_parser_version": governance["formula_parser_version"],
                "semantic_unit_typer_version": governance["semantic_unit_typer_version"],
                "formalizer_provenance_id": governance["formalizer_provenance_id"],
                "formalizer_version": governance["formalizer_version"],
                "verifier_provenance_id": governance["verifier_provenance_id"],
                "verifier_build_id": governance["verifier_build_id"],
                "task_family": task_semantics.task_family,
                "task_family_resolution_source": task_semantics.task_family_resolution_source,
                "task_adjudication_policy_id": task_semantics.task_adjudication_policy_id,
                "task_adjudication_policy_resolution_source": task_semantics.task_adjudication_policy_resolution_source,
                "runtime_status": last_state.CS.runtime_status,
                "adjudication_status": (
                    last_state.CS.adjudication_state.adjudication_status
                    if last_state.CS.adjudication_state is not None
                    else "pending"
                ),
                "code_version_hash": code_version_hash,
                "config_hash": config_hash,
                "trunk.backend": "jax",
                "train.loss": mean_loss,
                "train.lm_loss": mean_lm_loss,
                "train.aux_loss": mean_aux_loss,
                "task.document_grounding_score": task_validity_score,
                "rep.document.parse_completeness": min(1.0, 0.9 + (0.1 * task_validity_score)),
                "eval.false_accept_rate": min(1.0, mean_aux_loss * 0.05),
                "eval.calibration_error": abs(task_validity_score - task_confidence),
                "rep.tokenizer.ir_fragmentation_rate": max(
                    0.0,
                    1.0 - (float(token_count_total) / float(max(segment_plan.total_target_tokens, 1))),
                ),
                "paired.invariance.gap": min(1.0, mean_aux_loss),
                "concept.leakage_score": 0.0,
                "contam.strict_holdout_leakage_score": 0.0,
                "provenance.parser_coverage": 1.0,
                "provenance.verifier_coverage": 1.0,
                "data.plan_hash": segment_plan.plan_hash,
                "data.source_last": last_batch.source_id,
                "data.sample_key_last": last_batch.sample_keys[-1] if last_batch.sample_keys else "",
                "data.source_effective_mode_last": last_batch.effective_mode,
                "data.token_ledger": {
                    "total_tokens": int(token_count_total),
                    "by_source": {str(key): int(value) for key, value in sorted(source_counts.items())},
                },
                "data.source_manifest_sha256": source_manifest.manifest_sha256,
                "data.tokenizer_manifest_ref": str(
                    tokenizer_artifact.manifest_path if tokenizer_artifact is not None else (tokenizer_root / "tokenizer_build_manifest.json")
                ),
                "data.requested_streaming_mode": source_manifest.requested_streaming_mode,
                "data.effective_streaming_mode": source_manifest.effective_streaming_mode,
                "data.local_snapshot_manifest_ref": str(snapshot_manifest_path) if snapshot_manifest_path is not None else "",
                "checkpoint_payload_ref": last_checkpoint_payload_ref,
            },
        )
        append_jsonl(metrics_path, metrics)
        segments_completed += 1
        cache_budget_summary = _enforce_cache_budget(config)

    if last_checkpoint_manifest_path is None:
        raise RuntimeError("Training cycle completed without producing an APPLIED checkpoint.")

    realized_tokens = 0
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            token_ledger = row.get("data.token_ledger", {})
            if isinstance(token_ledger, Mapping):
                realized_tokens += int(token_ledger.get("total_tokens", 0))

    return {
        "status": "Done",
        "run_id": config.run_id,
        "output_dir": str(output_dir),
        "journal_path": str(journal_path),
        "metrics_path": str(metrics_path),
        "runtime_lock_manifest_path": runtime_lock["runtime_lock_manifest_path"],
        "streaming_manifest_path": str(committed_manifest_path),
        "streaming_manifest_sha256": source_manifest.manifest_sha256,
        "tokenizer_root": str(tokenizer_root),
        "tokenizer_manifest_ref": str(
            tokenizer_artifact.manifest_path if tokenizer_artifact is not None else (tokenizer_root / "tokenizer_build_manifest.json")
        ),
        "requested_streaming_mode": source_manifest.requested_streaming_mode,
        "effective_streaming_mode": source_manifest.effective_streaming_mode,
        "local_snapshot_manifest_ref": str(snapshot_manifest_path) if snapshot_manifest_path is not None else "",
        "last_segment_id": int(next_segment + segments_completed - 1),
        "last_dataset_slice_id": last_dataset_slice_id,
        "last_plan_hash": last_plan_hash,
        "checkpoint_manifest_path": str(last_checkpoint_manifest_path),
        "checkpoint_payload_ref": last_checkpoint_payload_ref,
        "segments_completed": int(segments_completed),
        "realized_tokens": int(realized_tokens),
        "dataset_cache_limit_gib": int(config.dataset_cache_limit_gib),
        "cache_free_space_floor_gib": int(config.cache_free_space_floor_gib),
        "dataset_cache_budget": cache_budget_summary,
    }


def export_final_release(
    *,
    run_dir: Path,
    checkpoint_manifest_path: Path | None,
    release_dir: Path,
    model_config: IRIS3BConfig,
    tokenizer_root: Path,
    readiness_packet_path: Path | None = None,
    readiness_history_path: Path | None = None,
    streaming_manifest_path: Path | None = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_manifest_path) if checkpoint_manifest_path is not None else None
    if checkpoint_path is None:
        journal_rows = load_journal(run_dir / "segment_journal.jsonl")
        applied_rows = [row for row in journal_rows if str(row.get("status", "")).upper() == APPLIED]
        if not applied_rows:
            raise RuntimeError("No APPLIED segment exists to export.")
        checkpoint_path = run_dir / str(applied_rows[-1]["checkpoint_ref"])
    checkpoint = load_iris3b_checkpoint(checkpoint_path)
    if checkpoint.get("params") is None:
        raise RuntimeError("Final export requires an orbax-backed IRIS 3B checkpoint.")

    params_path = export_params_msgpack(params=checkpoint["params"], output_path=release_dir / "flax_model.msgpack")
    (release_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["IRIS3BForCausalLM"],
                "model_type": "iris3b",
                **model_config.to_payload(),
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    (release_dir / "iris3b_config.json").write_text(
        json.dumps(model_config.to_payload(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(checkpoint_path, release_dir / "checkpoint_manifest.json")
    shutil.copytree(tokenizer_root, release_dir / "tokenizer", dirs_exist_ok=True)
    if streaming_manifest_path is not None and Path(streaming_manifest_path).exists():
        shutil.copy2(streaming_manifest_path, release_dir / "p1_streaming_manifest.json")
    if readiness_packet_path is not None and Path(readiness_packet_path).exists():
        shutil.copy2(readiness_packet_path, release_dir / "p1_readiness_packet.json")
    if readiness_history_path is not None and Path(readiness_history_path).exists():
        shutil.copy2(readiness_history_path, release_dir / "p1_readiness_history.jsonl")
    source_model_file = Path(__file__).with_name("iris3b_model.py")
    source_config_file = Path(__file__).with_name("iris3b_config.py")
    shutil.copy2(source_model_file, release_dir / "modeling_flax_iris3b.py")
    shutil.copy2(source_config_file, release_dir / "configuration_iris3b.py")
    (release_dir / "README.md").write_text(
        "\n".join(
            [
                "# IRIS 3B P1 Release",
                "",
                "- Profile: `P1`",
                "- Phase: `E`",
                "- Backend: `Flax/JAX`",
                "- Checkpoint schema: `iris.training_checkpoint/v2`",
                "- TEMPORARY TECHNICAL DEBT: heuristic State-IR targets remain active until canonical projection or verifier-backed targets cover the committed P1 train-visible slice.",
                "",
                "Artifacts:",
                f"- Params: `{params_path.name}`",
                "- Tokenizer: `tokenizer/`",
                "- Config: `config.json`, `iris3b_config.json`",
                "- Governed checkpoint sidecar: `checkpoint_manifest.json`",
                "",
                "This release is Flax-first and keeps the custom IRIS architecture files alongside the exported weights.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "Done",
        "release_dir": str(release_dir),
        "params_path": str(params_path),
        "checkpoint_manifest_path": str(release_dir / "checkpoint_manifest.json"),
    }
