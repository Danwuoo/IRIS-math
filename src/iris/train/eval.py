from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ..levels import LEVEL_IDS, apply_level_stack_params
from ..metrics import build_canonical_metrics, neutral_failure_credit
from ..runtime import assert_jax_runtime
from ..schema import STATE_IR_TOKEN_ORDER
from ..trunk import build_typed_sequence, forward_with_params
from .checkpoint import load_checkpoint
from .journal import last_applied_event, load_journal
from .synthetic import generate_synthetic_state


def _load_last_metrics_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _tree_to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.asarray(np.asarray(value, dtype=np.float32)), tree)


def _state_section_lengths_tuple(state: Any) -> tuple[int, ...]:
    lengths = state.section_lengths()
    return tuple(int(lengths[token_type]) for token_type in STATE_IR_TOKEN_ORDER)


def evaluate_latest_run(
    output_dir: Path,
    data_seed: int = 17,
    device: str = "cpu",
    strict_jax: bool = True,
) -> Dict[str, Any]:
    assert_jax_runtime(
        device=device,
        require_gpu=bool(strict_jax and str(device).lower() == "gpu"),
    )
    output_dir = Path(output_dir)
    journal_path = output_dir / "segment_journal.jsonl"
    events = load_journal(journal_path)
    applied_event = last_applied_event(events)
    if applied_event is None:
        raise RuntimeError("No APPLIED segment found. Run training first.")

    checkpoint = load_checkpoint(Path(applied_event["checkpoint_ref"]))
    model_state = checkpoint["model_state"]
    if model_state.get("schema") != "iris.model_state/v2":
        raise RuntimeError("Unsupported checkpoint model schema. Expected iris.model_state/v2.")
    hidden_dim = int(model_state["hidden_dim"])
    data_provenance = dict(checkpoint.get("data_provenance", {}))
    last_metrics = _load_last_metrics_record(output_dir / "metrics.jsonl")
    resolved_data_seed = int(last_metrics.get("data_seed", data_seed))
    phase = str(last_metrics.get("phase", "C"))
    model_params = {
        "trunk": _tree_to_jax(model_state["trunk"]),
        "levels": _tree_to_jax(model_state["levels"]),
    }

    segment_id = int(applied_event["segment_id"])
    dataset_slice_id = str(applied_event["dataset_slice_id"])
    run_id = str(applied_event["run_id"])
    state = generate_synthetic_state(
        run_id=run_id,
        dataset_slice_id=dataset_slice_id,
        segment_id=segment_id,
        micro_step_idx=0,
        hidden_dim=hidden_dim,
        data_seed=resolved_data_seed,
    )
    base_sequence = jnp.asarray(state.to_canonical_sequence(), dtype=jnp.float32)
    level_sequence, _, l6_credit = apply_level_stack_params(model_params["levels"], base_sequence, alpha=0.1)
    typed_sequence = build_typed_sequence(
        sequence=level_sequence,
        section_lengths=_state_section_lengths_tuple(state),
        type_embeddings=model_params["trunk"]["type_embeddings"],
    )
    pred_sequence, _ = forward_with_params(model_params["trunk"], typed_sequence)
    state_out = state.with_updated_sequence(np.asarray(pred_sequence, dtype=np.float32))
    failure_credit = {
        level_id: float(np.asarray(l6_credit[index]))
        for index, level_id in enumerate(LEVEL_IDS)
    }
    metrics = build_canonical_metrics(
        state=state_out,
        failure_credit=failure_credit or neutral_failure_credit(),
        task_validity_score=0.5,
        task_confidence=0.5,
        extra={
            "phase": phase,
            "profile_id": checkpoint.get("profile_id", last_metrics.get("profile_id", "")),
            "segment_id": segment_id,
            "dataset_slice_id": dataset_slice_id,
            "eval.source": "latest_checkpoint",
            "data_seed": resolved_data_seed,
            "policy_bundle_sha256": checkpoint.get(
                "policy_bundle_sha256", last_metrics.get("policy_bundle_sha256", "")
            ),
            "data_realization_policy_id": checkpoint.get(
                "data_realization_policy_id", last_metrics.get("data_realization_policy_id", "")
            ),
            "decontam_policy_id": checkpoint.get(
                "decontam_policy_id", last_metrics.get("decontam_policy_id", "")
            ),
            "learning_objective_bundle_id": checkpoint.get(
                "learning_objective_bundle_id",
                last_metrics.get("learning_objective_bundle_id", ""),
            ),
            "learning_objective_bundle_resolution_source": checkpoint.get(
                "learning_objective_bundle_resolution_source",
                last_metrics.get("learning_objective_bundle_resolution_source", ""),
            ),
            "benchmark_family_policy_refs": checkpoint.get(
                "benchmark_family_policy_refs",
                last_metrics.get("benchmark_family_policy_refs", []),
            ),
            "parser_provenance_id": checkpoint.get(
                "parser_provenance_id", last_metrics.get("parser_provenance_id", "")
            ),
            "parser_provenance_refs": checkpoint.get(
                "parser_provenance_refs", last_metrics.get("parser_provenance_refs", {})
            ),
            "parse_config_fingerprint": checkpoint.get(
                "parse_config_fingerprint", last_metrics.get("parse_config_fingerprint", "")
            ),
            "formalizer_provenance_id": checkpoint.get(
                "formalizer_provenance_id",
                last_metrics.get("formalizer_provenance_id", ""),
            ),
            "verifier_provenance_id": checkpoint.get(
                "verifier_provenance_id", last_metrics.get("verifier_provenance_id", "")
            ),
            "task_family": checkpoint.get("task_family", last_metrics.get("task_family", "")),
            "task_family_resolution_source": checkpoint.get(
                "task_family_resolution_source",
                last_metrics.get("task_family_resolution_source", ""),
            ),
            "task_adjudication_policy_id": checkpoint.get(
                "task_adjudication_policy_id",
                last_metrics.get("task_adjudication_policy_id", ""),
            ),
            "task_adjudication_policy_resolution_source": checkpoint.get(
                "task_adjudication_policy_resolution_source",
                last_metrics.get("task_adjudication_policy_resolution_source", ""),
            ),
            "runtime_status": checkpoint.get("runtime_status", last_metrics.get("runtime_status", "")),
            "adjudication_status": checkpoint.get(
                "adjudication_status",
                last_metrics.get("adjudication_status", ""),
            ),
            "data_source": last_metrics.get("data_source", data_provenance.get("data_source", "synthetic")),
            "data.profile_id": data_provenance.get("profile_id", last_metrics.get("data.profile_id", "")),
            "data.sources_manifest_sha256": data_provenance.get(
                "sources_manifest_sha256",
                last_metrics.get("data.sources_manifest_sha256", ""),
            ),
            "data.tokenizer_fingerprint": data_provenance.get(
                "tokenizer_fingerprint",
                last_metrics.get("data.tokenizer_fingerprint", ""),
            ),
            "data.streaming_mode_effective": data_provenance.get(
                "streaming_mode_effective",
                last_metrics.get("data.streaming_mode_effective", ""),
            ),
        },
    )
    metrics["trunk.backend"] = "jax"
    metrics["trunk.gain"] = 1.0
    return metrics
