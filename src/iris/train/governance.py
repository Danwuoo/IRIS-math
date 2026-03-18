from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .data import load_policy_bundle_for_profile_phase
from .objectives import resolve_learning_objective_bundle


def stable_hash(payload: Any) -> str:
    def _default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Unsupported stable-hash payload type: {type(value)!r}")

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def git_code_version_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_runtime_lock_manifest(*, phase: str, extra_env_keys: tuple[str, ...] = ()) -> Dict[str, Any]:
    env_keys = (
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DISABLE_JIT",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        *extra_env_keys,
    )
    return {
        "schema": "iris.runtime_lock_manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "phase": str(phase).strip().upper(),
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
                {"name": "jax", "version": safe_package_version("jax"), "hash": "n/a"},
                {"name": "jaxlib", "version": safe_package_version("jaxlib"), "hash": "n/a"},
                {"name": "flax", "version": safe_package_version("flax"), "hash": "n/a"},
                {"name": "optax", "version": safe_package_version("optax"), "hash": "n/a"},
                {
                    "name": "orbax-checkpoint",
                    "version": safe_package_version("orbax-checkpoint"),
                    "hash": "n/a",
                },
                {"name": "datasets", "version": safe_package_version("datasets"), "hash": "n/a"},
                {
                    "name": "transformers",
                    "version": safe_package_version("transformers"),
                    "hash": "n/a",
                },
                {
                    "name": "huggingface_hub",
                    "version": safe_package_version("huggingface-hub"),
                    "hash": "n/a",
                },
                {
                    "name": "sentencepiece",
                    "version": safe_package_version("sentencepiece"),
                    "hash": "n/a",
                },
                {"name": "numpy", "version": safe_package_version("numpy"), "hash": "n/a"},
            ],
        },
        "jax": {
            "jax": safe_package_version("jax"),
            "jaxlib": safe_package_version("jaxlib"),
            "jaxlib_build": "unknown",
            "xla_flags": str(os.environ.get("XLA_FLAGS", "")),
            "env": {key: str(os.environ.get(key, "")) for key in env_keys},
        },
    }


def load_pinned_runtime_lock_manifest(path: Path, *, phase: str) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise RuntimeError(f"Pinned runtime lock manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Pinned runtime lock manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pinned runtime lock manifest must be a JSON object: {manifest_path}")
    if str(payload.get("schema", "")) != "iris.runtime_lock_manifest/v1":
        raise RuntimeError("Pinned runtime lock manifest schema mismatch.")
    expected_phase = str(phase).strip().upper()
    actual_phase = str(payload.get("phase", "")).strip().upper()
    if actual_phase and actual_phase != expected_phase:
        raise RuntimeError(
            f"Pinned runtime lock manifest phase mismatch. Expected '{expected_phase}', got '{actual_phase}'."
        )
    return payload


def write_runtime_lock_manifest(
    *,
    output_dir: Path,
    phase: str,
    runtime_lock_manifest_path: Path | None = None,
) -> Dict[str, str]:
    if runtime_lock_manifest_path is None:
        manifest = build_runtime_lock_manifest(phase=phase)
    else:
        manifest = load_pinned_runtime_lock_manifest(runtime_lock_manifest_path, phase=phase)
    manifest_path = Path(output_dir) / "runtime_lock_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    return {
        "runtime_lock_manifest_id": manifest_sha[:12],
        "runtime_lock_manifest_sha256": manifest_sha,
        "runtime_lock_manifest_path": str(manifest_path),
    }


def active_governance_snapshot(*, profile_id: str, phase: str) -> Dict[str, Any]:
    bundle = load_policy_bundle_for_profile_phase(profile_id, phase)
    learning_bundle, resolution_source = resolve_learning_objective_bundle(
        profile_id=str(profile_id).strip().upper(),
        phase=str(phase).strip().upper(),
    )
    manifests = bundle.provenance_manifests
    parser_manifest = manifests.get("math-doc-pipeline-v1")
    layout_manifest = manifests.get("layout-parser-v1")
    ocr_manifest = manifests.get("ocr-parser-v1")
    formula_manifest = manifests.get("formula-parser-v1")
    semantic_manifest = manifests.get("semantic-unit-typer-v1")
    formalizer_manifest = manifests.get("formalizer-skeleton-v1")
    verifier_manifest = manifests.get("verifier-stack-v1")

    def _manifest_version(manifest: Any | None) -> str:
        if manifest is None:
            return "not_applicable"
        return str(getattr(manifest, "backend_version", "not_applicable") or "not_applicable")

    def _manifest_build_id(manifest: Any | None) -> str:
        if manifest is None:
            return "not_applicable"
        return str(getattr(manifest, "build_or_commit_hash", "not_applicable") or "not_applicable")

    return {
        "policy_profile_id": str(profile_id).strip().upper(),
        "phase": str(phase).strip().upper(),
        "policy_bundle_sha256": bundle.bundle_sha256,
        "data_realization_policy_id": bundle.data_realization_policy.data_realization_policy_id,
        "decontam_policy_id": bundle.decontam_policy.decontam_policy_id,
        "benchmark_family_policy_refs": list(
            bundle.data_realization_policy.benchmark_family_policy_refs
        ),
        "learning_objective_bundle_id": learning_bundle.learning_objective_bundle_id,
        "learning_objective_bundle_resolution_source": resolution_source,
        "learning_objective_bundle_sha256": stable_hash(asdict(learning_bundle)),
        "parser_provenance_id": getattr(parser_manifest, "manifest_id", "not_applicable"),
        "parser_provenance_refs": {
            "layout_parser_manifest_id": getattr(layout_manifest, "manifest_id", "not_applicable"),
            "ocr_manifest_id": getattr(ocr_manifest, "manifest_id", "not_applicable"),
            "formula_parser_manifest_id": getattr(formula_manifest, "manifest_id", "not_applicable"),
            "semantic_unit_typer_manifest_id": getattr(
                semantic_manifest, "manifest_id", "not_applicable"
            ),
        },
        "parse_config_fingerprint": getattr(parser_manifest, "config_fingerprint", "not_applicable"),
        "ocr_layout_extractor_version": (
            f"layout:{_manifest_version(layout_manifest)}|ocr:{_manifest_version(ocr_manifest)}"
        ),
        "formula_parser_version": _manifest_version(formula_manifest),
        "semantic_unit_typer_version": _manifest_version(semantic_manifest),
        "formalizer_provenance_id": getattr(formalizer_manifest, "manifest_id", "not_applicable"),
        "formalizer_version": _manifest_version(formalizer_manifest),
        "verifier_provenance_id": getattr(verifier_manifest, "manifest_id", "not_applicable"),
        "verifier_build_id": _manifest_build_id(verifier_manifest),
    }
