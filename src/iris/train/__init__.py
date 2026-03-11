from importlib import import_module

from .data import (
    BenchmarkFamilyPolicy,
    DataPolicyBundle,
    DataRealizationPolicy,
    DatasetSourceSpec,
    DecontamPolicy,
    PolicyValidationError,
    ProfileValidationError,
    ProvenanceManifest,
    PureLMProfile,
    build_document_slice_id,
    load_default_policy_bundle,
    load_default_pure_lm_profile,
    load_policy_bundle,
    load_policy_bundle_for_profile,
    load_profile,
    profile_sha256,
    sources_manifest_sha256,
)
from .data.iterator import PureLMStreamingProvider
from .data.token_accounting import TokenizerError, load_tokenizer_handle

__all__ = [
    "BenchmarkFamilyPolicy",
    "DataPolicyBundle",
    "DataRealizationPolicy",
    "DatasetSourceSpec",
    "DecontamPolicy",
    "PolicyValidationError",
    "ProfileValidationError",
    "ProvenanceManifest",
    "PureLMProfile",
    "PureLMStreamingProvider",
    "TokenizerError",
    "ToyTrainConfig",
    "build_document_slice_id",
    "evaluate_latest_run",
    "load_default_policy_bundle",
    "load_default_pure_lm_profile",
    "load_policy_bundle",
    "load_policy_bundle_for_profile",
    "load_profile",
    "load_tokenizer_handle",
    "profile_sha256",
    "run_toy_training",
    "sources_manifest_sha256",
]

_OPTIONAL_EXPORTS = {
    "ToyTrainConfig": ("iris.train.loop", "ToyTrainConfig"),
    "run_toy_training": ("iris.train.loop", "run_toy_training"),
    "evaluate_latest_run": ("iris.train.eval", "evaluate_latest_run"),
}


def __getattr__(name: str):
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module 'iris.train' has no attribute {name!r}")

    module_name, attr_name = _OPTIONAL_EXPORTS[name]
    try:
        module = import_module(module_name)
    except Exception as error:  # pragma: no cover - optional dependency failure
        raise RuntimeError(
            "Training runtime exports require the optional JAX/flax/optax stack."
        ) from error
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
