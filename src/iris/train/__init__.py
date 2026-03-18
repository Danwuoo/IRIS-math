from importlib import import_module

from .data import (
    AIMO3DatasetError,
    AIMO3Item,
    AIMO3LocalDataset,
    BenchmarkFamilyPolicy,
    DataPolicyBundle,
    DataRealizationPolicy,
    DatasetSourceSpec,
    DecontamPolicy,
    P1RecordAdmission,
    PolicyValidationError,
    ProfileValidationError,
    ProvenanceManifest,
    PureLMProfile,
    TrainVisibleRecord,
    admit_p1_train_visible_record,
    build_document_slice_id,
    build_document_pipeline_bundle,
    load_aimo3_local_dataset,
    load_default_policy_bundle,
    load_default_pure_lm_profile,
    load_policy_bundle,
    load_policy_bundle_for_profile,
    load_policy_bundle_for_profile_phase,
    load_profile,
    profile_sha256,
    sources_manifest_sha256,
    validate_train_visible_record,
)
from .data.iterator import PureLMStreamingProvider
from .data.token_accounting import TokenizerError, load_tokenizer_handle
from .hf_sync import (
    HFRepoSpec,
    HFSyncError,
    bootstrap_checkpoint_run,
    read_latest_pointer,
    resolve_dataset_commit_sha,
    sync_checkpoint_run,
    sync_final_release,
    write_latest_pointer,
)
from .iris3b_config import (
    IRIS3BConfig,
    InterleavedLevelPlacement,
    default_iris3b_config,
    iris3b_config_from_mapping,
)
from .objectives import (
    LEARNING_OBJECTIVE_BUNDLE_RESOLUTION_SOURCES,
    LearningObjectiveBundle,
    LearningObjectiveValidationError,
    ObjectiveFamilyActivation,
    load_learning_objective_bundle,
    load_learning_objective_bundle_registry,
    resolve_learning_objective_bundle,
)

__all__ = [
    "AIMO3DatasetError",
    "AIMO3Item",
    "AIMO3LocalDataset",
    "BenchmarkFamilyPolicy",
    "DataPolicyBundle",
    "DataRealizationPolicy",
    "DatasetSourceSpec",
    "DecontamPolicy",
    "HFRepoSpec",
    "HFSyncError",
    "IRIS3BConfig",
    "InterleavedLevelPlacement",
    "LEARNING_OBJECTIVE_BUNDLE_RESOLUTION_SOURCES",
    "LearningObjectiveBundle",
    "LearningObjectiveValidationError",
    "ObjectiveFamilyActivation",
    "P1TrainConfig",
    "P1RecordAdmission",
    "PolicyValidationError",
    "ProfileValidationError",
    "ProvenanceManifest",
    "PureLMProfile",
    "PureLMStreamingProvider",
    "TokenizerError",
    "ToyTrainConfig",
    "TrainVisibleRecord",
    "admit_p1_train_visible_record",
    "bootstrap_checkpoint_run",
    "build_document_pipeline_bundle",
    "build_document_slice_id",
    "default_iris3b_config",
    "evaluate_latest_run",
    "export_final_release",
    "load_default_policy_bundle",
    "load_default_pure_lm_profile",
    "load_aimo3_local_dataset",
    "load_learning_objective_bundle",
    "load_learning_objective_bundle_registry",
    "load_policy_bundle",
    "load_policy_bundle_for_profile",
    "load_policy_bundle_for_profile_phase",
    "load_profile",
    "load_tokenizer_handle",
    "iris3b_config_from_mapping",
    "read_latest_pointer",
    "profile_sha256",
    "resolve_dataset_commit_sha",
    "resolve_learning_objective_bundle",
    "run_p1_training_cycle",
    "run_toy_training",
    "sync_checkpoint_run",
    "sync_final_release",
    "sources_manifest_sha256",
    "validate_train_visible_record",
    "write_latest_pointer",
]

_OPTIONAL_EXPORTS = {
    "P1TrainConfig": ("iris.train.iris3b_training", "P1TrainConfig"),
    "ToyTrainConfig": ("iris.train.loop", "ToyTrainConfig"),
    "run_toy_training": ("iris.train.loop", "run_toy_training"),
    "run_p1_training_cycle": ("iris.train.iris3b_training", "run_p1_training_cycle"),
    "export_final_release": ("iris.train.iris3b_training", "export_final_release"),
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
