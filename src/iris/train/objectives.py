from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

OBJECTIVE_FAMILY_IDS: Tuple[str, ...] = (
    "obj.rep.state_construction",
    "obj.proc.frontier_induction",
    "obj.search.control_and_recovery",
    "obj.mem.applicability",
    "obj.abs.compression",
    "obj.eval.verification_and_calibration",
    "obj.task.outcome",
)
LEARNING_OBJECTIVE_BUNDLE_RESOLUTION_SOURCES: Tuple[str, ...] = (
    "run_manifest",
    "profile_phase_default",
)


class LearningObjectiveValidationError(ValueError):
    pass


def _require_text(field_name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise LearningObjectiveValidationError(f"{field_name} is required.")
    return text


def _tuple_of_str(values: Sequence[Any] | None) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class ObjectiveFamilyActivation:
    objective_family_id: str
    weight: float
    activation_status: str


@dataclass(frozen=True)
class LearningObjectiveBundle:
    schema: str
    learning_objective_bundle_id: str
    profile_id: str
    phase: str
    objective_families: Tuple[ObjectiveFamilyActivation, ...]
    level_surface_map: Dict[str, Tuple[str, ...]]
    control_learning_mode: str
    verifier_conditioning_mode: str
    failure_replay_policy: str
    curriculum_policy_summary: str
    benchmark_visibility_guardrails: str
    partial_mount_overrides: Tuple[str, ...]
    parent_bundle_id: str | None = None
    lineage_note: str | None = None


def _registry_path() -> Path:
    return Path(__file__).resolve().parent / "registries" / "learning_objective_bundles_v1.json"


def _parse_activation(payload: Mapping[str, Any]) -> ObjectiveFamilyActivation:
    activation = ObjectiveFamilyActivation(
        objective_family_id=_require_text(
            "objective_family.objective_family_id",
            payload.get("objective_family_id", ""),
        ),
        weight=float(payload.get("weight", 0.0)),
        activation_status=_require_text(
            "objective_family.activation_status",
            payload.get("activation_status", ""),
        ),
    )
    if activation.objective_family_id not in set(OBJECTIVE_FAMILY_IDS):
        raise LearningObjectiveValidationError(
            f"Unknown objective_family_id={activation.objective_family_id!r}."
        )
    return activation


def _parse_bundle(payload: Mapping[str, Any]) -> LearningObjectiveBundle:
    bundle = LearningObjectiveBundle(
        schema=_require_text("learning_objective_bundle.schema", payload.get("schema", "")),
        learning_objective_bundle_id=_require_text(
            "learning_objective_bundle.learning_objective_bundle_id",
            payload.get("learning_objective_bundle_id", ""),
        ),
        profile_id=_require_text("learning_objective_bundle.profile_id", payload.get("profile_id", "")),
        phase=_require_text("learning_objective_bundle.phase", payload.get("phase", "")),
        objective_families=tuple(
            _parse_activation(dict(item))
            for item in payload.get("objective_families", [])
            if isinstance(item, Mapping)
        ),
        level_surface_map={
            str(key): _tuple_of_str(value if isinstance(value, Sequence) else [])
            for key, value in dict(payload.get("level_surface_map", {})).items()
        },
        control_learning_mode=_require_text(
            "learning_objective_bundle.control_learning_mode",
            payload.get("control_learning_mode", ""),
        ),
        verifier_conditioning_mode=_require_text(
            "learning_objective_bundle.verifier_conditioning_mode",
            payload.get("verifier_conditioning_mode", ""),
        ),
        failure_replay_policy=_require_text(
            "learning_objective_bundle.failure_replay_policy",
            payload.get("failure_replay_policy", ""),
        ),
        curriculum_policy_summary=_require_text(
            "learning_objective_bundle.curriculum_policy_summary",
            payload.get("curriculum_policy_summary", ""),
        ),
        benchmark_visibility_guardrails=_require_text(
            "learning_objective_bundle.benchmark_visibility_guardrails",
            payload.get("benchmark_visibility_guardrails", ""),
        ),
        partial_mount_overrides=_tuple_of_str(payload.get("partial_mount_overrides", [])),
        parent_bundle_id=str(payload.get("parent_bundle_id", "")).strip() or None,
        lineage_note=str(payload.get("lineage_note", "")).strip() or None,
    )
    if bundle.schema != "learning_objective_bundle/v1":
        raise LearningObjectiveValidationError(
            "learning objective bundle schema must be learning_objective_bundle/v1."
        )
    if bundle.profile_id not in {"P1", "P2", "P3", "P4"}:
        raise LearningObjectiveValidationError(f"Unsupported profile_id={bundle.profile_id!r}.")
    if bundle.phase not in {"A", "B", "C", "D", "E"}:
        raise LearningObjectiveValidationError(f"Unsupported phase={bundle.phase!r}.")
    if not bundle.objective_families:
        raise LearningObjectiveValidationError(
            f"Bundle {bundle.learning_objective_bundle_id} must declare objective_families."
        )
    return bundle


def load_learning_objective_bundle_registry(
    path: Path | None = None,
) -> tuple[Dict[str, LearningObjectiveBundle], Dict[str, str]]:
    registry_path = Path(path) if path is not None else _registry_path()
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise LearningObjectiveValidationError("learning objective bundle registry must be a JSON object.")
    bundles = payload.get("bundles", [])
    if not isinstance(bundles, Sequence) or isinstance(bundles, (str, bytes, bytearray)):
        raise LearningObjectiveValidationError("learning objective bundle registry bundles must be an array.")
    parsed = {
        bundle.learning_objective_bundle_id: bundle
        for bundle in (_parse_bundle(dict(item)) for item in bundles if isinstance(item, Mapping))
    }
    default_map = {
        str(key): str(value).strip()
        for key, value in dict(payload.get("default_bundle_map", {})).items()
        if str(value).strip()
    }
    for key, bundle_id in default_map.items():
        if bundle_id not in parsed:
            raise LearningObjectiveValidationError(
                f"default_bundle_map[{key!r}] references unresolved bundle id {bundle_id!r}."
            )
    return parsed, default_map


def load_learning_objective_bundle(
    learning_objective_bundle_id: str,
    *,
    registry: Mapping[str, LearningObjectiveBundle] | None = None,
) -> LearningObjectiveBundle:
    bundles = dict(registry or load_learning_objective_bundle_registry()[0])
    try:
        return bundles[str(learning_objective_bundle_id)]
    except KeyError as error:
        raise LearningObjectiveValidationError(
            f"Unresolved learning_objective_bundle_id={learning_objective_bundle_id!r}."
        ) from error


def resolve_learning_objective_bundle(
    *,
    profile_id: str,
    phase: str,
    learning_objective_bundle_id: str | None = None,
    registry: Mapping[str, LearningObjectiveBundle] | None = None,
    default_map: Mapping[str, str] | None = None,
) -> tuple[LearningObjectiveBundle, str]:
    bundles, defaults = (
        (dict(registry), dict(default_map or {}))
        if registry is not None
        else load_learning_objective_bundle_registry()
    )
    if learning_objective_bundle_id:
        return (
            load_learning_objective_bundle(learning_objective_bundle_id, registry=bundles),
            "run_manifest",
        )
    key = f"{str(profile_id).strip().upper()}:{str(phase).strip().upper()}"
    bundle_id = str(defaults.get(key, "")).strip()
    if not bundle_id:
        raise LearningObjectiveValidationError(
            f"No default learning objective bundle registered for {key}."
        )
    return load_learning_objective_bundle(bundle_id, registry=bundles), "profile_phase_default"
