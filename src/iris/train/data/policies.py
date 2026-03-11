from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

POOL_IDS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
ROLE_IDS: Tuple[str, ...] = ("core", "auxiliary", "weak_supervision_only", "eval_only")
TIER_IDS: Tuple[str, ...] = ("Tier 1", "Tier 2", "Tier 3")
UNIT_CLASS_IDS: Tuple[str, ...] = (
    "structural_label",
    "process_fragment",
    "theorem_or_problem_statement",
    "proof_shape_fragment",
    "none",
)
FINGERPRINT_LAYER_IDS: Tuple[str, ...] = (
    "source_fingerprint",
    "document_fingerprint",
    "semantic_unit_fingerprint",
    "theorem_or_problem_fingerprint",
    "solution_or_proof_fragment_fingerprint",
    "diagram_anchor_fingerprint",
)
REQUIRED_TRAIN_VISIBLE_RECORD_FIELDS: Tuple[str, ...] = (
    "record_id",
    "pool_id",
    "pool_role",
    "data_realization_policy_id",
    "decontam_policy_id",
    "fingerprint_set",
    "source_family",
    "provenance_refs",
    "quality_flags",
    "source_record_lineage",
    "benchmark_family_id",
    "benchmark_tier",
    "math_document_record_id",
    "formalizer_provenance_id",
)


class PolicyValidationError(ValueError):
    pass


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_tuple_of_str(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _normalize_caps(raw_caps: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "token_cap": float(raw_caps.get("token_cap", 0.0)),
        "record_cap": float(raw_caps.get("record_cap", 0.0)),
    }


def _normalize_parser_refs(raw_refs: Mapping[str, Any] | None) -> Dict[str, str]:
    refs = dict(raw_refs or {})
    normalized: Dict[str, str] = {}
    for key in (
        "layout_parser_manifest_id",
        "ocr_manifest_id",
        "formula_parser_manifest_id",
        "semantic_unit_typer_manifest_id",
    ):
        value = str(refs.get(key, "not_applicable")).strip() or "not_applicable"
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class PoolAllocation:
    token_weight: float
    record_weight: float
    allowed_roles: Tuple[str, ...]


@dataclass(frozen=True)
class DataRealizationPolicy:
    data_realization_policy_id: str
    profile_id: str
    phase: str
    pool_allocations: Dict[str, PoolAllocation]
    source_family_allowlists: Dict[str, Dict[str, Tuple[str, ...]]]
    quality_gate_thresholds: Dict[str, Any]
    tier1_global_cap: Dict[str, float]
    weak_supervision_cap: Dict[str, float]
    benchmark_family_policy_refs: Tuple[str, ...]
    decontam_policy_id: str
    train_visible_record_required_fields: Tuple[str, ...]


@dataclass(frozen=True)
class DecontamPolicy:
    decontam_policy_id: str
    normalization_profile_id: str
    layer_payload_specs: Dict[str, Any]
    layer_thresholds: Dict[str, Any]
    pool_layer_requirements: Dict[str, Any]
    benchmark_family_overrides: Dict[str, Any]
    homologous_split_rules: Dict[str, Any]
    quarantine_and_drop_rules: Dict[str, Any]
    audit_method: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkFamilyPolicy:
    benchmark_family_id: str
    allowed_tiers: Tuple[str, ...]
    tier1_train_visible_units: Tuple[str, ...]
    tier1_label_allowlist: Tuple[str, ...]
    tier1_fragment_allowlist: Tuple[str, ...]
    tier2_homologous_source_id: str
    homology_axes: Tuple[str, ...]
    source_lineage_firewall: Tuple[str, ...]
    cluster_exclusion_key: str
    tier3_strict_holdout_source_id: str
    decontam_policy_id: str
    tuning_visible_surfaces: Tuple[str, ...]
    tuning_observe_only_surfaces: Tuple[str, ...]
    tuning_blocked_surfaces: Tuple[str, ...]
    derivative_family_refs: Tuple[str, ...]
    forbidden_uses: Tuple[str, ...]


@dataclass(frozen=True)
class ProvenanceManifest:
    manifest_id: str
    surface_kind: str
    backend_family: str
    backend_version: str
    build_or_commit_hash: str
    config_fingerprint: str
    artifact_fingerprint: str
    parent_manifest_ids: Tuple[str, ...]
    license_scope: str
    created_at: str


@dataclass(frozen=True)
class DataPolicyBundle:
    schema: str
    data_realization_policy: DataRealizationPolicy
    decontam_policy: DecontamPolicy
    benchmark_family_policies: Dict[str, BenchmarkFamilyPolicy]
    provenance_manifests: Dict[str, ProvenanceManifest]

    @property
    def bundle_sha256(self) -> str:
        return _stable_sha256(
            {
                "schema": self.schema,
                "data_realization_policy_id": self.data_realization_policy.data_realization_policy_id,
                "decontam_policy_id": self.decontam_policy.decontam_policy_id,
                "benchmark_family_ids": sorted(self.benchmark_family_policies.keys()),
                "provenance_manifest_ids": sorted(self.provenance_manifests.keys()),
            }
        )


def _parse_pool_allocations(raw_allocations: Mapping[str, Any]) -> Dict[str, PoolAllocation]:
    allocations: Dict[str, PoolAllocation] = {}
    for pool_id in POOL_IDS:
        if pool_id not in raw_allocations:
            raise PolicyValidationError(f"Missing pool allocation for pool '{pool_id}'.")
        allocation = raw_allocations[pool_id]
        if not isinstance(allocation, Mapping):
            raise PolicyValidationError(f"Pool allocation for '{pool_id}' must be an object.")
        roles = _as_tuple_of_str(allocation.get("allowed_roles", []))
        if not roles:
            raise PolicyValidationError(f"Pool allocation for '{pool_id}' must declare allowed_roles.")
        unknown_roles = sorted(set(roles) - set(ROLE_IDS))
        if unknown_roles:
            raise PolicyValidationError(
                f"Pool allocation for '{pool_id}' declares unknown roles: {unknown_roles}."
            )
        parsed = PoolAllocation(
            token_weight=float(allocation.get("token_weight", 0.0)),
            record_weight=float(allocation.get("record_weight", 0.0)),
            allowed_roles=roles,
        )
        if parsed.token_weight <= 0.0 or parsed.record_weight <= 0.0:
            raise PolicyValidationError(
                f"Pool allocation for '{pool_id}' must use positive token_weight and record_weight."
            )
        allocations[pool_id] = parsed
    return allocations


def _parse_data_realization_policy(raw_policy: Mapping[str, Any]) -> DataRealizationPolicy:
    policy = DataRealizationPolicy(
        data_realization_policy_id=str(raw_policy.get("data_realization_policy_id", "")).strip(),
        profile_id=str(raw_policy.get("profile_id", "")).strip(),
        phase=str(raw_policy.get("phase", "")).strip(),
        pool_allocations=_parse_pool_allocations(dict(raw_policy.get("pool_allocations", {}))),
        source_family_allowlists={
            str(pool_id): {
                str(role): _as_tuple_of_str(values if isinstance(values, Sequence) else [])
                for role, values in dict(raw_roles).items()
            }
            for pool_id, raw_roles in dict(raw_policy.get("source_family_allowlists", {})).items()
            if isinstance(raw_roles, Mapping)
        },
        quality_gate_thresholds=dict(raw_policy.get("quality_gate_thresholds", {})),
        tier1_global_cap=_normalize_caps(dict(raw_policy.get("tier1_global_cap", {}))),
        weak_supervision_cap=_normalize_caps(dict(raw_policy.get("weak_supervision_cap", {}))),
        benchmark_family_policy_refs=_as_tuple_of_str(raw_policy.get("benchmark_family_policy_refs", [])),
        decontam_policy_id=str(raw_policy.get("decontam_policy_id", "")).strip(),
        train_visible_record_required_fields=_as_tuple_of_str(
            raw_policy.get("train_visible_record_required_fields", [])
        ),
    )
    if not policy.data_realization_policy_id:
        raise PolicyValidationError("data_realization_policy_id is required.")
    if not policy.profile_id:
        raise PolicyValidationError("profile_id is required.")
    if not policy.phase:
        raise PolicyValidationError("phase is required.")
    if not policy.decontam_policy_id:
        raise PolicyValidationError("decontam_policy_id is required.")
    if not policy.benchmark_family_policy_refs:
        raise PolicyValidationError("benchmark_family_policy_refs must not be empty.")
    if policy.profile_id not in {"P1", "P2", "P3", "P4"}:
        raise PolicyValidationError(
            f"Unsupported profile_id={policy.profile_id!r}. Expected one of P1/P2/P3/P4."
        )
    if policy.phase not in {"A", "B", "C", "D", "E"}:
        raise PolicyValidationError(
            f"Unsupported phase={policy.phase!r}. Expected one of A/B/C/D/E."
        )
    for cap_name, caps in (
        ("tier1_global_cap", policy.tier1_global_cap),
        ("weak_supervision_cap", policy.weak_supervision_cap),
    ):
        if float(caps["token_cap"]) <= 0.0 or float(caps["record_cap"]) <= 0.0:
            raise PolicyValidationError(f"{cap_name} must declare positive token_cap and record_cap.")
    missing_record_fields = sorted(
        set(REQUIRED_TRAIN_VISIBLE_RECORD_FIELDS) - set(policy.train_visible_record_required_fields)
    )
    if missing_record_fields:
        raise PolicyValidationError(
            "train_visible_record_required_fields is missing required fields: "
            f"{missing_record_fields}."
        )
    pool_b_thresholds = dict(policy.quality_gate_thresholds.get("pool_b", {}))
    if not {"accept_train_visible", "downgrade_fragment_only", "reject"} <= set(pool_b_thresholds.keys()):
        raise PolicyValidationError(
            "quality_gate_thresholds.pool_b must declare accept_train_visible, "
            "downgrade_fragment_only, and reject."
        )
    pool_d_thresholds = dict(policy.quality_gate_thresholds.get("pool_d", {}))
    missing_pool_d_families = sorted(
        {"PDF", "DOCX", "image", "scanned_note", "diagram"} - set(pool_d_thresholds.keys())
    )
    if missing_pool_d_families:
        raise PolicyValidationError(
            f"quality_gate_thresholds.pool_d is missing source families: {missing_pool_d_families}."
        )
    return policy


def _parse_decontam_policy(raw_policy: Mapping[str, Any]) -> DecontamPolicy:
    policy = DecontamPolicy(
        decontam_policy_id=str(raw_policy.get("decontam_policy_id", "")).strip(),
        normalization_profile_id=str(raw_policy.get("normalization_profile_id", "")).strip(),
        layer_payload_specs=dict(raw_policy.get("layer_payload_specs", {})),
        layer_thresholds=dict(raw_policy.get("layer_thresholds", {})),
        pool_layer_requirements=dict(raw_policy.get("pool_layer_requirements", {})),
        benchmark_family_overrides=dict(raw_policy.get("benchmark_family_overrides", {})),
        homologous_split_rules=dict(raw_policy.get("homologous_split_rules", {})),
        quarantine_and_drop_rules=dict(raw_policy.get("quarantine_and_drop_rules", {})),
        audit_method=dict(raw_policy.get("audit_method", {})),
    )
    if not policy.decontam_policy_id:
        raise PolicyValidationError("decontam_policy_id is required.")
    if not policy.normalization_profile_id:
        raise PolicyValidationError("normalization_profile_id is required.")
    missing_layers = sorted(set(FINGERPRINT_LAYER_IDS) - set(policy.layer_payload_specs.keys()))
    if missing_layers:
        raise PolicyValidationError(
            f"decontam_policy.layer_payload_specs is missing fingerprint layers: {missing_layers}."
        )
    if not policy.layer_thresholds:
        raise PolicyValidationError("decontam_policy.layer_thresholds must not be empty.")
    missing_pools = sorted(set(POOL_IDS) - set(policy.pool_layer_requirements.keys()))
    if missing_pools:
        raise PolicyValidationError(
            f"decontam_policy.pool_layer_requirements is missing pools: {missing_pools}."
        )
    for pool_id in POOL_IDS:
        pool_layers = dict(policy.pool_layer_requirements.get(pool_id, {}))
        missing_pool_layers = sorted(set(FINGERPRINT_LAYER_IDS) - set(pool_layers.keys()))
        if missing_pool_layers:
            raise PolicyValidationError(
                f"decontam_policy.pool_layer_requirements[{pool_id}] is missing layers: "
                f"{missing_pool_layers}."
            )
    if not policy.homologous_split_rules:
        raise PolicyValidationError("decontam_policy.homologous_split_rules must not be empty.")
    if not policy.quarantine_and_drop_rules:
        raise PolicyValidationError("decontam_policy.quarantine_and_drop_rules must not be empty.")
    if not policy.audit_method:
        raise PolicyValidationError("decontam_policy.audit_method must not be empty.")
    return policy


def _parse_benchmark_family_policy(raw_policy: Mapping[str, Any]) -> BenchmarkFamilyPolicy:
    policy = BenchmarkFamilyPolicy(
        benchmark_family_id=str(raw_policy.get("benchmark_family_id", "")).strip(),
        allowed_tiers=_as_tuple_of_str(raw_policy.get("allowed_tiers", [])),
        tier1_train_visible_units=_as_tuple_of_str(raw_policy.get("tier1_train_visible_units", [])),
        tier1_label_allowlist=_as_tuple_of_str(raw_policy.get("tier1_label_allowlist", [])),
        tier1_fragment_allowlist=_as_tuple_of_str(raw_policy.get("tier1_fragment_allowlist", [])),
        tier2_homologous_source_id=str(raw_policy.get("tier2_homologous_source_id", "")).strip(),
        homology_axes=_as_tuple_of_str(raw_policy.get("homology_axes", [])),
        source_lineage_firewall=_as_tuple_of_str(raw_policy.get("source_lineage_firewall", [])),
        cluster_exclusion_key=str(raw_policy.get("cluster_exclusion_key", "")).strip(),
        tier3_strict_holdout_source_id=str(raw_policy.get("tier3_strict_holdout_source_id", "")).strip(),
        decontam_policy_id=str(raw_policy.get("decontam_policy_id", "")).strip(),
        tuning_visible_surfaces=_as_tuple_of_str(raw_policy.get("tuning_visible_surfaces", [])),
        tuning_observe_only_surfaces=_as_tuple_of_str(
            raw_policy.get("tuning_observe_only_surfaces", [])
        ),
        tuning_blocked_surfaces=_as_tuple_of_str(raw_policy.get("tuning_blocked_surfaces", [])),
        derivative_family_refs=_as_tuple_of_str(raw_policy.get("derivative_family_refs", [])),
        forbidden_uses=_as_tuple_of_str(raw_policy.get("forbidden_uses", [])),
    )
    if not policy.benchmark_family_id:
        raise PolicyValidationError("benchmark_family_id is required.")
    unknown_tiers = sorted(set(policy.allowed_tiers) - set(TIER_IDS))
    if unknown_tiers:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} declares unknown tiers: {unknown_tiers}."
        )
    unknown_units = sorted(set(policy.tier1_train_visible_units) - set(UNIT_CLASS_IDS))
    if unknown_units:
        raise PolicyValidationError(
            "benchmark_family_id="
            f"{policy.benchmark_family_id} declares unknown train-visible units: {unknown_units}."
        )
    if "none" in policy.tier1_train_visible_units and len(policy.tier1_train_visible_units) > 1:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} cannot mix 'none' with other Tier 1 units."
        )
    if "Tier 1" in policy.allowed_tiers and not policy.homology_axes:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must declare homology_axes for Tier 1."
        )
    if "Tier 1" in policy.allowed_tiers and not policy.tier2_homologous_source_id:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must declare tier2_homologous_source_id."
        )
    if "Tier 1" in policy.allowed_tiers and "Tier 2" not in policy.allowed_tiers:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must allow Tier 2 whenever Tier 1 is allowed."
        )
    if "Tier 3" in policy.allowed_tiers and not policy.tier3_strict_holdout_source_id:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must declare tier3_strict_holdout_source_id."
        )
    if "Tier 3" in policy.allowed_tiers and not any(
        "tier3" in surface.lower() for surface in policy.tuning_blocked_surfaces
    ):
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must block Tier 3 tuning surfaces."
        )
    if "theorem_or_problem_statement" in policy.tier1_train_visible_units and (
        "theorem_family" not in policy.homology_axes
    ):
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must include theorem_family in homology_axes "
            "when theorem_or_problem_statement is Tier 1-visible."
        )
    if "proof_shape_fragment" in policy.tier1_train_visible_units and "proof_pattern" not in policy.homology_axes:
        raise PolicyValidationError(
            f"benchmark_family_id={policy.benchmark_family_id} must include proof_pattern in homology_axes "
            "when proof_shape_fragment is Tier 1-visible."
        )
    return policy


def _parse_provenance_manifest(raw_manifest: Mapping[str, Any]) -> ProvenanceManifest:
    manifest = ProvenanceManifest(
        manifest_id=str(raw_manifest.get("manifest_id", "")).strip(),
        surface_kind=str(raw_manifest.get("surface_kind", "")).strip(),
        backend_family=str(raw_manifest.get("backend_family", "")).strip(),
        backend_version=str(raw_manifest.get("backend_version", "")).strip(),
        build_or_commit_hash=str(raw_manifest.get("build_or_commit_hash", "")).strip(),
        config_fingerprint=str(raw_manifest.get("config_fingerprint", "")).strip(),
        artifact_fingerprint=str(raw_manifest.get("artifact_fingerprint", "")).strip(),
        parent_manifest_ids=_as_tuple_of_str(raw_manifest.get("parent_manifest_ids", [])),
        license_scope=str(raw_manifest.get("license_scope", "")).strip(),
        created_at=str(raw_manifest.get("created_at", "")).strip(),
    )
    missing = [
        field_name
        for field_name in (
            "manifest_id",
            "surface_kind",
            "backend_family",
            "backend_version",
            "build_or_commit_hash",
            "config_fingerprint",
            "artifact_fingerprint",
            "license_scope",
            "created_at",
        )
        if not getattr(manifest, field_name)
    ]
    if missing:
        raise PolicyValidationError(
            f"provenance_manifest is missing required fields: {', '.join(missing)}."
        )
    return manifest


def validate_policy_bundle(bundle: DataPolicyBundle) -> DataPolicyBundle:
    if bundle.schema != "iris.data_policy_bundle/v1":
        raise PolicyValidationError("Data policy bundle schema must be iris.data_policy_bundle/v1.")
    if bundle.data_realization_policy.decontam_policy_id != bundle.decontam_policy.decontam_policy_id:
        raise PolicyValidationError("Data realization policy and decontam policy ids do not match.")
    missing_family_refs = sorted(
        set(bundle.data_realization_policy.benchmark_family_policy_refs)
        - set(bundle.benchmark_family_policies.keys())
    )
    if missing_family_refs:
        raise PolicyValidationError(
            f"Unresolved benchmark_family_policy_refs: {missing_family_refs}."
        )
    for family_policy in bundle.benchmark_family_policies.values():
        if family_policy.decontam_policy_id != bundle.decontam_policy.decontam_policy_id:
            raise PolicyValidationError(
                f"benchmark_family_id={family_policy.benchmark_family_id} references a different decontam policy."
            )
    return bundle


def load_policy_bundle(path: Path) -> DataPolicyBundle:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise PolicyValidationError(f"Data policy bundle not found: {bundle_path}")

    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise PolicyValidationError(f"Data policy bundle is not valid JSON: {bundle_path}") from error
    if not isinstance(payload, Mapping):
        raise PolicyValidationError("Data policy bundle must be a JSON object.")

    raw_family_policies = payload.get("benchmark_family_policies", [])
    raw_manifests = payload.get("provenance_manifests", [])
    if not isinstance(raw_family_policies, Sequence) or isinstance(
        raw_family_policies, (str, bytes, bytearray)
    ):
        raise PolicyValidationError("benchmark_family_policies must be an array.")
    if not isinstance(raw_manifests, Sequence) or isinstance(raw_manifests, (str, bytes, bytearray)):
        raise PolicyValidationError("provenance_manifests must be an array.")

    bundle = DataPolicyBundle(
        schema=str(payload.get("schema", "")).strip(),
        data_realization_policy=_parse_data_realization_policy(
            dict(payload.get("data_realization_policy", {}))
        ),
        decontam_policy=_parse_decontam_policy(dict(payload.get("decontam_policy", {}))),
        benchmark_family_policies={
            policy.benchmark_family_id: policy
            for policy in (
                _parse_benchmark_family_policy(dict(raw_policy))
                for raw_policy in raw_family_policies
                if isinstance(raw_policy, Mapping)
            )
        },
        provenance_manifests={
            manifest.manifest_id: manifest
            for manifest in (
                _parse_provenance_manifest(dict(raw_manifest))
                for raw_manifest in raw_manifests
                if isinstance(raw_manifest, Mapping)
            )
        },
    )
    return validate_policy_bundle(bundle)


def load_default_policy_bundle() -> DataPolicyBundle:
    bundle_path = Path(__file__).resolve().parent / "profiles" / "p1_bootstrap_policy_bundle_v1.json"
    return load_policy_bundle(bundle_path)


def load_policy_bundle_for_profile(profile_id: str) -> DataPolicyBundle:
    normalized = str(profile_id).strip().upper()
    bundle_names = {
        "P1": "p1_bootstrap_policy_bundle_v1.json",
        "P2": "p2_bootstrap_policy_bundle_v1.json",
        "P3": "p3_bootstrap_policy_bundle_v1.json",
    }
    if normalized not in bundle_names:
        raise PolicyValidationError(f"No bootstrap policy bundle registered for profile_id={profile_id!r}.")
    bundle_path = Path(__file__).resolve().parent / "profiles" / bundle_names[normalized]
    return load_policy_bundle(bundle_path)


def build_document_slice_id(
    *,
    run_id: str,
    segment_id: int,
    micro_step_idx: int,
    data_seed: int,
    math_document_record_id: str,
    data_realization_policy_id: str,
    decontam_policy_id: str,
    parser_provenance_id: str,
    parser_provenance_refs: Mapping[str, Any] | None,
    parse_config_fingerprint: str,
    ocr_layout_extractor_version: str = "not_applicable",
    formula_parser_version: str = "not_applicable",
    semantic_unit_typer_version: str = "not_applicable",
    formalizer_provenance_id: str = "not_applicable",
    verifier_provenance_id: str = "not_applicable",
    verifier_build_id: str = "not_applicable",
) -> str:
    payload = {
        "run_id": str(run_id),
        "segment_id": int(segment_id),
        "micro_step_idx": int(micro_step_idx),
        "data_seed": int(data_seed),
        "math_document_record_id": str(math_document_record_id),
        "data_realization_policy_id": str(data_realization_policy_id),
        "decontam_policy_id": str(decontam_policy_id),
        "parser_provenance_id": str(parser_provenance_id),
        "parser_provenance_refs": _normalize_parser_refs(parser_provenance_refs),
        "parse_config_fingerprint": str(parse_config_fingerprint),
        "ocr_layout_extractor_version": str(ocr_layout_extractor_version or "not_applicable"),
        "formula_parser_version": str(formula_parser_version or "not_applicable"),
        "semantic_unit_typer_version": str(semantic_unit_typer_version or "not_applicable"),
        "formalizer_provenance_id": str(formalizer_provenance_id or "not_applicable"),
        "verifier_provenance_id": str(verifier_provenance_id or "not_applicable"),
        "verifier_build_id": str(verifier_build_id or "not_applicable"),
    }
    return "docslice-" + _stable_sha256(payload)[:24]
