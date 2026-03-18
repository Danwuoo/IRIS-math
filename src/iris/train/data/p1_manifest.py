from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from ..governance import stable_hash

_VALID_POOLS = {"A", "B", "C", "D", "E"}
_VALID_SOURCE_KINDS = {"hf", "synthetic"}
_BOOTSTRAP_ALLOWED_REVISION_PREFIXES = ("resolve:", "sha:")
_BOOTSTRAP_POOL_TOKEN_WEIGHTS = {"A": 0.20, "B": 0.10, "C": 0.25, "D": 0.35, "E": 0.10}
_BOOTSTRAP_POOL_RECORD_WEIGHTS = {"A": 0.25, "B": 0.10, "C": 0.25, "D": 0.20, "E": 0.20}


class P1ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class P1StreamingSource:
    source_id: str
    source_kind: str
    pool_id: str
    pool_role: str
    token_weight: float
    record_weight: float
    hf_path: str | None = None
    hf_name: str | None = None
    split: str = "train"
    revision: str | None = None
    payload_field: str = "text"
    filter_signature: str = "default"
    source_family: str = "text"
    benchmark_family_refs: Tuple[str, ...] = ()
    local_snapshot_pattern: str | None = None
    admission_posture: str = "streaming_ready"
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "pool_id": self.pool_id,
            "pool_role": self.pool_role,
            "token_weight": float(self.token_weight),
            "record_weight": float(self.record_weight),
            "hf_path": self.hf_path,
            "hf_name": self.hf_name,
            "split": self.split,
            "revision": self.revision,
            "payload_field": self.payload_field,
            "filter_signature": self.filter_signature,
            "source_family": self.source_family,
            "benchmark_family_refs": list(self.benchmark_family_refs),
            "local_snapshot_pattern": self.local_snapshot_pattern,
            "admission_posture": self.admission_posture,
            "required": bool(self.required),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class P1StreamingManifest:
    schema: str
    manifest_id: str
    profile_id: str
    phase: str
    commit_posture: str
    default_streaming_mode: str
    sources: Tuple[P1StreamingSource, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "profile_id": self.profile_id,
            "phase": self.phase,
            "commit_posture": self.commit_posture,
            "default_streaming_mode": self.default_streaming_mode,
            "sources": [source.to_payload() for source in self.sources],
        }

    @property
    def source_by_id(self) -> Dict[str, P1StreamingSource]:
        return {source.source_id: source for source in self.sources}

    @property
    def sha256(self) -> str:
        return manifest_sha256(self)


def _tuple_of_str(values: Sequence[Any] | None) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _coerce_source(raw_source: Mapping[str, Any]) -> P1StreamingSource:
    metadata = raw_source.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return P1StreamingSource(
        source_id=str(raw_source.get("source_id", "")).strip(),
        source_kind=str(raw_source.get("source_kind", "hf")).strip().lower(),
        pool_id=str(raw_source.get("pool_id", "")).strip().upper(),
        pool_role=str(raw_source.get("pool_role", "")).strip(),
        token_weight=float(raw_source.get("token_weight", 0.0)),
        record_weight=float(raw_source.get("record_weight", 0.0)),
        hf_path=(
            None
            if raw_source.get("hf_path") in (None, "")
            else str(raw_source.get("hf_path", "")).strip()
        ),
        hf_name=(
            None
            if raw_source.get("hf_name") in (None, "")
            else str(raw_source.get("hf_name", "")).strip()
        ),
        split=str(raw_source.get("split", "train")).strip() or "train",
        revision=(
            None
            if raw_source.get("revision") in (None, "")
            else str(raw_source.get("revision", "")).strip()
        ),
        payload_field=str(raw_source.get("payload_field", "text")).strip() or "text",
        filter_signature=str(raw_source.get("filter_signature", "default")).strip() or "default",
        source_family=str(raw_source.get("source_family", "text")).strip() or "text",
        benchmark_family_refs=_tuple_of_str(raw_source.get("benchmark_family_refs", [])),
        local_snapshot_pattern=(
            None
            if raw_source.get("local_snapshot_pattern") in (None, "")
            else str(raw_source.get("local_snapshot_pattern", "")).strip()
        ),
        admission_posture=str(raw_source.get("admission_posture", "streaming_ready")).strip(),
        required=bool(raw_source.get("required", True)),
        metadata={str(key): value for key, value in metadata.items()},
    )


def _is_immutable_revision(revision: str) -> bool:
    normalized = str(revision).strip()
    if not normalized:
        return False
    if normalized.startswith("sha:"):
        normalized = normalized.split("sha:", 1)[1]
    if len(normalized) < 7 or len(normalized) > 64:
        return False
    return all(character in "0123456789abcdef" for character in normalized.lower())


def validate_manifest(
    manifest: P1StreamingManifest,
    *,
    require_committed_revisions: bool | None = None,
) -> P1StreamingManifest:
    if manifest.schema != "p1_streaming_manifest/v1":
        raise P1ManifestValidationError("schema must be p1_streaming_manifest/v1.")
    if manifest.profile_id != "P1":
        raise P1ManifestValidationError("profile_id must be P1.")
    if manifest.phase not in {"A", "B", "C", "D", "E"}:
        raise P1ManifestValidationError("phase must be one of A/B/C/D/E.")
    if manifest.commit_posture not in {"bootstrap", "committed"}:
        raise P1ManifestValidationError("commit_posture must be bootstrap or committed.")
    if not manifest.sources:
        raise P1ManifestValidationError("manifest must declare at least one source.")

    require_committed_revisions = (
        manifest.commit_posture == "committed"
        if require_committed_revisions is None
        else bool(require_committed_revisions)
    )

    seen_ids: set[str] = set()
    token_totals = {pool_id: 0.0 for pool_id in _VALID_POOLS}
    record_totals = {pool_id: 0.0 for pool_id in _VALID_POOLS}

    for source in manifest.sources:
        if not source.source_id:
            raise P1ManifestValidationError("source_id cannot be empty.")
        if source.source_id in seen_ids:
            raise P1ManifestValidationError(f"Duplicate source_id: {source.source_id}")
        seen_ids.add(source.source_id)
        if source.source_kind not in _VALID_SOURCE_KINDS:
            raise P1ManifestValidationError(
                f"source {source.source_id} uses unsupported source_kind={source.source_kind!r}."
            )
        if source.pool_id not in _VALID_POOLS:
            raise P1ManifestValidationError(
                f"source {source.source_id} uses unsupported pool_id={source.pool_id!r}."
            )
        if source.token_weight <= 0.0 or source.record_weight <= 0.0:
            raise P1ManifestValidationError(
                f"source {source.source_id} must use positive token_weight and record_weight."
            )
        token_totals[source.pool_id] += float(source.token_weight)
        record_totals[source.pool_id] += float(source.record_weight)

        if source.source_kind == "hf":
            if not source.hf_path:
                raise P1ManifestValidationError(
                    f"HF source {source.source_id} is missing hf_path."
                )
            if not source.revision:
                raise P1ManifestValidationError(
                    f"HF source {source.source_id} is missing revision."
                )
            revision = str(source.revision).strip()
            if require_committed_revisions:
                if not _is_immutable_revision(revision):
                    raise P1ManifestValidationError(
                        f"HF source {source.source_id} must use an immutable revision, got {revision!r}."
                    )
            else:
                if not (
                    _is_immutable_revision(revision)
                    or revision.startswith(_BOOTSTRAP_ALLOWED_REVISION_PREFIXES)
                ):
                    raise P1ManifestValidationError(
                        f"HF source {source.source_id} must use an immutable revision or bootstrap resolver, got {revision!r}."
                    )
        else:
            if source.revision not in (None, "", "synthetic-v1"):
                raise P1ManifestValidationError(
                    f"Synthetic source {source.source_id} must not declare an HF revision."
                )

    total_token = sum(token_totals.values())
    total_record = sum(record_totals.values())
    if abs(total_token - 1.0) > 1.0e-9:
        raise P1ManifestValidationError(
            f"Manifest token weights must sum to 1.0, got {total_token:.6f}."
        )
    if abs(total_record - 1.0) > 1.0e-9:
        raise P1ManifestValidationError(
            f"Manifest record weights must sum to 1.0, got {total_record:.6f}."
        )
    for pool_id in sorted(_VALID_POOLS):
        if abs(token_totals[pool_id] - _BOOTSTRAP_POOL_TOKEN_WEIGHTS[pool_id]) > 1.0e-9:
            raise P1ManifestValidationError(
                f"Pool {pool_id} token weight must be {_BOOTSTRAP_POOL_TOKEN_WEIGHTS[pool_id]}, got {token_totals[pool_id]:.6f}."
            )
        if abs(record_totals[pool_id] - _BOOTSTRAP_POOL_RECORD_WEIGHTS[pool_id]) > 1.0e-9:
            raise P1ManifestValidationError(
                f"Pool {pool_id} record weight must be {_BOOTSTRAP_POOL_RECORD_WEIGHTS[pool_id]}, got {record_totals[pool_id]:.6f}."
            )
    return manifest


def load_p1_streaming_manifest(path: Path) -> P1StreamingManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise P1ManifestValidationError(f"P1 streaming manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise P1ManifestValidationError(f"Manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(payload, Mapping):
        raise P1ManifestValidationError("Manifest payload must be a JSON object.")
    sources = payload.get("sources", [])
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)):
        raise P1ManifestValidationError("Manifest 'sources' must be an array.")
    manifest = P1StreamingManifest(
        schema=str(payload.get("schema", "")).strip(),
        manifest_id=str(payload.get("manifest_id", "")).strip(),
        profile_id=str(payload.get("profile_id", "")).strip().upper(),
        phase=str(payload.get("phase", "")).strip().upper(),
        commit_posture=str(payload.get("commit_posture", "bootstrap")).strip().lower(),
        default_streaming_mode=str(payload.get("default_streaming_mode", "auto")).strip().lower(),
        sources=tuple(_coerce_source(item) for item in sources if isinstance(item, Mapping)),
    )
    return validate_manifest(manifest)


def load_default_p1_streaming_manifest() -> P1StreamingManifest:
    manifest_path = Path(__file__).resolve().parent / "profiles" / "p1_streaming_manifest_v1.json"
    return load_p1_streaming_manifest(manifest_path)


def manifest_sha256(manifest: P1StreamingManifest) -> str:
    payload = manifest.to_payload()
    payload.pop("manifest_id", None)
    return stable_hash(payload)


def resolve_manifest_revisions(
    manifest: P1StreamingManifest,
    *,
    dataset_sha_resolver: Callable[[str, str | None, str], str],
) -> P1StreamingManifest:
    resolved_sources = []
    for source in manifest.sources:
        if source.source_kind != "hf":
            resolved_sources.append(source)
            continue
        revision = str(source.revision or "").strip()
        if _is_immutable_revision(revision):
            resolved_revision = revision.split("sha:", 1)[-1] if revision.startswith("sha:") else revision
        else:
            if not revision.startswith("resolve:"):
                raise P1ManifestValidationError(
                    f"Cannot resolve non-immutable revision for source {source.source_id}: {revision!r}"
                )
            revision_hint = revision.split("resolve:", 1)[1] or "main"
            resolved_revision = dataset_sha_resolver(str(source.hf_path), source.hf_name, revision_hint)
        resolved_sources.append(
            P1StreamingSource(
                source_id=source.source_id,
                source_kind=source.source_kind,
                pool_id=source.pool_id,
                pool_role=source.pool_role,
                token_weight=source.token_weight,
                record_weight=source.record_weight,
                hf_path=source.hf_path,
                hf_name=source.hf_name,
                split=source.split,
                revision=resolved_revision,
                payload_field=source.payload_field,
                filter_signature=source.filter_signature,
                source_family=source.source_family,
                benchmark_family_refs=source.benchmark_family_refs,
                local_snapshot_pattern=source.local_snapshot_pattern,
                admission_posture=source.admission_posture,
                required=source.required,
                metadata=source.metadata,
            )
        )
    committed = P1StreamingManifest(
        schema=manifest.schema,
        manifest_id=manifest.manifest_id,
        profile_id=manifest.profile_id,
        phase=manifest.phase,
        commit_posture="committed",
        default_streaming_mode=manifest.default_streaming_mode,
        sources=tuple(resolved_sources),
    )
    committed = validate_manifest(committed, require_committed_revisions=True)
    manifest_id = "p1-committed-" + manifest_sha256(committed)[:12]
    committed = P1StreamingManifest(
        schema=committed.schema,
        manifest_id=manifest_id,
        profile_id=committed.profile_id,
        phase=committed.phase,
        commit_posture=committed.commit_posture,
        default_streaming_mode=committed.default_streaming_mode,
        sources=committed.sources,
    )
    return validate_manifest(committed, require_committed_revisions=True)
