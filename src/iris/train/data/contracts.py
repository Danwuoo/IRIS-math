from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

PURE_LM_RATIO_TOTAL = 0.90
REQUIRED_PURE_LM_SOURCE_COUNT = 9
FORBIDDEN_HF_PATHS = {"bigcode/the-stack-v2-dedup"}


@dataclass(frozen=True)
class DatasetSourceSpec:
    source_id: str
    hf_path: str
    hf_name: str | None
    split: str
    revision: str
    text_field: str
    ratio_total: float
    required: bool = True
    local_snapshot_pattern: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def manifest_entry(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "hf_path": self.hf_path,
            "hf_name": self.hf_name,
            "split": self.split,
            "revision": self.revision,
            "text_field": self.text_field,
            "ratio_total": float(self.ratio_total),
            "required": bool(self.required),
            "local_snapshot_pattern": self.local_snapshot_pattern,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PureLMProfile:
    profile_id: str
    version: str
    pure_lm_ratio_total: float
    tokenizer_required: bool
    sources: Tuple[DatasetSourceSpec, ...]

    @property
    def source_by_id(self) -> Dict[str, DatasetSourceSpec]:
        return {source.source_id: source for source in self.sources}

    def stable_payload(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "pure_lm_ratio_total": float(self.pure_lm_ratio_total),
            "tokenizer_required": bool(self.tokenizer_required),
            "sources": [source.manifest_entry() for source in self.sources],
        }


class ProfileValidationError(ValueError):
    pass


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_source(raw_source: Mapping[str, Any]) -> DatasetSourceSpec:
    metadata = raw_source.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return DatasetSourceSpec(
        source_id=str(raw_source.get("source_id", "")).strip(),
        hf_path=str(raw_source.get("hf_path", "")).strip(),
        hf_name=(None if raw_source.get("hf_name") in (None, "") else str(raw_source.get("hf_name"))),
        split=str(raw_source.get("split", "train")).strip() or "train",
        revision=str(raw_source.get("revision", "")).strip(),
        text_field=str(raw_source.get("text_field", "")).strip(),
        ratio_total=float(raw_source.get("ratio_total", 0.0)),
        required=bool(raw_source.get("required", True)),
        local_snapshot_pattern=(
            None
            if raw_source.get("local_snapshot_pattern") in (None, "")
            else str(raw_source.get("local_snapshot_pattern"))
        ),
        metadata={str(key): value for key, value in metadata.items()},
    )


def validate_profile(profile: PureLMProfile) -> PureLMProfile:
    if not profile.profile_id:
        raise ProfileValidationError("Profile id is required.")
    if not profile.version:
        raise ProfileValidationError("Profile version is required.")
    if len(profile.sources) != REQUIRED_PURE_LM_SOURCE_COUNT:
        raise ProfileValidationError(
            f"Pure LM profile must contain {REQUIRED_PURE_LM_SOURCE_COUNT} sources, got {len(profile.sources)}."
        )

    source_ids = [source.source_id for source in profile.sources]
    if len(set(source_ids)) != len(source_ids):
        raise ProfileValidationError("source_id values must be unique.")

    total_ratio = sum(float(source.ratio_total) for source in profile.sources)
    if abs(total_ratio - float(profile.pure_lm_ratio_total)) > 1e-9:
        raise ProfileValidationError(
            f"Profile source ratios must sum to pure_lm_ratio_total ({profile.pure_lm_ratio_total}), got {total_ratio}."
        )
    if abs(float(profile.pure_lm_ratio_total) - PURE_LM_RATIO_TOTAL) > 1e-9:
        raise ProfileValidationError(
            f"pure_lm_ratio_total must remain {PURE_LM_RATIO_TOTAL}, got {profile.pure_lm_ratio_total}."
        )

    for source in profile.sources:
        if not source.source_id:
            raise ProfileValidationError("source_id cannot be empty.")
        if source.hf_path in FORBIDDEN_HF_PATHS:
            raise ProfileValidationError(
                f"Forbidden dataset configured in Pure LM profile: {source.hf_path}."
            )
        if source.ratio_total <= 0.0:
            raise ProfileValidationError(f"Source {source.source_id} must have positive ratio_total.")
        if not source.hf_path:
            raise ProfileValidationError(f"Source {source.source_id} missing hf_path.")
        if not source.revision:
            raise ProfileValidationError(f"Source {source.source_id} missing revision lock.")
        if not source.text_field:
            raise ProfileValidationError(f"Source {source.source_id} missing text_field.")

    return profile


def load_profile(path: Path) -> PureLMProfile:
    profile_path = Path(path)
    if not profile_path.exists():
        raise ProfileValidationError(f"Pure LM profile not found: {profile_path}")

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ProfileValidationError(f"Pure LM profile is not valid JSON: {profile_path}") from error

    if not isinstance(payload, Mapping):
        raise ProfileValidationError("Pure LM profile must be a JSON object.")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes, bytearray)):
        raise ProfileValidationError("Pure LM profile 'sources' must be an array.")

    profile = PureLMProfile(
        profile_id=str(payload.get("profile_id", "")).strip(),
        version=str(payload.get("version", "")).strip(),
        pure_lm_ratio_total=float(payload.get("pure_lm_ratio_total", PURE_LM_RATIO_TOTAL)),
        tokenizer_required=bool(payload.get("tokenizer_required", True)),
        sources=tuple(_coerce_source(raw_source) for raw_source in raw_sources if isinstance(raw_source, Mapping)),
    )
    return validate_profile(profile)


def load_default_pure_lm_profile() -> PureLMProfile:
    profile_path = Path(__file__).resolve().parent / "profiles" / "pure_lm_90_v1.json"
    return load_profile(profile_path)


def sources_manifest_sha256(profile: PureLMProfile) -> str:
    manifest_payload = {
        "profile_id": profile.profile_id,
        "version": profile.version,
        "sources": [source.manifest_entry() for source in profile.sources],
    }
    return _stable_sha256(manifest_payload)


def profile_sha256(profile: PureLMProfile) -> str:
    return _stable_sha256(profile.stable_payload())
