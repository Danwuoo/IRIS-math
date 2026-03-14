from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..runtime import TASK_FAMILY_IDS

_EVAL_PARTITIONS: Tuple[str, ...] = ("diagnostic", "strict_holdout")


class MathNativeFixtureValidationError(ValueError):
    pass


def _require_text(field_name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise MathNativeFixtureValidationError(f"{field_name} is required.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _require_string_array(field_name: str, value: Any) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MathNativeFixtureValidationError(f"{field_name} must be an array.")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if not items:
        raise MathNativeFixtureValidationError(f"{field_name} must not be empty.")
    return items


def _partition(field_name: str, value: Any) -> str:
    partition = _require_text(field_name, value).lower()
    if partition not in _EVAL_PARTITIONS:
        raise MathNativeFixtureValidationError(
            f"{field_name} must be one of: {', '.join(_EVAL_PARTITIONS)}."
        )
    return partition


def _task_family(field_name: str, value: Any) -> str:
    family = _require_text(field_name, value)
    if family not in set(TASK_FAMILY_IDS):
        raise MathNativeFixtureValidationError(
            f"{field_name} must be one of: {', '.join(TASK_FAMILY_IDS)}."
        )
    return family


def _resolve_relative(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise MathNativeFixtureValidationError(f"Fixture payload must be a JSON object: {path}")
    return dict(payload)


@dataclass(frozen=True)
class DocumentEvalFixture:
    schema: str
    fixture_id: str
    fixture_path: Path
    document_path: str
    sidecar_path: str | None
    source_format_override: str | None
    eval_partition: str
    task_family: str
    benchmark_family_policy_ref: str | None
    expected_slot_targets: Tuple[str, ...]
    expected_min_parse_completeness: float
    expected_min_document_grounding_score: float
    expected_min_anchor_count: int
    expected_runtime_status: str
    expected_adjudication_status: str
    paired_group_id: str | None
    pair_variant_id: str | None
    notes: str | None

    def resolved_document_path(self) -> Path:
        return _resolve_relative(self.fixture_path.parent, self.document_path)

    def resolved_sidecar_path(self) -> Path | None:
        if self.sidecar_path is None:
            return None
        return _resolve_relative(self.fixture_path.parent, self.sidecar_path)


@dataclass(frozen=True)
class ProofEvalFixture:
    schema: str
    fixture_id: str
    fixture_path: Path
    proof_seed: str
    task_type: str
    target_spec: str
    required_output: Dict[str, Any]
    document_fixture_id: str | None
    eval_partition: str
    task_family: str
    benchmark_family_policy_ref: str | None
    item_policy_id: str | None
    expected_task_adjudication_policy_id: str
    required_evidence_classes: Tuple[str, ...]
    expected_runtime_status: str
    expected_adjudication_status: str
    expected_formal_bridge_status: str | None
    min_validity_score: float
    max_false_accept_rate: float
    max_calibration_error: float
    min_verifier_coverage: float
    notes: str | None


def validate_document_eval_fixture(
    payload: Mapping[str, Any],
    *,
    fixture_path: Path,
) -> DocumentEvalFixture:
    schema = _require_text("document_eval_fixture.schema", payload.get("schema", ""))
    if schema != "iris.math_native.document_eval_fixture/v1":
        raise MathNativeFixtureValidationError(
            "document_eval_fixture schema must be iris.math_native.document_eval_fixture/v1."
        )
    return DocumentEvalFixture(
        schema=schema,
        fixture_id=_require_text("document_eval_fixture.fixture_id", payload.get("fixture_id", "")),
        fixture_path=Path(fixture_path).resolve(),
        document_path=_require_text("document_eval_fixture.document_path", payload.get("document_path", "")),
        sidecar_path=_optional_text(payload.get("sidecar_path")),
        source_format_override=_optional_text(payload.get("source_format_override")),
        eval_partition=_partition("document_eval_fixture.eval_partition", payload.get("eval_partition", "diagnostic")),
        task_family=_task_family("document_eval_fixture.task_family", payload.get("task_family", "")),
        benchmark_family_policy_ref=_optional_text(payload.get("benchmark_family_policy_ref")),
        expected_slot_targets=_require_string_array(
            "document_eval_fixture.expected_slot_targets",
            payload.get("expected_slot_targets", ()),
        ),
        expected_min_parse_completeness=float(payload.get("expected_min_parse_completeness", 0.0)),
        expected_min_document_grounding_score=float(payload.get("expected_min_document_grounding_score", 0.0)),
        expected_min_anchor_count=int(payload.get("expected_min_anchor_count", 1)),
        expected_runtime_status=_require_text(
            "document_eval_fixture.expected_runtime_status",
            payload.get("expected_runtime_status", ""),
        ),
        expected_adjudication_status=_require_text(
            "document_eval_fixture.expected_adjudication_status",
            payload.get("expected_adjudication_status", ""),
        ),
        paired_group_id=_optional_text(payload.get("paired_group_id")),
        pair_variant_id=_optional_text(payload.get("pair_variant_id")),
        notes=_optional_text(payload.get("notes")),
    )


def validate_proof_eval_fixture(
    payload: Mapping[str, Any],
    *,
    fixture_path: Path,
) -> ProofEvalFixture:
    schema = _require_text("proof_eval_fixture.schema", payload.get("schema", ""))
    if schema != "iris.math_native.proof_eval_fixture/v1":
        raise MathNativeFixtureValidationError(
            "proof_eval_fixture schema must be iris.math_native.proof_eval_fixture/v1."
        )
    required_output = payload.get("required_output", {})
    if not isinstance(required_output, Mapping):
        raise MathNativeFixtureValidationError("proof_eval_fixture.required_output must be an object.")
    return ProofEvalFixture(
        schema=schema,
        fixture_id=_require_text("proof_eval_fixture.fixture_id", payload.get("fixture_id", "")),
        fixture_path=Path(fixture_path).resolve(),
        proof_seed=_require_text("proof_eval_fixture.proof_seed", payload.get("proof_seed", "")),
        task_type=_require_text("proof_eval_fixture.task_type", payload.get("task_type", "")),
        target_spec=_require_text("proof_eval_fixture.target_spec", payload.get("target_spec", "")),
        required_output=dict(required_output),
        document_fixture_id=_optional_text(payload.get("document_fixture_id")),
        eval_partition=_partition("proof_eval_fixture.eval_partition", payload.get("eval_partition", "diagnostic")),
        task_family=_task_family("proof_eval_fixture.task_family", payload.get("task_family", "")),
        benchmark_family_policy_ref=_optional_text(payload.get("benchmark_family_policy_ref")),
        item_policy_id=_optional_text(payload.get("item_policy_id")),
        expected_task_adjudication_policy_id=_require_text(
            "proof_eval_fixture.expected_task_adjudication_policy_id",
            payload.get("expected_task_adjudication_policy_id", ""),
        ),
        required_evidence_classes=_require_string_array(
            "proof_eval_fixture.required_evidence_classes",
            payload.get("required_evidence_classes", ()),
        ),
        expected_runtime_status=_require_text(
            "proof_eval_fixture.expected_runtime_status",
            payload.get("expected_runtime_status", ""),
        ),
        expected_adjudication_status=_require_text(
            "proof_eval_fixture.expected_adjudication_status",
            payload.get("expected_adjudication_status", ""),
        ),
        expected_formal_bridge_status=_optional_text(payload.get("expected_formal_bridge_status")),
        min_validity_score=float(payload.get("min_validity_score", 0.0)),
        max_false_accept_rate=float(payload.get("max_false_accept_rate", 1.0)),
        max_calibration_error=float(payload.get("max_calibration_error", 1.0)),
        min_verifier_coverage=float(payload.get("min_verifier_coverage", 0.0)),
        notes=_optional_text(payload.get("notes")),
    )


def load_document_eval_fixture(path: Path) -> DocumentEvalFixture:
    fixture_path = Path(path)
    return validate_document_eval_fixture(_load_json(fixture_path), fixture_path=fixture_path)


def load_proof_eval_fixture(path: Path) -> ProofEvalFixture:
    fixture_path = Path(path)
    return validate_proof_eval_fixture(_load_json(fixture_path), fixture_path=fixture_path)


def load_document_eval_fixtures(
    root: Path,
    *,
    eval_partition: str | None = None,
) -> Tuple[DocumentEvalFixture, ...]:
    root_path = Path(root)
    if not root_path.exists():
        return ()
    fixture_paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.json"))
    fixtures = []
    for fixture_path in fixture_paths:
        payload = _load_json(fixture_path)
        if str(payload.get("schema", "")).strip() != "iris.math_native.document_eval_fixture/v1":
            continue
        fixture = validate_document_eval_fixture(payload, fixture_path=fixture_path)
        if eval_partition and fixture.eval_partition != eval_partition:
            continue
        fixtures.append(fixture)
    return tuple(sorted(fixtures, key=lambda item: item.fixture_id))


def load_proof_eval_fixtures(
    root: Path,
    *,
    eval_partition: str | None = None,
) -> Tuple[ProofEvalFixture, ...]:
    root_path = Path(root)
    if not root_path.exists():
        return ()
    fixture_paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.json"))
    fixtures = []
    for fixture_path in fixture_paths:
        payload = _load_json(fixture_path)
        if str(payload.get("schema", "")).strip() != "iris.math_native.proof_eval_fixture/v1":
            continue
        fixture = validate_proof_eval_fixture(payload, fixture_path=fixture_path)
        if eval_partition and fixture.eval_partition != eval_partition:
            continue
        fixtures.append(fixture)
    return tuple(sorted(fixtures, key=lambda item: item.fixture_id))
