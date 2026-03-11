from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


class DocumentRecordValidationError(ValueError):
    pass


def _require_text(field_name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise DocumentRecordValidationError(f"{field_name} is required.")
    return text


def _coerce_sequence(field_name: str, value: Any) -> Tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DocumentRecordValidationError(f"{field_name} must be an array.")
    return tuple(value)


@dataclass(frozen=True)
class MathDocumentSource:
    schema: str
    source_id: str
    source_format: str
    source_uri_or_snapshot: str
    source_hash: str
    modality: str
    license_state: str
    parser_eligibility: str


@dataclass(frozen=True)
class MathDocumentRecord:
    schema: str
    record_id: str
    source_id: str
    source_format: str
    source_uri_or_snapshot: str
    content_sha256: str
    pages: Tuple[Any, ...]
    reading_order: Tuple[Any, ...]
    anchor_index: Tuple[Any, ...]
    clean_text_blocks: Tuple[Any, ...]
    formula_blocks: Tuple[Any, ...]
    table_blocks: Tuple[Any, ...]
    diagram_regions: Tuple[Any, ...]
    section_structure: Tuple[Any, ...]
    math_semantic_units: Tuple[Any, ...]
    cross_reference_edges: Tuple[Any, ...]
    parse_confidence: float
    semantic_unit_confidence: float
    bbox_coverage_score: float
    record_quality_flags: Tuple[Any, ...]
    parser_provenance_id: str
    parser_provenance_refs: Dict[str, str]
    ocr_layout_extractor_version: str
    formula_parser_version: str
    semantic_unit_typer_version: str
    parse_config_fingerprint: str
    unresolved_region_ratio: float


@dataclass(frozen=True)
class MathDocumentProjection:
    schema: str
    projection_id: str
    record_id: str
    projection_kind: str
    state_ir_slot_targets: Tuple[str, ...]
    anchor_refs: Tuple[Any, ...]
    payload: Dict[str, Any]


def validate_math_document_source(payload: Mapping[str, Any]) -> MathDocumentSource:
    schema = _require_text("math_document_source.schema", payload.get("schema", ""))
    if schema != "math_document_source/v1":
        raise DocumentRecordValidationError("math_document_source schema must be math_document_source/v1.")
    return MathDocumentSource(
        schema=schema,
        source_id=_require_text("math_document_source.source_id", payload.get("source_id", "")),
        source_format=_require_text("math_document_source.source_format", payload.get("source_format", "")),
        source_uri_or_snapshot=_require_text(
            "math_document_source.source_uri_or_snapshot",
            payload.get("source_uri_or_snapshot", ""),
        ),
        source_hash=_require_text("math_document_source.source_hash", payload.get("source_hash", "")),
        modality=_require_text("math_document_source.modality", payload.get("modality", "")),
        license_state=_require_text("math_document_source.license_state", payload.get("license_state", "")),
        parser_eligibility=_require_text(
            "math_document_source.parser_eligibility",
            payload.get("parser_eligibility", ""),
        ),
    )


def validate_math_document_record(payload: Mapping[str, Any]) -> MathDocumentRecord:
    schema = _require_text("math_document_record.schema", payload.get("schema", ""))
    if schema != "math_document_record/v1":
        raise DocumentRecordValidationError("math_document_record schema must be math_document_record/v1.")
    refs = dict(payload.get("parser_provenance_refs", {}))
    normalized_refs = {
        key: str(refs.get(key, "not_applicable")).strip() or "not_applicable"
        for key in (
            "layout_parser_manifest_id",
            "ocr_manifest_id",
            "formula_parser_manifest_id",
            "semantic_unit_typer_manifest_id",
        )
    }
    return MathDocumentRecord(
        schema=schema,
        record_id=_require_text("math_document_record.record_id", payload.get("record_id", "")),
        source_id=_require_text("math_document_record.source_id", payload.get("source_id", "")),
        source_format=_require_text("math_document_record.source_format", payload.get("source_format", "")),
        source_uri_or_snapshot=_require_text(
            "math_document_record.source_uri_or_snapshot",
            payload.get("source_uri_or_snapshot", ""),
        ),
        content_sha256=_require_text(
            "math_document_record.content_sha256",
            payload.get("content_sha256", ""),
        ),
        pages=_coerce_sequence("math_document_record.pages", payload.get("pages", [])),
        reading_order=_coerce_sequence("math_document_record.reading_order", payload.get("reading_order", [])),
        anchor_index=_coerce_sequence("math_document_record.anchor_index", payload.get("anchor_index", [])),
        clean_text_blocks=_coerce_sequence(
            "math_document_record.clean_text_blocks",
            payload.get("clean_text_blocks", []),
        ),
        formula_blocks=_coerce_sequence("math_document_record.formula_blocks", payload.get("formula_blocks", [])),
        table_blocks=_coerce_sequence("math_document_record.table_blocks", payload.get("table_blocks", [])),
        diagram_regions=_coerce_sequence(
            "math_document_record.diagram_regions",
            payload.get("diagram_regions", []),
        ),
        section_structure=_coerce_sequence(
            "math_document_record.section_structure",
            payload.get("section_structure", []),
        ),
        math_semantic_units=_coerce_sequence(
            "math_document_record.math_semantic_units",
            payload.get("math_semantic_units", []),
        ),
        cross_reference_edges=_coerce_sequence(
            "math_document_record.cross_reference_edges",
            payload.get("cross_reference_edges", []),
        ),
        parse_confidence=float(payload.get("parse_confidence", 0.0)),
        semantic_unit_confidence=float(payload.get("semantic_unit_confidence", 0.0)),
        bbox_coverage_score=float(payload.get("bbox_coverage_score", 0.0)),
        record_quality_flags=_coerce_sequence(
            "math_document_record.record_quality_flags",
            payload.get("record_quality_flags", []),
        ),
        parser_provenance_id=_require_text(
            "math_document_record.parser_provenance_id",
            payload.get("parser_provenance_id", ""),
        ),
        parser_provenance_refs=normalized_refs,
        ocr_layout_extractor_version=str(payload.get("ocr_layout_extractor_version", "not_applicable")),
        formula_parser_version=str(payload.get("formula_parser_version", "not_applicable")),
        semantic_unit_typer_version=str(payload.get("semantic_unit_typer_version", "not_applicable")),
        parse_config_fingerprint=_require_text(
            "math_document_record.parse_config_fingerprint",
            payload.get("parse_config_fingerprint", ""),
        ),
        unresolved_region_ratio=float(payload.get("unresolved_region_ratio", 0.0)),
    )


def validate_math_document_projection(payload: Mapping[str, Any]) -> MathDocumentProjection:
    schema = _require_text("math_document_projection.schema", payload.get("schema", ""))
    if schema != "math_document_projection/v1":
        raise DocumentRecordValidationError(
            "math_document_projection schema must be math_document_projection/v1."
        )
    targets = tuple(str(item) for item in _coerce_sequence("math_document_projection.state_ir_slot_targets", payload.get("state_ir_slot_targets", [])))
    if not targets:
        raise DocumentRecordValidationError("math_document_projection.state_ir_slot_targets must not be empty.")
    return MathDocumentProjection(
        schema=schema,
        projection_id=_require_text(
            "math_document_projection.projection_id",
            payload.get("projection_id", ""),
        ),
        record_id=_require_text("math_document_projection.record_id", payload.get("record_id", "")),
        projection_kind=_require_text(
            "math_document_projection.projection_kind",
            payload.get("projection_kind", ""),
        ),
        state_ir_slot_targets=targets,
        anchor_refs=_coerce_sequence("math_document_projection.anchor_refs", payload.get("anchor_refs", [])),
        payload=dict(payload.get("payload", {})),
    )


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise DocumentRecordValidationError(f"Document payload must be a JSON object: {path}")
    return dict(payload)


def load_math_document_source(path: Path) -> MathDocumentSource:
    return validate_math_document_source(_load_json(path))


def load_math_document_record(path: Path) -> MathDocumentRecord:
    return validate_math_document_record(_load_json(path))


def load_math_document_projection(path: Path) -> MathDocumentProjection:
    return validate_math_document_projection(_load_json(path))
