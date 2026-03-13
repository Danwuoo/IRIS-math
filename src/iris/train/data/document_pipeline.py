from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from xml.etree import ElementTree as ET

from .document_records import (
    MathDocumentProjection,
    MathDocumentRecord,
    MathDocumentSource,
    validate_math_document_projection,
    validate_math_document_record,
    validate_math_document_source,
)

_DOCX_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class DocumentPipelineError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentPipelineBundle:
    source: MathDocumentSource
    record: MathDocumentRecord
    projection: MathDocumentProjection


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(encoded)


def _anchor(
    anchor_id: str,
    page_id: str,
    role: str,
    *,
    text: str | None = None,
    bbox: Sequence[float] | None = None,
    parent_anchor_id: str | None = None,
    confidence: float = 1.0,
    parser_provenance_id: str = "math-doc-pipeline-v1",
    reading_order_index: int = 0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "anchor_id": anchor_id,
        "page_or_canvas_id": page_id,
        "structural_role": role,
        "reading_order_index": int(reading_order_index),
        "confidence": float(confidence),
        "parser_provenance_id": parser_provenance_id,
    }
    if bbox is not None:
        payload["bbox"] = [float(value) for value in bbox]
    if text is not None:
        payload["span_start"] = 0
        payload["span_end"] = len(text)
    if parent_anchor_id:
        payload["parent_anchor_id"] = str(parent_anchor_id)
    return payload


def _projection_payload_from_record(record: MathDocumentRecord) -> Dict[str, Any]:
    joined_text = " ".join(
        str(block.get("text", "")).strip()
        for block in record.clean_text_blocks
        if isinstance(block, Mapping)
    ).strip()
    symbol_matches = sorted(set(re.findall(r"[A-Za-z]+", joined_text)))
    candidate_relations: List[Dict[str, Any]] = []
    for block in record.formula_blocks:
        if not isinstance(block, Mapping):
            continue
        latex = str(block.get("latex", "")).strip()
        if "=" in latex:
            left, right = [part.strip() for part in latex.split("=", 1)]
            candidate_relations.append(
                {
                    "relation_type": "equality",
                    "arguments": [left or "lhs", right or "rhs"],
                    "relation_status": "candidate",
                    "anchor_id": str(block.get("anchor_id", "")),
                }
            )
    if not candidate_relations and symbol_matches:
        candidate_relations.append(
            {
                "relation_type": "mentions",
                "arguments": symbol_matches[:2] or ["document_symbol"],
                "relation_status": "candidate",
                "anchor_id": str(record.anchor_index[0].get("anchor_id", "")) if record.anchor_index else "",
            }
        )
    return {
        "task_type": "prove" if candidate_relations else "formalize",
        "target_text": joined_text or "bootstrap document target",
        "required_output": {
            "output_kind": "proof",
            "answer_channel": "structured_object",
            "formality_level": "semi-formal" if record.source_format == "DOCX" else "informal",
            "verifier_mode": "proof_gap_plus_local_validity",
        },
        "symbols": [
            {
                "surface_form": symbol,
                "entity_kind": "symbol",
                "binding_state": "unresolved",
                "type_status": "unknown",
                "anchor_id": str(record.anchor_index[min(index, len(record.anchor_index) - 1)].get("anchor_id", ""))
                if record.anchor_index
                else "",
            }
            for index, symbol in enumerate(symbol_matches[:6])
        ],
        "candidate_relations": candidate_relations,
        "parse_confidence": float(record.parse_confidence),
        "document_grounding_score": float(record.bbox_coverage_score),
        "quality_flags": list(record.record_quality_flags),
    }


def _build_projection(record: MathDocumentRecord) -> MathDocumentProjection:
    return validate_math_document_projection(
        {
            "schema": "math_document_projection/v1",
            "projection_id": f"{record.record_id}-projection",
            "record_id": record.record_id,
            "projection_kind": "state_ir_seed",
            "state_ir_slot_targets": ["PF", "SY", "CG"],
            "anchor_refs": [item.get("anchor_id", "") for item in record.anchor_index if isinstance(item, Mapping)],
            "payload": _projection_payload_from_record(record),
        }
    )


def _build_pdf_record(
    source: MathDocumentSource,
    sidecar_payload: Mapping[str, Any],
) -> MathDocumentRecord:
    record_payload = dict(sidecar_payload)
    record_payload.update(
        {
            "schema": "math_document_record/v1",
            "record_id": str(record_payload.get("record_id", f"{source.source_id}-record")),
            "source_id": source.source_id,
            "source_format": source.source_format,
            "source_uri_or_snapshot": source.source_uri_or_snapshot,
            "content_sha256": _content_hash(record_payload),
            "parser_provenance_id": str(record_payload.get("parser_provenance_id", "math-doc-pipeline-v1")),
            "parser_provenance_refs": dict(record_payload.get("parser_provenance_refs", {})),
            "parse_config_fingerprint": str(
                record_payload.get("parse_config_fingerprint", "cfg-doc-pipeline-bootstrap")
            ),
        }
    )
    return validate_math_document_record(record_payload)


def _docx_text_runs(document_xml: bytes) -> List[str]:
    root = ET.fromstring(document_xml)
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", _DOCX_NAMESPACE):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", _DOCX_NAMESPACE)]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return paragraphs


def _build_docx_record(source: MathDocumentSource, source_path: Path) -> MathDocumentRecord:
    with zipfile.ZipFile(source_path, "r") as archive:
        try:
            document_xml = archive.read("word/document.xml")
        except KeyError as error:
            raise DocumentPipelineError("DOCX fixture is missing word/document.xml.") from error
    paragraphs = _docx_text_runs(document_xml)
    if not paragraphs:
        raise DocumentPipelineError("DOCX fixture did not yield any paragraph text.")

    page_id = "page-1"
    reading_order: List[Dict[str, Any]] = []
    anchor_index: List[Dict[str, Any]] = []
    clean_text_blocks: List[Dict[str, Any]] = []
    formula_blocks: List[Dict[str, Any]] = []
    section_structure: List[Dict[str, Any]] = []
    math_semantic_units: List[Dict[str, Any]] = []

    for index, paragraph in enumerate(paragraphs):
        anchor_id = f"anchor-{index + 1}"
        anchor = _anchor(
            anchor_id,
            page_id,
            "paragraph",
            text=paragraph,
            bbox=(0.0, 12.0 * index, 100.0, 12.0 * index + 10.0),
            reading_order_index=index,
        )
        reading_order.append({"anchor_id": anchor_id})
        anchor_index.append({"anchor_id": anchor_id, "page_id": page_id, **anchor})
        clean_text_blocks.append({"anchor_id": anchor_id, "text": paragraph})
        if "=" in paragraph:
            formula_blocks.append({"anchor_id": f"formula-{index + 1}", "latex": paragraph})
        if index == 0:
            section_structure.append({"section_id": "sec-1", "title": paragraph})
        unit_type = "proof" if any(token in paragraph.lower() for token in ("proof", "therefore")) else "theorem"
        math_semantic_units.append({"unit_id": f"unit-{index + 1}", "unit_type": unit_type})

    record_payload = {
        "schema": "math_document_record/v1",
        "record_id": f"{source.source_id}-record",
        "source_id": source.source_id,
        "source_format": source.source_format,
        "source_uri_or_snapshot": source.source_uri_or_snapshot,
        "content_sha256": _content_hash({"paragraphs": paragraphs}),
        "pages": [{"page_id": page_id}],
        "reading_order": reading_order,
        "anchor_index": anchor_index,
        "clean_text_blocks": clean_text_blocks,
        "formula_blocks": formula_blocks,
        "table_blocks": [],
        "diagram_regions": [],
        "section_structure": section_structure,
        "math_semantic_units": math_semantic_units,
        "cross_reference_edges": [],
        "parse_confidence": 0.94,
        "semantic_unit_confidence": 0.92,
        "bbox_coverage_score": 0.93,
        "record_quality_flags": ["anchors_complete", "docx_zip_xml_extracted"],
        "parser_provenance_id": "math-doc-pipeline-v1",
        "parser_provenance_refs": {
            "layout_parser_manifest_id": "layout-parser-v1",
            "ocr_manifest_id": "not_applicable",
            "formula_parser_manifest_id": "formula-parser-v1",
            "semantic_unit_typer_manifest_id": "semantic-unit-typer-v1"
        },
        "ocr_layout_extractor_version": "docx-xml-bootstrap-v1",
        "formula_parser_version": "bootstrap-v1",
        "semantic_unit_typer_version": "bootstrap-v1",
        "parse_config_fingerprint": "cfg-doc-pipeline-bootstrap",
        "unresolved_region_ratio": 0.0
    }
    return validate_math_document_record(record_payload)


def build_document_pipeline_bundle(
    source_path: Path,
    *,
    sidecar_path: Path | None = None,
    source_uri_or_snapshot: str | None = None,
) -> DocumentPipelineBundle:
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise DocumentPipelineError("Only bootstrap PDF and DOCX sources are supported in the P1 pipeline.")
    source_format = suffix.lstrip(".").upper()
    source = validate_math_document_source(
        {
            "schema": "math_document_source/v1",
            "source_id": source_path.stem,
            "source_format": source_format,
            "source_uri_or_snapshot": source_uri_or_snapshot or str(source_path),
            "source_hash": _source_hash(source_path),
            "modality": "document",
            "license_state": "internal-eval-approved",
            "parser_eligibility": "eligible",
        }
    )
    if source_format == "PDF":
        if sidecar_path is None:
            raise DocumentPipelineError("Bootstrap PDF ingestion requires a checked-in extraction sidecar.")
        sidecar_payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8-sig"))
        if not isinstance(sidecar_payload, Mapping):
            raise DocumentPipelineError("PDF sidecar payload must be a JSON object.")
        record = _build_pdf_record(source, sidecar_payload)
    else:
        record = _build_docx_record(source, source_path)
    projection = _build_projection(record)
    return DocumentPipelineBundle(source=source, record=record, projection=projection)
