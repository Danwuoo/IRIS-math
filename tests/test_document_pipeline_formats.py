from __future__ import annotations

from pathlib import Path

from iris.train import build_document_pipeline_bundle


def _asset_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "p1_phase_de" / "assets"


def test_document_pipeline_accepts_scanned_note_sidecar() -> None:
    root = _asset_root()
    bundle = build_document_pipeline_bundle(
        root / "ocr_identity_note.png",
        sidecar_path=root / "ocr_identity_note.png.sidecar.json",
        source_format_override="scanned_note",
    )
    assert bundle.source.source_format == "SCANNED_NOTE"
    assert bundle.record.source_format == "SCANNED_NOTE"
    assert bundle.record.parser_provenance_refs["ocr_manifest_id"] == "ocr-parser-v1"
    assert bundle.record.unresolved_region_ratio == 0.08
    assert len(bundle.projection.anchor_refs) == 2


def test_document_pipeline_accepts_diagram_sidecar_without_override() -> None:
    root = _asset_root()
    bundle = build_document_pipeline_bundle(
        root / "diagram_identity.png",
        sidecar_path=root / "diagram_identity.png.sidecar.json",
    )
    assert bundle.source.source_format == "DIAGRAM"
    assert bundle.record.diagram_regions
    assert bundle.projection.payload["candidate_relations"]


def test_document_pipeline_accepts_image_sidecar_without_override() -> None:
    root = _asset_root()
    bundle = build_document_pipeline_bundle(
        root / "formula_crop.jpg",
        sidecar_path=root / "formula_crop.jpg.sidecar.json",
    )
    assert bundle.source.source_format == "IMAGE"
    assert bundle.record.record_quality_flags
    assert bundle.record.unresolved_region_ratio == 0.05
