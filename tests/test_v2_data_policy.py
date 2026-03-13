from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.train.data import (
    PolicyValidationError,
    build_document_slice_id,
    load_default_policy_bundle,
    load_policy_bundle,
    load_policy_bundle_for_profile,
    load_policy_bundle_for_profile_phase,
    validate_math_document_projection,
    validate_math_document_record,
    validate_math_document_source,
)


def test_default_policy_bundle_resolves_bootstrap_v2_objects() -> None:
    bundle = load_default_policy_bundle()
    assert bundle.schema == "iris.data_policy_bundle/v1"
    assert bundle.data_realization_policy.data_realization_policy_id == "p1-bootstrap-b-v2"
    assert bundle.data_realization_policy.profile_id == "P1"
    assert bundle.data_realization_policy.pool_allocations["D"].token_weight == 35.0
    assert "aimo-v1" in bundle.benchmark_family_policies
    assert "verifier-stack-v1" in bundle.provenance_manifests
    assert bundle.benchmark_family_policies["frontiermath-original-v1"].allowed_tiers == ("Tier 3",)


def test_profile_bundle_registry_loads_p2_and_p3() -> None:
    p2 = load_policy_bundle_for_profile("P2")
    p3 = load_policy_bundle_for_profile("P3")

    assert p2.data_realization_policy.data_realization_policy_id == "p2-bootstrap-c-v1"
    assert p2.data_realization_policy.pool_allocations["D"].allowed_roles == (
        "core",
        "auxiliary",
        "eval_only",
    )
    assert p2.data_realization_policy.tier1_global_cap["token_cap"] == 8.0

    assert p3.data_realization_policy.data_realization_policy_id == "p3-bootstrap-d-v1"
    assert p3.data_realization_policy.profile_id == "P3"
    assert p3.data_realization_policy.phase == "D"
    assert p3.data_realization_policy.pool_allocations["C"].record_weight == 30.0
    assert p3.data_realization_policy.source_family_allowlists["D"]["core"] == ("PDF", "DOCX", "image")


def test_p1_phase_specific_bundle_allows_zero_tier1_cap_in_phase_a() -> None:
    phase_a = load_policy_bundle_for_profile_phase("P1", "A")
    phase_c = load_policy_bundle_for_profile_phase("P1", "C")

    assert phase_a.data_realization_policy.data_realization_policy_id == "p1-bootstrap-a-v1"
    assert phase_a.data_realization_policy.tier1_global_cap == {"token_cap": 0.0, "record_cap": 0.0}
    assert phase_c.data_realization_policy.data_realization_policy_id == "p1-bootstrap-c-v1"


def test_document_slice_id_is_stable_and_tracks_provenance_changes() -> None:
    shared = dict(
        run_id="run-a",
        segment_id=3,
        micro_step_idx=7,
        data_seed=17,
        math_document_record_id="doc-001",
        data_realization_policy_id="p1-bootstrap-b-v2",
        decontam_policy_id="global-decontam-v2",
        parser_provenance_id="math-doc-pipeline-v1",
        parser_provenance_refs={
            "layout_parser_manifest_id": "layout-parser-v1",
            "ocr_manifest_id": "ocr-parser-v1",
            "formula_parser_manifest_id": "formula-parser-v1",
            "semantic_unit_typer_manifest_id": "semantic-unit-typer-v1",
        },
        parse_config_fingerprint="cfg-doc-pipeline-bootstrap",
        ocr_layout_extractor_version="bootstrap-v1",
        formula_parser_version="bootstrap-v1",
        semantic_unit_typer_version="bootstrap-v1",
        formalizer_provenance_id="formalizer-skeleton-v1",
        verifier_provenance_id="verifier-stack-v1",
        verifier_build_id="verifier-bootstrap-001",
    )
    left = build_document_slice_id(**shared)
    right = build_document_slice_id(**shared)
    changed = build_document_slice_id(**{**shared, "formula_parser_version": "bootstrap-v2"})

    assert left == right
    assert left != changed


def test_policy_bundle_rejects_unresolved_benchmark_ref(tmp_path: Path) -> None:
    bundle = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src"
            / "iris"
            / "train"
            / "data"
            / "profiles"
            / "p1_bootstrap_policy_bundle_v1.json"
        ).read_text(encoding="utf-8")
    )
    bundle["data_realization_policy"]["benchmark_family_policy_refs"] = ["missing-family"]
    bundle_path = tmp_path / "bad_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(PolicyValidationError):
        load_policy_bundle(bundle_path)


def test_math_document_contract_objects_validate() -> None:
    source = validate_math_document_source(
        {
            "schema": "math_document_source/v1",
            "source_id": "src-001",
            "source_format": "PDF",
            "source_uri_or_snapshot": "snapshot://pdf-001",
            "source_hash": "sha256:abc",
            "modality": "document",
            "license_state": "internal-eval-approved",
            "parser_eligibility": "eligible",
        }
    )
    record = validate_math_document_record(
        {
            "schema": "math_document_record/v1",
            "record_id": "doc-001",
            "source_id": source.source_id,
            "source_format": source.source_format,
            "source_uri_or_snapshot": source.source_uri_or_snapshot,
            "content_sha256": "sha256:def",
            "pages": [{"page_id": "p1"}],
            "reading_order": [{"anchor_id": "anchor-1"}],
            "anchor_index": [{"anchor_id": "anchor-1", "page_id": "p1"}],
            "clean_text_blocks": [{"anchor_id": "anchor-1", "text": "Lemma 1"}],
            "formula_blocks": [{"anchor_id": "anchor-2", "latex": "x=x"}],
            "table_blocks": [],
            "diagram_regions": [],
            "section_structure": [{"section_id": "sec-1", "title": "Lemma"}],
            "math_semantic_units": [{"unit_id": "u-1", "unit_type": "lemma"}],
            "cross_reference_edges": [],
            "parse_confidence": 0.97,
            "semantic_unit_confidence": 0.95,
            "bbox_coverage_score": 0.98,
            "record_quality_flags": ["anchors_complete"],
            "parser_provenance_id": "math-doc-pipeline-v1",
            "parser_provenance_refs": {
                "layout_parser_manifest_id": "layout-parser-v1",
                "ocr_manifest_id": "ocr-parser-v1",
                "formula_parser_manifest_id": "formula-parser-v1",
                "semantic_unit_typer_manifest_id": "semantic-unit-typer-v1",
            },
            "ocr_layout_extractor_version": "bootstrap-v1",
            "formula_parser_version": "bootstrap-v1",
            "semantic_unit_typer_version": "bootstrap-v1",
            "parse_config_fingerprint": "cfg-doc-pipeline-bootstrap",
            "unresolved_region_ratio": 0.01,
        }
    )
    projection = validate_math_document_projection(
        {
            "schema": "math_document_projection/v1",
            "projection_id": "proj-001",
            "record_id": record.record_id,
            "projection_kind": "state_ir_seed",
            "state_ir_slot_targets": ["PF", "SY", "CG"],
            "anchor_refs": ["anchor-1", "anchor-2"],
            "payload": {"task_type": "proof"},
        }
    )

    assert source.source_format == "PDF"
    assert record.parser_provenance_refs["semantic_unit_typer_manifest_id"] == "semantic-unit-typer-v1"
    assert projection.state_ir_slot_targets == ("PF", "SY", "CG")
