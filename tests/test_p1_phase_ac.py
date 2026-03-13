from __future__ import annotations

from pathlib import Path

from iris.runtime import resolve_task_semantics
from iris.train import (
    admit_p1_train_visible_record,
    build_document_pipeline_bundle,
    load_learning_objective_bundle_registry,
    load_policy_bundle_for_profile_phase,
    resolve_learning_objective_bundle,
    validate_train_visible_record,
)

from tests.state_ir_factory import make_state_ir


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "p1_phase_c"


def test_resolve_task_semantics_uses_benchmark_defaults_for_family_specific_policy() -> None:
    state = make_state_ir(seed=11)
    bundle = load_policy_bundle_for_profile_phase("P1", "C")
    benchmark_policy = bundle.benchmark_family_policies["miniF2F-v1"]

    semantics = resolve_task_semantics(
        state.PF,
        item_task_family="formalization",
        benchmark_family_policy=benchmark_policy,
    )

    assert semantics.task_family == "formalization"
    assert semantics.task_family_resolution_source == "item_explicit"
    assert semantics.task_adjudication_policy_id == "minif2f-formalization-tight-v1"
    assert semantics.task_adjudication_policy_resolution_source == "benchmark_family_default"


def test_learning_objective_bundle_registry_resolves_p1_phase_defaults() -> None:
    bundles, default_map = load_learning_objective_bundle_registry()
    assert default_map["P1:A"] == "p1-phase-a-bundle-v1"
    assert default_map["P1:B"] == "p1-phase-b-bundle-v1"
    assert default_map["P1:C"] == "p1-phase-c-bundle-v1"
    assert bundles["p1-phase-c-bundle-v1"].objective_families

    bundle_a, source_a = resolve_learning_objective_bundle(profile_id="P1", phase="A")
    bundle_c, source_c = resolve_learning_objective_bundle(profile_id="P1", phase="C")

    assert bundle_a.learning_objective_bundle_id == "p1-phase-a-bundle-v1"
    assert bundle_c.learning_objective_bundle_id == "p1-phase-c-bundle-v1"
    assert source_a == "profile_phase_default"
    assert source_c == "profile_phase_default"


def test_phase_a_allows_zero_tier1_cap_but_marks_tier1_record_non_train_visible() -> None:
    bundle = load_policy_bundle_for_profile_phase("P1", "A")
    record = validate_train_visible_record(
        {
            "record_id": "record-001",
            "pool_id": "B",
            "pool_role": "auxiliary",
            "data_realization_policy_id": bundle.data_realization_policy.data_realization_policy_id,
            "decontam_policy_id": bundle.data_realization_policy.decontam_policy_id,
            "fingerprint_set": {"source_fingerprint": "sha256:abc"},
            "source_family": "worked_example",
            "provenance_refs": {"parser_provenance_id": "math-doc-pipeline-v1"},
            "quality_flags": ["trace_reconstructable"],
            "source_record_lineage": {"source_id": "src-001"},
            "benchmark_family_id": "aimo-v1",
            "benchmark_tier": "Tier 1",
            "math_document_record_id": "doc-001",
            "formalizer_provenance_id": "formalizer-skeleton-v1",
        },
        bundle.data_realization_policy,
    )

    admission = admit_p1_train_visible_record(record, bundle)

    assert bundle.data_realization_policy.tier1_global_cap == {"token_cap": 0.0, "record_cap": 0.0}
    assert admission.train_visible is False
    assert admission.admission_status == "admitted"


def test_document_pipeline_builds_pdf_and_docx_bundles_from_checked_in_fixtures() -> None:
    fixture_root = _fixture_root()
    pdf_bundle = build_document_pipeline_bundle(
        fixture_root / "documents" / "bootstrap_identity.pdf",
        sidecar_path=fixture_root / "documents" / "bootstrap_identity.pdf.sidecar.json",
    )
    docx_bundle = build_document_pipeline_bundle(
        fixture_root / "documents" / "bootstrap_identity.docx"
    )

    assert pdf_bundle.source.source_format == "PDF"
    assert pdf_bundle.record.anchor_index
    assert pdf_bundle.projection.state_ir_slot_targets == ("PF", "SY", "CG")
    assert docx_bundle.source.source_format == "DOCX"
    assert docx_bundle.record.clean_text_blocks
    assert docx_bundle.projection.payload["symbols"]
