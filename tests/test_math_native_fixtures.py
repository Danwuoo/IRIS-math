from __future__ import annotations

from pathlib import Path

from iris.regression import load_document_eval_fixtures, load_proof_eval_fixtures


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "p1_phase_de"


def test_load_document_eval_fixtures_filters_by_partition() -> None:
    root = _fixture_root() / "document_eval"
    diagnostic = load_document_eval_fixtures(root, eval_partition="diagnostic")
    heldout = load_document_eval_fixtures(root, eval_partition="strict_holdout")
    assert {fixture.fixture_id for fixture in diagnostic} >= {"typed_identity_pdf", "scanned_identity_note"}
    assert {fixture.fixture_id for fixture in heldout} == {"formula_image_heldout", "typed_identity_heldout"}


def test_load_proof_eval_fixtures_filters_by_partition() -> None:
    root = _fixture_root() / "proof_eval"
    diagnostic = load_proof_eval_fixtures(root, eval_partition="diagnostic")
    heldout = load_proof_eval_fixtures(root, eval_partition="strict_holdout")
    assert {fixture.fixture_id for fixture in diagnostic} == {"diagnostic_nl_grounded", "diagnostic_semi_formal_ocr"}
    assert {fixture.fixture_id for fixture in heldout} == {"heldout_formalization_bridge", "heldout_semi_formal_formula"}
