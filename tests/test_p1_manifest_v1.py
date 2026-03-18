from __future__ import annotations

from pathlib import Path

from iris.train.data import (
    load_default_p1_streaming_manifest,
    load_p1_streaming_manifest,
    resolve_manifest_revisions,
)


def test_default_p1_manifest_resolves_to_committed_sha_locked_variant() -> None:
    manifest = load_default_p1_streaming_manifest()

    assert manifest.schema == "p1_streaming_manifest/v1"
    assert manifest.commit_posture == "bootstrap"
    assert {source.pool_id for source in manifest.sources} == {"A", "B", "C", "D", "E"}
    prooflang = manifest.source_by_id["prooflang_document_aux"]
    pes2o = manifest.source_by_id["pes2o_math_documents"]
    lean_workbook = manifest.source_by_id["lean_workbook_bridge"]

    assert prooflang.required is True
    assert prooflang.hf_name == "proofs"
    assert prooflang.payload_field == "proof"
    assert prooflang.local_snapshot_pattern == "**/*.tsv"
    assert tuple(prooflang.metadata.get("hf_snapshot_archives", [])) == ("proofs.zip",)

    assert pes2o.hf_name == "v2"
    assert pes2o.metadata.get("hf_force_file_fallback") is True
    assert pes2o.metadata.get("hf_file_fallback_builder") == "json"
    assert pes2o.metadata["hf_repo_files_by_split"]["train"][0] == "data/v2/train-00010-of-00020.json.gz"
    assert pes2o.metadata.get("qa_gate_profile") == "document_relaxed"
    assert pes2o.metadata["document_ingestion"]["mode"] == "paragraph_window"
    assert pes2o.metadata["hf_repo_file_patterns_by_split"]["train"] == ["data/v2/train-*.json.gz"]

    assert lean_workbook.payload_field == "natural_language_statement"
    assert manifest.source_by_id["numina_math_weak_supervision"].metadata["text_join_fields"] == [
        "problem",
        "solution",
    ]
    assert manifest.source_by_id["numina_math_lean_core"].metadata["text_join_fields"] == [
        "problem",
        "formal_statement",
        "formal_proof",
    ]

    resolved = resolve_manifest_revisions(
        manifest,
        dataset_sha_resolver=lambda dataset_id, config_name, revision_hint: (
            "0123456789abcdef0123456789abcdef01234567"
        ),
    )

    assert resolved.commit_posture == "committed"
    assert resolved.manifest_id.startswith("p1-committed-")
    for source in resolved.sources:
        if source.source_kind == "hf":
            assert source.revision == "0123456789abcdef0123456789abcdef01234567"
        else:
            assert source.revision in (None, "synthetic-v1")


def test_proofpile_variant_manifest_is_valid() -> None:
    manifest = load_p1_streaming_manifest(
        Path(__file__).resolve().parents[1]
        / "src"
        / "iris"
        / "train"
        / "data"
        / "profiles"
        / "p1_streaming_manifest_proofpile_v1.json"
    )

    assert manifest.manifest_id == "p1-bootstrap-streaming-proofpile-v1"
    proof_pile = manifest.source_by_id["proof_pile_document_aux"]
    assert proof_pile.hf_path == "hoskinson-center/proof-pile"
    assert proof_pile.metadata["hf_file_fallback_builder"] == "json"
