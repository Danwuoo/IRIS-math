from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from iris.train.data.access import open_streaming_source
from iris.train.data.contracts import DatasetSourceSpec
from iris.train.data.filters import prepare_clean_text
from iris.train.data.qa_gate import evaluate_text_quality


def test_open_streaming_source_uses_forced_file_fallback_for_pes2o(tmp_path: Path) -> None:
    source = DatasetSourceSpec(
        source_id="pes2o_math_documents",
        hf_path="allenai/peS2o",
        hf_name="v2",
        split="train",
        revision="0123456789abcdef0123456789abcdef01234567",
        text_field="text",
        ratio_total=0.2,
        metadata={
            "required_source": "s2orc",
            "hf_force_file_fallback": True,
            "hf_file_fallback_builder": "json",
            "hf_repo_files_by_split": {
                "train": [
                    "data/v2/train-00010-of-00020.json.gz",
                    "data/v2/train-00011-of-00020.json.gz",
                ],
            },
        },
    )
    calls = []

    def loader(dataset_or_builder: str, **kwargs):
        calls.append((dataset_or_builder, kwargs))
        assert dataset_or_builder == "json"
        return iter(
            [
                {
                    "text": "This s2orc document contains enough mathematical exposition to pass the text gate.",
                    "source": "s2orc",
                }
            ]
        )

    opened = open_streaming_source(
        source,
        streaming_mode="auto",
        snapshot_root=tmp_path,
        loader=loader,
    )

    assert opened.effective_mode == "hf_online"
    first_record = next(iter(opened.iterable))
    assert first_record["source"] == "s2orc"
    assert len(calls) == 1
    assert calls[0][0] == "json"
    assert calls[0][1]["data_files"][0].startswith(
        "hf://datasets/allenai/peS2o@0123456789abcdef0123456789abcdef01234567/data/v2/train-00010"
    )


def test_open_streaming_source_supports_proof_pile_file_fallback(monkeypatch, tmp_path: Path) -> None:
    source = DatasetSourceSpec(
        source_id="proof_pile_document_aux",
        hf_path="hoskinson-center/proof-pile",
        hf_name="default",
        split="train",
        revision="fedcba9876543210fedcba9876543210fedcba98",
        text_field="text",
        ratio_total=0.15,
        metadata={
            "hf_file_fallback_builder": "json",
            "hf_repo_file_patterns_by_split": {
                "train": ["train/proofpile_train_*.jsonl.gz"],
            },
        },
    )
    monkeypatch.setattr(
        "iris.train.data.access._list_hf_dataset_files",
        lambda repo_id, revision: (
            "train/proofpile_train_0.jsonl.gz",
            "train/proofpile_train_1.jsonl.gz",
        ),
    )

    def loader(dataset_or_builder: str, **kwargs):
        if dataset_or_builder == "hoskinson-center/proof-pile":
            raise RuntimeError("Dataset scripts are no longer supported")
        assert dataset_or_builder == "json"
        return iter(
            [
                {
                    "text": "A long proof-pile sample with enough mathematical prose to pass the generic cleaner.",
                    "meta": '{"subset":"arxiv"}',
                }
            ]
        )

    opened = open_streaming_source(
        source,
        streaming_mode="auto",
        snapshot_root=tmp_path,
        loader=loader,
    )

    assert opened.effective_mode == "hf_online"
    assert next(iter(opened.iterable))["meta"] == '{"subset":"arxiv"}'


def test_open_streaming_source_normalizes_resolve_revision(tmp_path: Path) -> None:
    source = DatasetSourceSpec(
        source_id="fineweb_edu_math_core",
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="sample-10BT",
        split="train",
        revision="resolve:main",
        text_field="text",
        ratio_total=0.12,
    )
    seen = {}

    def loader(dataset_or_builder: str, **kwargs):
        seen["dataset_or_builder"] = dataset_or_builder
        seen["kwargs"] = kwargs
        return iter(
            [
                {
                    "text": "This sample contains enough mathematical exposition to survive the generic cleaner.",
                    "url": "https://example.org/math/intro",
                }
            ]
        )

    opened = open_streaming_source(
        source,
        streaming_mode="auto",
        snapshot_root=tmp_path,
        loader=loader,
    )

    assert opened.effective_mode == "hf_online"
    assert seen["dataset_or_builder"] == "HuggingFaceFW/fineweb-edu"
    assert seen["kwargs"]["revision"] == "main"


def test_open_streaming_source_materializes_prooflang_zip_snapshot(monkeypatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "proofs.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "proofs.tsv",
            "paper\tproof\n1234.5678\tThis proof text is sufficiently long to survive the cleaner.\n",
        )

    source = DatasetSourceSpec(
        source_id="prooflang_document_aux",
        hf_path="proofcheck/prooflang",
        hf_name="proofs",
        split="train",
        revision="00112233445566778899aabbccddeeff00112233",
        text_field="proof",
        ratio_total=0.15,
        required=True,
        local_snapshot_pattern="**/*.tsv",
        metadata={"hf_snapshot_archives": ["proofs.zip"]},
    )
    monkeypatch.setattr(
        "iris.train.data.access._download_hf_dataset_file",
        lambda repo_id, revision, filename, cache_dir=None: archive_path,
    )

    def loader(dataset_or_builder: str, **kwargs):
        if dataset_or_builder == "proofcheck/prooflang":
            raise RuntimeError("Dataset scripts are no longer supported")
        assert dataset_or_builder == "csv"
        assert kwargs["delimiter"] == "\t"
        data_files = kwargs["data_files"]
        assert any(str(path).endswith(".tsv") for path in data_files)
        return iter(
            [
                {
                    "paper": "1234.5678",
                    "proof": "This proof text is sufficiently long to survive the cleaner.",
                }
            ]
        )

    opened = open_streaming_source(
        source,
        streaming_mode="auto",
        snapshot_root=tmp_path,
        loader=loader,
    )

    assert opened.effective_mode == "local_snapshot"
    assert next(iter(opened.iterable))["paper"] == "1234.5678"
    marker = (
        tmp_path
        / "prooflang_document_aux"
        / "_materialized"
        / "00112233445566778899aabbccddeeff00112233"
        / ".materialized.json"
    )
    assert marker.exists()


def test_open_streaming_source_uses_snapshot_fallback_root(monkeypatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "proofs.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "proofs.tsv",
            "paper\tproof\n1234.5678\tThis proof text is sufficiently long to survive the cleaner.\n",
        )

    primary_root = tmp_path / "primary_snapshots"
    fallback_root = tmp_path / "fallback_snapshots"
    source = DatasetSourceSpec(
        source_id="prooflang_document_aux",
        hf_path="proofcheck/prooflang",
        hf_name="proofs",
        split="train",
        revision="00112233445566778899aabbccddeeff00112233",
        text_field="proof",
        ratio_total=0.15,
        required=True,
        local_snapshot_pattern="**/*.tsv",
        metadata={"hf_snapshot_archives": ["proofs.zip"]},
    )
    monkeypatch.setattr(
        "iris.train.data.access._download_hf_dataset_file",
        lambda repo_id, revision, filename, cache_dir=None: archive_path,
    )

    def loader(dataset_or_builder: str, **kwargs):
        if dataset_or_builder == "proofcheck/prooflang":
            raise RuntimeError("Dataset scripts are no longer supported")
        assert dataset_or_builder == "csv"
        return iter(
            [
                {
                    "paper": "1234.5678",
                    "proof": "This proof text is sufficiently long to survive the cleaner.",
                }
            ]
        )

    opened = open_streaming_source(
        source,
        streaming_mode="auto",
        snapshot_root=primary_root,
        snapshot_fallback_root=fallback_root,
        loader=loader,
    )

    assert opened.effective_mode == "local_snapshot"
    assert next(iter(opened.iterable))["paper"] == "1234.5678"
    marker = (
        fallback_root
        / "prooflang_document_aux"
        / "_materialized"
        / "00112233445566778899aabbccddeeff00112233"
        / ".materialized.json"
    )
    assert marker.exists()
    assert not (primary_root / "prooflang_document_aux").exists()


def test_open_streaming_source_materializes_prooflang_when_snapshot_root_is_stale(
    monkeypatch, tmp_path: Path
) -> None:
    archive_path = tmp_path / "proofs.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "proofs.tsv",
            "paper\tproof\n1234.5678\tThis proof text is sufficiently long to survive the cleaner.\n",
        )

    stale_root = tmp_path / "prooflang_document_aux"
    stale_root.mkdir(parents=True, exist_ok=True)
    (stale_root / "_hf_hub_cache").mkdir(parents=True, exist_ok=True)

    source = DatasetSourceSpec(
        source_id="prooflang_document_aux",
        hf_path="proofcheck/prooflang",
        hf_name="proofs",
        split="train",
        revision="00112233445566778899aabbccddeeff00112233",
        text_field="proof",
        ratio_total=0.15,
        required=True,
        local_snapshot_pattern="**/*.tsv",
        metadata={"hf_snapshot_archives": ["proofs.zip"]},
    )
    monkeypatch.setattr(
        "iris.train.data.access._download_hf_dataset_file",
        lambda repo_id, revision, filename, cache_dir=None: archive_path,
    )

    def loader(dataset_or_builder: str, **kwargs):
        if dataset_or_builder == "proofcheck/prooflang":
            raise RuntimeError("Dataset scripts are no longer supported")
        assert dataset_or_builder == "csv"
        return iter(
            [
                {
                    "paper": "1234.5678",
                    "proof": "This proof text is sufficiently long to survive the cleaner.",
                }
            ]
        )

    opened = open_streaming_source(
        source,
        streaming_mode="auto",
        snapshot_root=tmp_path,
        loader=loader,
    )

    assert opened.effective_mode == "local_snapshot"
    assert next(iter(opened.iterable))["paper"] == "1234.5678"
    marker = (
        tmp_path
        / "prooflang_document_aux"
        / "_materialized"
        / "00112233445566778899aabbccddeeff00112233"
        / ".materialized.json"
    )
    assert marker.exists()


def test_prepare_clean_text_uses_fallback_fields_and_required_source() -> None:
    source = DatasetSourceSpec(
        source_id="lean_workbook_bridge",
        hf_path="pkuAI4M/LeanWorkbook",
        hf_name=None,
        split="train",
        revision="sha",
        text_field="natural_statement",
        ratio_total=0.1,
        metadata={
            "fallback_text_field": "natural_language_statement",
            "required_source": "s2orc",
        },
    )
    record = {
        "natural_language_statement": "This natural language statement is long enough to pass the cleaner.",
        "source": "s2orc",
    }

    assert prepare_clean_text(source, record) is not None
    assert prepare_clean_text(source, dict(record, source="s2orc/train")) is not None
    assert prepare_clean_text(source, dict(record, source="s2ag")) is None


def test_prepare_clean_text_can_join_multiple_fields() -> None:
    source = DatasetSourceSpec(
        source_id="numina_math_lean_core",
        hf_path="AI-MO/NuminaMath-LEAN",
        hf_name=None,
        split="train",
        revision="sha",
        text_field="formal_statement",
        ratio_total=0.1,
        metadata={
            "text_join_fields": ["problem", "formal_statement", "formal_proof"],
        },
    )
    record = {
        "problem": "Explain why the series converges and provide the main proof idea in prose.",
        "formal_statement": "theorem convergence_statement : True := by trivial",
        "formal_proof": "by trivial",
    }

    text = prepare_clean_text(source, record)
    assert text is not None
    assert "Explain why the series converges" in text
    assert "theorem convergence_statement" in text


def test_prepare_clean_text_can_join_problem_and_solution() -> None:
    source = DatasetSourceSpec(
        source_id="numina_math_weak_supervision",
        hf_path="AI-MO/NuminaMath-1.5",
        hf_name=None,
        split="train",
        revision="sha",
        text_field="problem",
        ratio_total=0.1,
        metadata={
            "text_join_fields": ["problem", "solution"],
        },
    )
    record = {
        "problem": "Solve the equation and justify the substitution in full sentences.",
        "solution": "We substitute x = y + 1, simplify the polynomial, and then verify the resulting roots carefully.",
    }

    text = prepare_clean_text(source, record)
    assert text is not None
    assert "Solve the equation" in text
    assert "verify the resulting roots carefully" in text


def test_document_relaxed_qa_profile_accepts_fragmented_document_text() -> None:
    text = "\n".join(
        [
            "Section 1",
            "Let us consider the following argument in a long-form academic document.",
            "Equation (1) implies the bound after rearranging the terms carefully.",
            "Proof proceeds by induction on n with several intermediate reductions.",
        ]
        * 16
    )

    default_report = evaluate_text_quality(text)
    relaxed_report = evaluate_text_quality(text, profile="document_relaxed")

    assert relaxed_report["pass"] or not default_report["pass"]


def test_prepare_clean_text_uses_document_chunk_fallback_for_pes2o() -> None:
    source = DatasetSourceSpec(
        source_id="pes2o_math_documents",
        hf_path="allenai/peS2o",
        hf_name="v2",
        split="train",
        revision="sha",
        text_field="text",
        ratio_total=0.2,
        metadata={
            "required_source": "s2orc",
            "qa_gate_profile": "document_relaxed",
            "document_ingestion": {
                "mode": "paragraph_window",
                "window_paragraphs": 4,
                "stride_paragraphs": 2,
                "min_chunk_chars": 120,
                "max_chunk_chars": 500,
            },
        },
    )
    good_paragraph = (
        "This section develops a mathematical argument with enough prose to survive the cleaner "
        "while preserving document-level structure and explanatory detail."
    )
    noisy_tail = "\n\n".join(
        f"https://example.org/reference/{idx} additional metadata"
        for idx in range(12)
    )
    record = {
        "source": "s2orc/train",
        "text": "\n\n".join([good_paragraph] * 5 + [noisy_tail]),
    }

    text = prepare_clean_text(source, record)
    assert text is not None
    assert "mathematical argument" in text
    assert text.count("https://") < 8
