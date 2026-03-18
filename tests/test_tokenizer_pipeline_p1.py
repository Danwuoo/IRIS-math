from __future__ import annotations

from pathlib import Path

import pytest

from iris.train.data.p1_manifest import P1StreamingManifest, P1StreamingSource
from iris.train.data.token_accounting import load_tokenizer_handle
from iris.train.tokenizer_pipeline import TokenizerBuildConfig, write_tokenizer_corpus


def _synthetic_manifest() -> P1StreamingManifest:
    return P1StreamingManifest(
        schema="p1_streaming_manifest/v1",
        manifest_id="p1-synth-test",
        profile_id="P1",
        phase="E",
        commit_posture="committed",
        default_streaming_mode="auto",
        sources=(
            P1StreamingSource("a", "synthetic", "A", "core", 0.20, 0.25, revision="synthetic-v1"),
            P1StreamingSource("b", "synthetic", "B", "core", 0.10, 0.10, revision="synthetic-v1"),
            P1StreamingSource("c", "synthetic", "C", "core", 0.25, 0.25, revision="synthetic-v1"),
            P1StreamingSource("d", "synthetic", "D", "core", 0.35, 0.20, revision="synthetic-v1"),
            P1StreamingSource("e", "synthetic", "E", "core", 0.10, 0.20, revision="synthetic-v1"),
        ),
    )


def test_tokenizer_corpus_generation_is_deterministic_for_synthetic_manifest(tmp_path: Path) -> None:
    manifest = _synthetic_manifest()
    build_config = TokenizerBuildConfig(
        sample_records_per_source=4,
        max_corpus_chars=2_000,
        seed=23,
    )

    path_a = write_tokenizer_corpus(
        manifest,
        output_dir=tmp_path / "a",
        streaming_mode="auto",
        snapshot_root=None,
        snapshot_fallback_root=None,
        build_config=build_config,
    )
    path_b = write_tokenizer_corpus(
        manifest,
        output_dir=tmp_path / "b",
        streaming_mode="auto",
        snapshot_root=None,
        snapshot_fallback_root=None,
        build_config=build_config,
    )

    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_tokenizer_corpus_generation_matches_serial_output_when_parallelized(tmp_path: Path) -> None:
    manifest = _synthetic_manifest()
    build_config = TokenizerBuildConfig(
        sample_records_per_source=8,
        max_corpus_chars=4_000,
        seed=31,
    )

    serial_path = write_tokenizer_corpus(
        manifest,
        output_dir=tmp_path / "serial",
        streaming_mode="auto",
        snapshot_root=None,
        snapshot_fallback_root=None,
        build_config=build_config,
        corpus_workers=1,
    )
    parallel_path = write_tokenizer_corpus(
        manifest,
        output_dir=tmp_path / "parallel",
        streaming_mode="auto",
        snapshot_root=None,
        snapshot_fallback_root=None,
        build_config=build_config,
        corpus_workers=4,
    )

    assert serial_path.read_text(encoding="utf-8") == parallel_path.read_text(encoding="utf-8")


def test_sentencepiece_training_produces_reloadable_tokenizer_dir(tmp_path: Path) -> None:
    pytest.importorskip("sentencepiece")
    pytest.importorskip("transformers")

    from iris.train.tokenizer_pipeline import train_sentencepiece_tokenizer

    artifact = train_sentencepiece_tokenizer(
        manifest=_synthetic_manifest(),
        output_dir=tmp_path / "tok",
        streaming_mode="auto",
        snapshot_root=None,
        snapshot_fallback_root=None,
        build_config=TokenizerBuildConfig(
            vocab_size=384,
            sample_records_per_source=4,
            max_corpus_chars=4_000,
            seed=19,
        ),
        corpus_workers=2,
        sentencepiece_threads=2,
    )

    handle = load_tokenizer_handle(str(artifact.tokenizer_dir))

    assert artifact.manifest_path.exists()
    assert handle.fingerprint
    assert handle.tokenizer.encode("Synthetic math tokenizer probe.", add_special_tokens=False)
