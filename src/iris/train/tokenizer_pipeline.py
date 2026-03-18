from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .data.access import open_streaming_source
from .data.contracts import DatasetSourceSpec
from .data.filters import prepare_clean_text
from .data.p1_manifest import P1StreamingManifest, P1StreamingSource, manifest_sha256
from .governance import stable_hash


class TokenizerBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenizerBuildConfig:
    vocab_size: int = 50_176
    sample_records_per_source: int = 2_048
    max_corpus_chars: int = 4_000_000
    seed: int = 17
    model_prefix: str = "iris_p1_tokenizer"


@dataclass(frozen=True)
class TokenizerArtifact:
    tokenizer_dir: Path
    manifest_path: Path
    tokenizer_manifest_id: str
    tokenizer_manifest_sha256: str


def _stable_record_seed(*, source_id: str, index: int, seed: int, manifest_sha: str) -> str:
    return stable_hash(
        {
            "source_id": str(source_id),
            "index": int(index),
            "seed": int(seed),
            "manifest_sha": str(manifest_sha),
        }
    )


def _coerce_dataset_spec(source: P1StreamingSource) -> DatasetSourceSpec:
    return DatasetSourceSpec(
        source_id=source.source_id,
        hf_path=str(source.hf_path or ""),
        hf_name=source.hf_name,
        split=source.split,
        revision=str(source.revision or ""),
        text_field=source.payload_field,
        ratio_total=float(source.token_weight),
        required=bool(source.required),
        local_snapshot_pattern=source.local_snapshot_pattern,
        metadata=dict(source.metadata),
    )


def _synthetic_tokenizer_lines(source: P1StreamingSource, *, max_records: int, seed: int) -> Iterable[str]:
    family = str(source.source_family or "synthetic")
    for index in range(int(max_records)):
        digest = _stable_record_seed(
            source_id=source.source_id,
            index=index,
            seed=seed,
            manifest_sha=family,
        )[:16]
        yield (
            f"Synthetic {source.source_id} sample {index} for pool {source.pool_id} "
            f"uses digest {digest} to preserve deterministic tokenizer coverage."
        )


def iter_manifest_texts(
    manifest: P1StreamingManifest,
    *,
    streaming_mode: str,
    snapshot_root: Path | None,
    snapshot_fallback_root: Path | None,
    sample_records_per_source: int,
    seed: int,
) -> Iterable[str]:
    for source in manifest.sources:
        if source.source_kind == "synthetic":
            yield from _synthetic_tokenizer_lines(
                source,
                max_records=sample_records_per_source,
                seed=seed,
            )
            continue
        dataset_spec = _coerce_dataset_spec(source)
        opened = open_streaming_source(
            dataset_spec,
            streaming_mode=streaming_mode,
            snapshot_root=snapshot_root,
            snapshot_fallback_root=snapshot_fallback_root,
        )
        iterator = iter(opened.iterable)
        yielded = 0
        while yielded < int(sample_records_per_source):
            try:
                record = next(iterator)
            except StopIteration:
                break
            if not isinstance(record, Mapping):
                continue
            text = prepare_clean_text(dataset_spec, record)
            if not text:
                continue
            yielded += 1
            yield text


def _collect_source_texts(
    source: P1StreamingSource,
    *,
    streaming_mode: str,
    snapshot_root: Path | None,
    snapshot_fallback_root: Path | None,
    sample_records_per_source: int,
    seed: int,
) -> tuple[str, ...]:
    if source.source_kind == "synthetic":
        return tuple(
            _synthetic_tokenizer_lines(
                source,
                max_records=sample_records_per_source,
                seed=seed,
            )
        )

    dataset_spec = _coerce_dataset_spec(source)
    opened = open_streaming_source(
        dataset_spec,
        streaming_mode=streaming_mode,
        snapshot_root=snapshot_root,
        snapshot_fallback_root=snapshot_fallback_root,
    )
    iterator = iter(opened.iterable)
    rows = []
    yielded = 0
    while yielded < int(sample_records_per_source):
        try:
            record = next(iterator)
        except StopIteration:
            break
        if not isinstance(record, Mapping):
            continue
        text = prepare_clean_text(dataset_spec, record)
        if not text:
            continue
        yielded += 1
        rows.append(text)
    return tuple(rows)


def write_tokenizer_corpus(
    manifest: P1StreamingManifest,
    *,
    output_dir: Path,
    streaming_mode: str,
    snapshot_root: Path | None,
    snapshot_fallback_root: Path | None,
    build_config: TokenizerBuildConfig,
    corpus_workers: int = 1,
) -> Path:
    corpus_path = Path(output_dir) / "tokenizer_corpus.txt"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    sources = tuple(manifest.sources)
    source_rows: Dict[str, tuple[str, ...]] = {}
    worker_count = max(1, min(int(corpus_workers), len(sources))) if sources else 1
    if worker_count <= 1:
        for source in sources:
            source_rows[source.source_id] = _collect_source_texts(
                source,
                streaming_mode=streaming_mode,
                snapshot_root=snapshot_root,
                snapshot_fallback_root=snapshot_fallback_root,
                sample_records_per_source=build_config.sample_records_per_source,
                seed=build_config.seed,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="iris-tok-corpus") as executor:
            futures = {
                source.source_id: executor.submit(
                    _collect_source_texts,
                    source,
                    streaming_mode=streaming_mode,
                    snapshot_root=snapshot_root,
                    snapshot_fallback_root=snapshot_fallback_root,
                    sample_records_per_source=build_config.sample_records_per_source,
                    seed=build_config.seed,
                )
                for source in sources
            }
            for source in sources:
                source_rows[source.source_id] = futures[source.source_id].result()

    written_chars = 0
    with corpus_path.open("w", encoding="utf-8") as handle:
        for source in sources:
            for text in source_rows.get(source.source_id, ()):
                if written_chars >= int(build_config.max_corpus_chars):
                    break
                clipped = text[: max(int(build_config.max_corpus_chars) - written_chars, 0)]
                if not clipped:
                    break
                handle.write(clipped.replace("\r\n", "\n").replace("\r", "\n") + "\n")
                written_chars += len(clipped)
            if written_chars >= int(build_config.max_corpus_chars):
                break
    if written_chars <= 0:
        raise TokenizerBuildError("Tokenizer corpus generation produced no usable text.")
    return corpus_path


def build_tokenizer_manifest_payload(
    *,
    manifest: P1StreamingManifest,
    tokenizer_dir: Path,
    build_config: TokenizerBuildConfig,
    corpus_path: Path,
) -> Dict[str, Any]:
    return {
        "schema": "iris.tokenizer_build_manifest/v1",
        "manifest_id": "tok-" + manifest_sha256(manifest)[:12],
        "profile_id": manifest.profile_id,
        "phase": manifest.phase,
        "streaming_manifest_id": manifest.manifest_id,
        "streaming_manifest_sha256": manifest_sha256(manifest),
        "vocab_size": int(build_config.vocab_size),
        "seed": int(build_config.seed),
        "sample_records_per_source": int(build_config.sample_records_per_source),
        "max_corpus_chars": int(build_config.max_corpus_chars),
        "corpus_path": str(corpus_path),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_files": sorted(path.name for path in Path(tokenizer_dir).glob("*") if path.is_file()),
    }


def tokenizer_manifest_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("tokenizer_manifest_sha256", None)
    return stable_hash(normalized)


def _maybe_write_hf_tokenizer_wrapper(tokenizer_dir: Path, model_path: Path) -> None:
    try:
        from transformers import LlamaTokenizer
    except ImportError:
        return
    tokenizer = LlamaTokenizer(
        vocab_file=str(model_path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        legacy=False,
    )
    tokenizer.save_pretrained(str(tokenizer_dir))


def train_sentencepiece_tokenizer(
    *,
    manifest: P1StreamingManifest,
    output_dir: Path,
    streaming_mode: str,
    snapshot_root: Path | None,
    snapshot_fallback_root: Path | None,
    build_config: TokenizerBuildConfig,
    corpus_workers: int = 1,
    sentencepiece_threads: int = 1,
) -> TokenizerArtifact:
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise TokenizerBuildError(
            "sentencepiece is required to build the P1 tokenizer."
        ) from error

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_root / build_config.model_prefix
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tokenizer_dir / "tokenizer_build_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = tokenizer_manifest_sha256(payload)
        if expected_sha == str(payload.get("tokenizer_manifest_sha256", "")):
            return TokenizerArtifact(
                tokenizer_dir=tokenizer_dir,
                manifest_path=manifest_path,
                tokenizer_manifest_id=str(payload.get("manifest_id", "")),
                tokenizer_manifest_sha256=expected_sha,
            )

    corpus_path = write_tokenizer_corpus(
        manifest,
        output_dir=output_root,
        streaming_mode=streaming_mode,
        snapshot_root=snapshot_root,
        snapshot_fallback_root=snapshot_fallback_root,
        build_config=build_config,
        corpus_workers=corpus_workers,
    )
    model_prefix = tokenizer_dir / "sentencepiece"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=int(build_config.vocab_size),
        model_type="bpe",
        byte_fallback=True,
        character_coverage=1.0,
        shuffle_input_sentence=False,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        num_threads=max(int(sentencepiece_threads), 1),
    )
    _maybe_write_hf_tokenizer_wrapper(tokenizer_dir, model_prefix.with_suffix(".model"))
    payload = build_tokenizer_manifest_payload(
        manifest=manifest,
        tokenizer_dir=tokenizer_dir,
        build_config=build_config,
        corpus_path=corpus_path,
    )
    payload["tokenizer_manifest_sha256"] = tokenizer_manifest_sha256(payload)
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return TokenizerArtifact(
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        tokenizer_manifest_id=str(payload["manifest_id"]),
        tokenizer_manifest_sha256=str(payload["tokenizer_manifest_sha256"]),
    )
