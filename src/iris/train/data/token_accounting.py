from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


class TokenizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenizerHandle:
    id_or_path: str
    tokenizer: Any
    fingerprint: str


@dataclass
class TokenLedger:
    total_tokens: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)

    def add(self, source_id: str, token_count: int) -> None:
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        self.total_tokens += token_count
        self.by_source[source_id] = int(self.by_source.get(source_id, 0)) + token_count

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": int(self.total_tokens),
            "by_source": {str(source_id): int(count) for source_id, count in sorted(self.by_source.items())},
        }

    def ratio_for_source(self, source_id: str) -> float:
        if self.total_tokens <= 0:
            return 0.0
        return float(self.by_source.get(source_id, 0)) / float(self.total_tokens)


def _load_auto_tokenizer(id_or_path: str, *, use_fast: bool = True) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TokenizerError(
            "transformers is required for streaming pretrain mode. Install transformers first."
        ) from error
    return AutoTokenizer.from_pretrained(id_or_path, use_fast=use_fast, trust_remote_code=False)


def _tokenizer_fingerprint_payload(tokenizer: Any, id_or_path: str) -> Dict[str, Any]:
    vocab = tokenizer.get_vocab()
    # Keep hashing cost bounded while preserving deterministic identity.
    sample_items = sorted(vocab.items(), key=lambda item: item[0])[:4096]
    return {
        "id_or_path": id_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(vocab))),
        "special_tokens_map": dict(getattr(tokenizer, "special_tokens_map", {})),
        "sample_vocab": sample_items,
    }


def tokenizer_fingerprint(tokenizer: Any, id_or_path: str) -> str:
    payload = _tokenizer_fingerprint_payload(tokenizer, id_or_path)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_tokenizer_handle(id_or_path: str) -> TokenizerHandle:
    if not str(id_or_path).strip():
        raise TokenizerError("tokenizer-id-or-path is required.")
    tokenizer = _load_auto_tokenizer(str(id_or_path).strip())
    fingerprint = tokenizer_fingerprint(tokenizer, str(id_or_path).strip())
    return TokenizerHandle(
        id_or_path=str(id_or_path).strip(),
        tokenizer=tokenizer,
        fingerprint=fingerprint,
    )


def count_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return int(len(encoded))


def truncate_text_to_tokens(tokenizer: Any, text: str, target_tokens: int) -> str:
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= target_tokens:
        return text
    truncated_ids = token_ids[:target_tokens]
    return tokenizer.decode(truncated_ids, clean_up_tokenization_spaces=False)


def validate_tokenizer_required(data_source: str, tokenizer_id_or_path: str | None) -> None:
    needs_tokenizer = str(data_source).strip().lower() in {"pure_lm_streaming", "hybrid_mixture"}
    if needs_tokenizer and not str(tokenizer_id_or_path or "").strip():
        raise TokenizerError(
            "tokenizer-id-or-path is required when data_source is pure_lm_streaming or hybrid_mixture."
        )


def compute_realized_ratios(ledger: Mapping[str, Any], *, precision: int = 6) -> Dict[str, float]:
    total_tokens = int(ledger.get("total_tokens", 0))
    by_source = dict(ledger.get("by_source", {}))
    if total_tokens <= 0:
        return {str(source_id): 0.0 for source_id in by_source.keys()}
    return {
        str(source_id): round(float(int(token_count)) / float(total_tokens), int(precision))
        for source_id, token_count in by_source.items()
    }
