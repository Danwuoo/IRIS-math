from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .contracts import DatasetSourceSpec
from .qa_gate import enforce_qa_gate

_MATH_SIGNAL_PATTERN = re.compile(r"(\\\\Gamma|\\\\vdash|\\\\lambda|\\\\Pi|\\\\Sigma|\\frac|\\sum|\\int)")
_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)


def _get_value(record: Mapping[str, Any], field: str) -> Any:
    if "." not in field:
        return record.get(field)
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value.get(part)
    return value


def _text_field_candidates(source: DatasetSourceSpec) -> tuple[str, ...]:
    candidates = [str(source.text_field).strip()]
    metadata = source.metadata
    for key in ("fallback_text_field", "secondary_field"):
        value = str(metadata.get(key, "")).strip()
        if value:
            candidates.append(value)
    extra_fields = metadata.get("text_field_candidates", [])
    if isinstance(extra_fields, (list, tuple)):
        for value in extra_fields:
            normalized = str(value).strip()
            if normalized:
                candidates.append(normalized)
    ordered = []
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return tuple(ordered)


def _joined_text(source: DatasetSourceSpec, record: Mapping[str, Any]) -> str:
    join_fields = source.metadata.get("text_join_fields", [])
    if not isinstance(join_fields, (list, tuple)):
        return ""
    fragments = []
    for field in join_fields:
        normalized_field = str(field).strip()
        if not normalized_field:
            continue
        text = _normalize_text(_to_text(_get_value(record, normalized_field)))
        if text:
            fragments.append(text)
    return "\n\n".join(fragments)


def _document_ingestion_config(source: DatasetSourceSpec) -> Mapping[str, Any]:
    value = source.metadata.get("document_ingestion", {})
    if not isinstance(value, Mapping):
        return {}
    return value


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return payload
    return {}


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = normalized.replace("\x00", "")
    return normalized


def _document_paragraphs(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    paragraphs = []
    for block in re.split(r"\n\s*\n+", normalized):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        paragraphs.append(" ".join(lines))
    return paragraphs


def _truncate_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    last_space = clipped.rfind(" ")
    if last_space >= max(int(max_chars * 0.7), 1):
        clipped = clipped[:last_space]
    return clipped.rstrip()


def _candidate_texts(source: DatasetSourceSpec, text: str) -> tuple[str, ...]:
    base_text = _normalize_text(text)
    if not base_text:
        return ()
    candidates = [base_text]
    config = _document_ingestion_config(source)
    if str(config.get("mode", "")).strip().lower() != "paragraph_window":
        return tuple(candidates)

    paragraphs = _document_paragraphs(base_text)
    if not paragraphs:
        return tuple(candidates)

    full_document = "\n\n".join(paragraphs)
    if full_document and full_document != base_text:
        candidates.append(full_document)

    window = max(int(config.get("window_paragraphs", 8)), 1)
    stride = max(int(config.get("stride_paragraphs", max(window // 2, 1))), 1)
    min_chunk_chars = max(int(config.get("min_chunk_chars", 800)), 1)
    max_chunk_chars = max(int(config.get("max_chunk_chars", 6000)), min_chunk_chars)

    for start in range(0, len(paragraphs), stride):
        chunk_parts: list[str] = []
        for paragraph in paragraphs[start : start + window]:
            if paragraph:
                chunk_parts.append(paragraph)
            merged = "\n\n".join(chunk_parts)
            if len(merged) >= min_chunk_chars:
                break
        merged = "\n\n".join(chunk_parts)
        merged = _truncate_chars(merged, max_chunk_chars)
        if len(merged) >= min_chunk_chars:
            candidates.append(merged)

    ordered = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _global_contamination_filter(text: str) -> bool:
    if len(text) < 24:
        return False
    url_count = len(_URL_PATTERN.findall(text))
    if url_count > 8:
        return False
    punctuation_ratio = (
        sum(1 for char in text if char in "#*[]`<>{}") / float(max(len(text), 1))
    )
    if punctuation_ratio > 0.35:
        return False
    return True


def _is_source_allowed(source: DatasetSourceSpec, record: Mapping[str, Any], text: str) -> bool:
    source_id = source.source_id
    metadata = source.metadata

    required_source = str(metadata.get("required_source", "")).strip().lower()
    if required_source:
        actual_source = str(
            record.get("source", record.get("dataset_source", ""))
        ).strip().lower()
        if actual_source and actual_source != required_source:
            return False

    if source_id == "pes2o_s2orc":
        return str(record.get("source", "")).strip().lower() == str(
            metadata.get("required_source", "s2orc")
        ).lower()

    if source_id == "the_stack_code":
        allow_languages = {
            str(lang).strip().lower() for lang in metadata.get("allow_languages", [])
        }
        language = str(record.get("lang", record.get("language", ""))).strip().lower()
        if allow_languages and language and language not in allow_languages:
            return False
        blocked_ext = {str(ext).strip().lower() for ext in metadata.get("blocked_extensions", [])}
        extension = str(record.get("ext", "")).strip().lower()
        if extension and extension in blocked_ext:
            return False
        path_value = str(record.get("path", "")).strip().lower()
        if any(path_value.endswith(ext) for ext in blocked_ext):
            return False
        return True

    if source_id == "open_web_math":
        signal_count = len(_MATH_SIGNAL_PATTERN.findall(text))
        return signal_count >= 1

    if source_id == "lean4_mathlib":
        fact_type = str(record.get("type", "")).strip().lower()
        allowed_fact_types = {
            str(item).strip().lower() for item in metadata.get("allowed_fact_types", [])
        }
        if allowed_fact_types and fact_type and fact_type not in allowed_fact_types:
            return False
        min_chars = int(metadata.get("min_fact_chars", 0))
        if min_chars > 0 and len(text) < min_chars:
            return False
        return True

    if source_id in {"redpajama_arxiv", "redpajama_stackexchange"}:
        subset_field = str(metadata.get("subset_field", "red_pajama_subset"))
        subset_value = str(metadata.get("subset_value", "")).strip().lower()
        actual_subset = str(record.get(subset_field, "")).strip().lower()
        if subset_value and actual_subset != subset_value:
            return False

    if source_id == "redpajama_arxiv":
        patterns = [str(item) for item in metadata.get("logic_patterns", [])]
        lowered = text.lower()
        return any(pattern.lower() in lowered for pattern in patterns)

    if source_id == "redpajama_stackexchange":
        site_allowlist = {str(item).strip().lower() for item in metadata.get("site_allowlist", [])}
        meta_payload = _as_mapping(record.get("meta", {}))
        site = str(
            record.get("domain", record.get("site", meta_payload.get("domain", meta_payload.get("site", ""))))
        ).strip().lower()
        if site_allowlist and site and site not in site_allowlist:
            return False

    if source_id == "openstax_text":
        markers = [str(marker).lower() for marker in metadata.get("procedural_markers", [])]
        lowered = text.lower()
        return any(marker in lowered for marker in markers)

    return True


def prepare_clean_text(source: DatasetSourceSpec, record: Mapping[str, Any]) -> str | None:
    clean_text = _joined_text(source, record)
    if not clean_text:
        clean_text = ""
    for field in _text_field_candidates(source):
        if clean_text:
            break
        raw_text = _to_text(_get_value(record, field))
        clean_text = _normalize_text(raw_text)
    if not clean_text:
        return None

    for candidate_text in _candidate_texts(source, clean_text):
        if not _global_contamination_filter(candidate_text):
            continue
        if not _is_source_allowed(source, record, candidate_text):
            continue
        try:
            enforce_qa_gate(
                candidate_text,
                profile=str(source.metadata.get("qa_gate_profile", "default")),
            )
        except ValueError:
            continue
        return candidate_text
    return None
