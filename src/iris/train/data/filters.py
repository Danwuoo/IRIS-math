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
    raw_text = _to_text(_get_value(record, source.text_field))
    clean_text = _normalize_text(raw_text)
    if not clean_text:
        return None
    if not _global_contamination_filter(clean_text):
        return None
    if not _is_source_allowed(source, record, clean_text):
        return None
    try:
        enforce_qa_gate(clean_text)
    except ValueError:
        return None
    return clean_text
