from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..schema import ProblemFrame

TASK_FAMILY_IDS: Tuple[str, ...] = (
    "answer_only",
    "answer_with_rationale",
    "proof_natural_language",
    "proof_semi_formal",
    "formalization",
    "counterexample_or_construction",
)
TASK_FAMILY_RESOLUTION_SOURCES: Tuple[str, ...] = (
    "pf_classifier",
    "item_explicit",
    "eval_manifest_override",
    "benchmark_family_default",
)
TASK_ADJUDICATION_POLICY_RESOLUTION_SOURCES: Tuple[str, ...] = (
    "item_policy",
    "suite_item_override",
    "suite_task_family_default",
    "benchmark_family_default",
    "global_task_family_default",
)
DEFAULT_TASK_FAMILY_POLICY_IDS: Dict[str, str] = {
    "answer_only": "task-family-answer-only-default-v1",
    "answer_with_rationale": "task-family-answer-with-rationale-default-v1",
    "proof_natural_language": "task-family-proof-natural-language-default-v1",
    "proof_semi_formal": "task-family-proof-semi-formal-default-v1",
    "formalization": "task-family-formalization-default-v1",
    "counterexample_or_construction": "task-family-counterexample-or-construction-default-v1",
}


class TaskAdjudicationValidationError(ValueError):
    pass


def _require_text(field_name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise TaskAdjudicationValidationError(f"{field_name} is required.")
    return text


def _tuple_of_str(values: Sequence[Any] | None) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class TaskAdjudicationPolicy:
    schema: str
    task_adjudication_policy_id: str
    task_family: str
    verifier_mode: str
    required_evidence_classes: Tuple[str, ...]
    acceptance_rule: str
    rejection_rule: str
    abstention_rule: str
    escalation_rule: str
    document_grounding_requirement: str
    benchmark_family_overrides: Tuple[str, ...]
    output_packaging_rules: str


@dataclass(frozen=True)
class ResolvedTaskSemantics:
    task_family: str
    task_family_resolution_source: str
    task_adjudication_policy_id: str
    task_adjudication_policy_resolution_source: str
    benchmark_family_override_ref: str | None
    policy: TaskAdjudicationPolicy


def _registry_path() -> Path:
    return Path(__file__).resolve().parent / "registries" / "task_adjudication_policies_v1.json"


def _parse_policy(payload: Mapping[str, Any]) -> TaskAdjudicationPolicy:
    policy = TaskAdjudicationPolicy(
        schema=_require_text("task_adjudication_policy.schema", payload.get("schema", "")),
        task_adjudication_policy_id=_require_text(
            "task_adjudication_policy.task_adjudication_policy_id",
            payload.get("task_adjudication_policy_id", ""),
        ),
        task_family=_require_text("task_adjudication_policy.task_family", payload.get("task_family", "")),
        verifier_mode=_require_text("task_adjudication_policy.verifier_mode", payload.get("verifier_mode", "")),
        required_evidence_classes=_tuple_of_str(payload.get("required_evidence_classes", [])),
        acceptance_rule=_require_text(
            "task_adjudication_policy.acceptance_rule",
            payload.get("acceptance_rule", ""),
        ),
        rejection_rule=_require_text(
            "task_adjudication_policy.rejection_rule",
            payload.get("rejection_rule", ""),
        ),
        abstention_rule=_require_text(
            "task_adjudication_policy.abstention_rule",
            payload.get("abstention_rule", ""),
        ),
        escalation_rule=_require_text(
            "task_adjudication_policy.escalation_rule",
            payload.get("escalation_rule", ""),
        ),
        document_grounding_requirement=_require_text(
            "task_adjudication_policy.document_grounding_requirement",
            payload.get("document_grounding_requirement", ""),
        ),
        benchmark_family_overrides=_tuple_of_str(payload.get("benchmark_family_overrides", [])),
        output_packaging_rules=_require_text(
            "task_adjudication_policy.output_packaging_rules",
            payload.get("output_packaging_rules", ""),
        ),
    )
    if policy.schema != "task_adjudication_policy/v1":
        raise TaskAdjudicationValidationError(
            "task adjudication policy schema must be task_adjudication_policy/v1."
        )
    if policy.task_family not in set(TASK_FAMILY_IDS):
        raise TaskAdjudicationValidationError(
            f"Unknown task_family={policy.task_family!r} for policy {policy.task_adjudication_policy_id}."
        )
    if not policy.required_evidence_classes:
        raise TaskAdjudicationValidationError(
            f"Policy {policy.task_adjudication_policy_id} must declare required_evidence_classes."
        )
    return policy


def load_task_adjudication_policy_registry(path: Path | None = None) -> Dict[str, TaskAdjudicationPolicy]:
    registry_path = Path(path) if path is not None else _registry_path()
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise TaskAdjudicationValidationError("task adjudication registry must be a JSON object.")
    policies = payload.get("policies", [])
    if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes, bytearray)):
        raise TaskAdjudicationValidationError("task adjudication registry policies must be an array.")
    parsed = {
        policy.task_adjudication_policy_id: policy
        for policy in (_parse_policy(dict(item)) for item in policies if isinstance(item, Mapping))
    }
    missing_defaults = sorted(
        default_policy_id
        for default_policy_id in DEFAULT_TASK_FAMILY_POLICY_IDS.values()
        if default_policy_id not in parsed
    )
    if missing_defaults:
        raise TaskAdjudicationValidationError(
            f"task adjudication registry is missing canonical default policy ids: {missing_defaults}."
        )
    return parsed


def load_task_adjudication_policy(
    task_adjudication_policy_id: str,
    *,
    registry: Mapping[str, TaskAdjudicationPolicy] | None = None,
) -> TaskAdjudicationPolicy:
    policies = dict(registry or load_task_adjudication_policy_registry())
    try:
        return policies[str(task_adjudication_policy_id)]
    except KeyError as error:
        raise TaskAdjudicationValidationError(
            f"Unresolved task_adjudication_policy_id={task_adjudication_policy_id!r}."
        ) from error


def classify_task_family(
    problem_frame: ProblemFrame,
    *,
    item_task_family: str | None = None,
    eval_manifest_override: str | None = None,
    benchmark_family_default: str | None = None,
) -> tuple[str, str]:
    if item_task_family:
        family = _require_text("item_task_family", item_task_family)
        if family not in set(TASK_FAMILY_IDS):
            raise TaskAdjudicationValidationError(f"Unknown item_task_family={family!r}.")
        return family, "item_explicit"
    if eval_manifest_override:
        family = _require_text("eval_manifest_override", eval_manifest_override)
        if family not in set(TASK_FAMILY_IDS):
            raise TaskAdjudicationValidationError(f"Unknown eval_manifest_override={family!r}.")
        return family, "eval_manifest_override"
    if benchmark_family_default:
        family = _require_text("benchmark_family_default", benchmark_family_default)
        if family not in set(TASK_FAMILY_IDS):
            raise TaskAdjudicationValidationError(
                f"Unknown benchmark_family_default={family!r}."
            )
        return family, "benchmark_family_default"

    task_type = str(problem_frame.task_type).strip().lower()
    output_kind = str(problem_frame.required_output.output_kind).strip().lower()
    answer_channel = str(problem_frame.required_output.answer_channel).strip().lower()
    formality_level = str(problem_frame.required_output.formality_level or "").strip().lower()
    verifier_mode = str(problem_frame.required_output.verifier_mode or "").strip().lower()

    if (
        "formal" in output_kind
        or "formal" in formality_level
        or verifier_mode in {"formal_checker", "formal_bridge", "lean_checker"}
    ):
        return "formalization", "pf_classifier"
    if (
        "semi" in formality_level
        or "semi" in output_kind
        or answer_channel in {"semi_formal_proof", "bridge_aligned_structure"}
    ):
        return "proof_semi_formal", "pf_classifier"
    if task_type in {"prove", "proof"} or output_kind in {"proof", "proof_artifact"}:
        return "proof_natural_language", "pf_classifier"
    if task_type in {"construct", "find_counterexample"} or output_kind in {
        "construction",
        "witness",
        "counterexample",
    }:
        return "counterexample_or_construction", "pf_classifier"
    if answer_channel in {"natural_language", "mixed_output", "rationale"} or output_kind in {
        "answer_with_rationale",
        "explained_answer",
    }:
        return "answer_with_rationale", "pf_classifier"
    return "answer_only", "pf_classifier"


def _benchmark_family_default_for_task(
    benchmark_family_policy: Any | None,
    task_family: str,
    registry: Mapping[str, TaskAdjudicationPolicy],
) -> tuple[str | None, str | None]:
    if benchmark_family_policy is None:
        return None, None
    override_map = dict(getattr(benchmark_family_policy, "benchmark_family_adjudication_overrides", {}) or {})
    override_ref = str(override_map.get(task_family, "")).strip() or None
    for policy_id in tuple(getattr(benchmark_family_policy, "task_adjudication_policy_refs", ())):
        policy = registry.get(str(policy_id))
        if policy is not None and policy.task_family == task_family:
            return policy.task_adjudication_policy_id, override_ref
    return None, override_ref


def resolve_task_adjudication_policy(
    task_family: str,
    *,
    item_policy_id: str | None = None,
    suite_item_override: str | None = None,
    suite_task_family_defaults: Mapping[str, str] | None = None,
    benchmark_family_policy: Any | None = None,
    registry: Mapping[str, TaskAdjudicationPolicy] | None = None,
) -> tuple[TaskAdjudicationPolicy, str, str | None]:
    policies = dict(registry or load_task_adjudication_policy_registry())
    if task_family not in set(TASK_FAMILY_IDS):
        raise TaskAdjudicationValidationError(f"Unknown task_family={task_family!r}.")

    if item_policy_id:
        return load_task_adjudication_policy(item_policy_id, registry=policies), "item_policy", None
    if suite_item_override:
        return (
            load_task_adjudication_policy(suite_item_override, registry=policies),
            "suite_item_override",
            None,
        )
    suite_defaults = dict(suite_task_family_defaults or {})
    if task_family in suite_defaults:
        return (
            load_task_adjudication_policy(suite_defaults[task_family], registry=policies),
            "suite_task_family_default",
            None,
        )

    benchmark_policy_id, benchmark_override_ref = _benchmark_family_default_for_task(
        benchmark_family_policy,
        task_family,
        policies,
    )
    if benchmark_policy_id:
        return (
            load_task_adjudication_policy(benchmark_policy_id, registry=policies),
            "benchmark_family_default",
            benchmark_override_ref,
        )

    default_policy_id = DEFAULT_TASK_FAMILY_POLICY_IDS[task_family]
    return (
        load_task_adjudication_policy(default_policy_id, registry=policies),
        "global_task_family_default",
        None,
    )


def resolve_task_semantics(
    problem_frame: ProblemFrame,
    *,
    item_task_family: str | None = None,
    eval_manifest_override: str | None = None,
    benchmark_family_policy: Any | None = None,
    item_policy_id: str | None = None,
    suite_item_override: str | None = None,
    suite_task_family_defaults: Mapping[str, str] | None = None,
    registry: Mapping[str, TaskAdjudicationPolicy] | None = None,
) -> ResolvedTaskSemantics:
    default_family = None
    if benchmark_family_policy is not None:
        default_map = dict(getattr(benchmark_family_policy, "default_task_family_map", {}) or {})
        default_family = str(default_map.get("default", "")).strip() or None
    task_family, family_source = classify_task_family(
        problem_frame,
        item_task_family=item_task_family,
        eval_manifest_override=eval_manifest_override,
        benchmark_family_default=default_family,
    )
    policy, policy_source, override_ref = resolve_task_adjudication_policy(
        task_family,
        item_policy_id=item_policy_id,
        suite_item_override=suite_item_override,
        suite_task_family_defaults=suite_task_family_defaults,
        benchmark_family_policy=benchmark_family_policy,
        registry=registry,
    )
    return ResolvedTaskSemantics(
        task_family=task_family,
        task_family_resolution_source=family_source,
        task_adjudication_policy_id=policy.task_adjudication_policy_id,
        task_adjudication_policy_resolution_source=policy_source,
        benchmark_family_override_ref=override_ref,
        policy=policy,
    )
