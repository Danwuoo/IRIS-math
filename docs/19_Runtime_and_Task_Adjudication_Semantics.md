# Runtime and Task Adjudication Semantics

**Document Type:** Active Companion Authority  
**Scope:** Runtime-cycle semantics, terminal-status semantics, task-family classification, and task adjudication policy for IRIS-math v2  
**Boundary:** This document defines how runtime attempts are interpreted and how final task outcomes are accepted, rejected, or abstained. It does not replace benchmark tiering, verifier backend policy, or metric definitions.

---

## 0. Purpose and Positioning

IRIS-math v2 spans:

- answer-bearing tasks,
- proof-bearing tasks,
- formalization tasks,
- construction/counterexample tasks,
- document-grounded tasks.

The project therefore needs an explicit authority for:

- what a runtime attempt means,
- when `stop` is success vs non-success,
- how verifier evidence is translated into final task acceptance,
- how answer-only and proof-bearing families differ without fragmenting the architecture.

---

## 1. Runtime Cycle Contract

The runtime need not execute a rigid handwritten loop, but every attempt must remain interpretable in terms of the following semantic moments:

| Semantic moment | Typical owning levels | Minimum visible effect |
| --- | --- | --- |
| ingress / normalization | `L0/L1` | canonical source interpretation, initial `PF`/`SY`/`CG` seeding |
| frontier proposal | `L2` | candidate branches, subgoals, obligations, or strategies in `FR` |
| retrieval / abstraction update | `L4/L5` | `LM`, `FR`, or `CG` patches that change reusable or branch-local structure |
| verification | `L6` | `VS` evidence objects plus validity/risk interpretation |
| control selection | `L3` with verifier-conditioned context | `CS.selected_action`, budgets, escalation, recovery targeting |
| terminal adjudication check | `L6` evidence interpreted through active task adjudication policy | `CS.runtime_status` and `CS.adjudication_state` become terminal-ready or remain unresolved |

Rules:

1. Learned control may revisit, skip, or reorder these semantic moments, but it may not hide them behind an opaque side channel.
2. A runtime attempt is not terminal merely because `selected_action.action_type = stop`; terminality is determined by `CS.runtime_status`.
3. Final task acceptance may occur only when the active task adjudication policy has the evidence it requires.

---

## 2. Runtime Status and Terminal Semantics

The canonical persisted `CS.runtime_status` vocabulary is:

- `in_progress`
- `candidate_ready`
- `accepted`
- `rejected`
- `abstained`
- `budget_exhausted`

Internal aliases are allowed only before normalization.

The canonical persisted `CS.adjudication_state.adjudication_status` vocabulary is:

- `pending`
- `ready`
- `accepted`
- `rejected`
- `abstained`
- `blocked`

Rules:

1. `accepted` means the current candidate satisfies the active task adjudication policy.
2. `rejected` means the current candidate fails the active task adjudication policy on evidence of invalidity, not merely because evidence is incomplete.
3. `abstained` means the system intentionally withholds acceptance because evidence is insufficient or risk is too high.
4. `budget_exhausted` is a non-success terminal status and must not be reported as implicit rejection or implicit acceptance.
5. `candidate_ready` means the system has a candidate output but terminal adjudication has not yet resolved.
6. Persisted logs, checkpoints, eval packets, and dashboards must use the canonical vocabularies above; internal synonyms such as `done`, `fail`, `need_more_check`, or `accepted_with_low_margin` must be normalized before persistence.
7. `candidate_ready` may pair only with adjudication status `pending` or `ready`.
8. `accepted`, `rejected`, and `abstained` runtime statuses must pair with the same adjudication status.
9. `budget_exhausted` must pair with adjudication status `blocked` unless an earlier accepted/rejected/abstained verdict was already finalized.

---

## 3. Task Family Classification

Task-family classification is determined from `PF.task_type`, `required_output.output_kind`, `required_output.answer_channel`, and `required_output.formality_level`.

The canonical task families are:

| Task family | Typical task pattern | Default acceptance posture |
| --- | --- | --- |
| `answer_only` | scalar answer, classification, short witness, or exact expression where answer correctness dominates | answer validity is primary; rationale may be absent |
| `answer_with_rationale` | answer-bearing task where rationale is present but not itself the sole success object | answer validity is primary; rationale must not contain fatal verifier contradiction |
| `proof_natural_language` | informal or natural-language proof is the success object | no critical unresolved obligations; proof evidence must be positive enough for acceptance |
| `proof_semi_formal` | semi-formal proof or proof skeleton is the success object | proof evidence plus bridge-aligned structure must be positive enough for acceptance |
| `formalization` | formal statement/proof artifact is the success object | formal bridge or checker evidence is required |
| `counterexample_or_construction` | construction, witness, or counterexample is the success object | constructed object/witness must satisfy the target spec under checking or constraint validation |

Rules:

1. Benchmark family does not define task family; benchmark family only overlays stricter visibility or evaluation rules.
2. A mixed benchmark family such as `Omni-MATH` must still classify each task instance by task family.
3. `PF.required_output.verifier_mode` may be deferred at creation, but it must resolve before `accepted` or `rejected` is emitted.

### 3.1 Task-Family Resolution Precedence

Canonical resolution order:

1. if the success object is a formal artifact or the declared verifier mode requires a formal checker verdict, classify as `formalization`,
2. else if the success object is a proof artifact and bridge-aligned or semi-formal structure is required for success, classify as `proof_semi_formal`,
3. else if proof completeness determines success, classify as `proof_natural_language`,
4. else if the success object is a witness, construction, or counterexample object, classify as `counterexample_or_construction`,
5. else if answer validity is primary and rationale is present or required but is not itself the sole success object, classify as `answer_with_rationale`,
6. else classify as `answer_only`.

### 3.2 Ambiguity Rules

1. If a final numeric or symbolic answer appears alongside a required proof, the proof-bearing family wins.
2. If an explanation is requested for readability or audit but score is still determined by answer validity, the task remains `answer_with_rationale`.
3. If a benchmark family contains heterogeneous item types, family-level defaults may seed classification, but explicit item-level metadata overrides them.
4. If the classifier cannot resolve the family from `PF.task_type`, `required_output.output_kind`, `required_output.answer_channel`, and `required_output.formality_level`, the eval manifest must provide an explicit `task_family_override`; missing override on an ambiguous item is non-compliant.

Canonical `task_family_resolution_source` values:

- `pf_classifier`
- `item_explicit`
- `eval_manifest_override`
- `benchmark_family_default`

---

## 4. Executable Task Adjudication Contract

Every outcome-facing eval surface should resolve to a `task_adjudication_policy/v1`.

Minimum fields:

| Field | Requirement |
| --- | --- |
| `task_adjudication_policy_id` | immutable policy identity |
| `task_family` | one of the canonical task families or a contract-compatible refinement |
| `verifier_mode` | declared verifier surface expected to judge acceptance |
| `required_evidence_classes` | which canonical verifier evidence-class ids from `docs/16` must be consulted |
| `acceptance_rule` | minimum conditions for `accepted` |
| `rejection_rule` | minimum conditions for `rejected` |
| `abstention_rule` | conditions under which the system must emit `abstained` instead of guessing |
| `escalation_rule` | when stronger verification or search is required before terminal adjudication |
| `document_grounding_requirement` | whether source-grounding is required for acceptance when anchors exist |
| `benchmark_family_overrides` | family-specific tightening rules that do not replace task-family semantics |
| `output_packaging_rules` | what artifact form must be emitted for the adjudicated output |

Rules:

1. `benchmark_family_overrides` may only tighten or specialize the task-family policy; they may not relax strict held-out, verifier, or contamination rules.
2. Training-time weak labels are insufficient for `acceptance_rule`; final acceptance must rely on eval-time evidence or a declared evaluation policy surface.
3. A policy may distinguish between `accepted` and `accepted_with_low_margin` internally, but the external runtime status must still map to the canonical status vocabulary.

### 4.1 Policy Resolution Order

An outcome-facing attempt must resolve to exactly one authoritative task adjudication policy before terminal adjudication.

Canonical resolution order:

1. explicit item-level `task_adjudication_policy_id`,
2. suite-level item override,
3. suite default policy for the resolved `task_family`,
4. benchmark-family default policy or benchmark-family tightening override for the resolved `task_family`,
5. global canonical default policy for that `task_family`.

If no policy resolves after these steps, the attempt is non-compliant and may not emit `accepted` or `rejected`.

Canonical global default policy ids:

| Task family | Default policy id |
| --- | --- |
| `answer_only` | `task-family-answer-only-default-v1` |
| `answer_with_rationale` | `task-family-answer-with-rationale-default-v1` |
| `proof_natural_language` | `task-family-proof-natural-language-default-v1` |
| `proof_semi_formal` | `task-family-proof-semi-formal-default-v1` |
| `formalization` | `task-family-formalization-default-v1` |
| `counterexample_or_construction` | `task-family-counterexample-or-construction-default-v1` |

### 4.2 Registry and Benchmark Attachment Rules

1. Each `task_adjudication_policy/v1` must live in an immutable registry surface resolvable from `task_adjudication_policy_id`.
2. If the policy registry is outside the codebase, the eval artifact must also preserve immutable object ref and content digest.
3. Suite default policies must be published in the eval manifest or equivalent eval-surface registry.
4. Benchmark-family policy objects may attach default adjudication policies and tightening overlays, but they may not override an explicit item contract with a weaker rule.
5. Outcome-facing artifacts must preserve:
   - resolved `task_family`,
   - `task_family_resolution_source`,
   - resolved `task_adjudication_policy_id`,
   - `task_adjudication_policy_resolution_source`,
   - benchmark-family override ref when one was applied.

Canonical `task_adjudication_policy_resolution_source` values:

- `item_policy`
- `suite_item_override`
- `suite_task_family_default`
- `benchmark_family_default`
- `global_task_family_default`

### 4.3 Minimal Example Payload

Illustrative minimum payload:

```json
{
  "schema": "task_adjudication_policy/v1",
  "task_adjudication_policy_id": "answer-only-symbolic-v1",
  "task_family": "answer_only",
  "verifier_mode": "symbolic_answer_check",
  "required_evidence_classes": ["local_validity"],
  "acceptance_rule": "direct answer validation passes and no decisive contradiction remains",
  "rejection_rule": "direct answer validation fails",
  "abstention_rule": "candidate exists but direct validity evidence is missing",
  "escalation_rule": "use stronger verifier when answer validator is unavailable",
  "document_grounding_requirement": "required_when_source_grounded",
  "benchmark_family_overrides": ["aimo-tight-abstention-v1"],
  "output_packaging_rules": "emit final answer plus adjudication summary"
}
```

---

## 5. Canonical Family Adjudication Rules

Unless a stricter policy is declared, `required_evidence_classes` must satisfy the minimum task-family evidence bundle in `docs/16_Verifier_and_Formalization_Stack.md`.

### 5.1 `answer_only`

Default rule:

- accept when the answer-bearing output validates under the declared verifier mode and no decisive contradiction or counterexample remains alive,
- reject when the answer is directly invalidated,
- abstain when the answer candidate exists but the required validity evidence is insufficient.

### 5.2 `answer_with_rationale`

Default rule:

- answer validity remains primary,
- rationale must not contain a decisive contradiction or fatal verifier failure,
- incomplete rationale may reduce confidence or trigger abstention, but it does not automatically force rejection unless the active policy requires proof-complete rationale.

### 5.3 `proof_natural_language`

Default rule:

- accept only when critical proof obligations are discharged or non-critical by policy,
- reject when verifier evidence shows invalidity or contradiction,
- abstain when the proof remains plausible but unresolved.

### 5.4 `proof_semi_formal`

Default rule:

- accept only when proof-level evidence is positive and the semi-formal bridge is not materially blocked,
- reject when the bridge or verifier evidence shows decisive failure,
- abstain when translation or bridge coverage is insufficient for a trustworthy verdict.

### 5.5 `formalization`

Default rule:

- accept only when the required formal checker or bridge verdict succeeds,
- reject when the formal artifact fails or is contradicted by the checker surface,
- natural-language plausibility alone is insufficient for acceptance.

### 5.6 `counterexample_or_construction`

Default rule:

- accept only when the witness or construction satisfies the target spec,
- reject when the witness is invalid or the construction violates constraints,
- abstain when validation is inconclusive.

---

## 6. Document-Grounded Overlay Rules

Document-grounded status is an overlay, not a separate task family.

If a task is document-grounded, the adjudication policy must declare whether acceptance requires:

- source-anchor traceability,
- minimum document-grounding quality,
- absence of blocking grounding warnings,
- stronger reparse/escalation before abstention or rejection.

Rules:

1. A document-grounded task may not be accepted if the active policy requires source grounding and the candidate cannot be traced back to the required anchors.
2. Parse confidence alone is not proof-validity evidence, but blocking grounding failures may prevent terminal acceptance.

---

## 7. Benchmark-Family Overlay Rules

Benchmark-family rules do not replace task-family rules.

Default posture:

- `AIMO` tasks are usually adjudicated as `answer_only` or `answer_with_rationale`,
- `Omni-MATH` tasks may be `answer_only`, `answer_with_rationale`, or proof-bearing depending on the item,
- `miniF2F` tasks are usually `proof_semi_formal` or `formalization`,
- `FrontierMath` tasks must declare their task family explicitly and remain subject to strict held-out posture.

Rules:

1. Benchmark-family overlays may require stricter abstention or stronger verifier modes.
2. Benchmark-family overlays may not turn Tier 2 or Tier 3 into train-visible acceptance teachers.

---

## 8. Required Runtime and Eval Artifacts

Outcome-facing artifacts should preserve at least:

- `task_family`,
- `task_family_resolution_source`,
- `task_adjudication_policy_id`,
- `task_adjudication_policy_resolution_source`,
- external policy ref and digest when the registry is out-of-tree,
- `runtime_status`,
- `adjudication_status`,
- terminal `selected_action`,
- decisive verifier evidence refs or summary,
- benchmark-family override ref when relevant,
- unresolved-obligation summary when relevant,
- document-grounding requirement flag when relevant,
- budget-exhaustion reason when relevant.

These fields are part of the auditable meaning of accepted or rejected outcomes.

---

## 9. Explicit Non-Goals

This document does not:

- define benchmark tiers,
- choose the verifier backend mix,
- require every task to use the same verifier mode,
- replace metric vocabulary or regression gates,
- turn the runtime into a handwritten if/else solver loop.

---

## 10. Related Documents

- `docs/02_State_IR_Spec.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/05_Eval_Metrics_Spec.md`
- `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`
- `docs/16_Verifier_and_Formalization_Stack.md`
