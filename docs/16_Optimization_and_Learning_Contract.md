# Optimization and Learning Contract

**Document Type:** Active Companion Authority  
**Scope:** Learning-objective families, curriculum activation, level-addressable optimization, and optimization-bundle semantics for IRIS-math v2  
**Boundary:** This document defines how the system may be trained while staying attributable. It does not freeze optimizer brands, scheduler brands, or model-size-specific hyperparameters.

---

## 0. Purpose and Positioning

IRIS-math v2 already defines:

- architecture,
- State IR,
- levels,
- credit routing,
- data governance,
- verifier evidence,
- scaling/readiness.

This document closes the remaining theory gap between those contracts and actual learning behavior.

It defines:

- which objective families are canonical,
- how objective families attach to `L0-L6` and State IR surfaces,
- how control, recovery, retrieval, abstraction, and verification remain learned behaviors,
- how curriculum activation must stay phase-appropriate,
- what metadata is required to make optimization changes auditable.

---

## 1. Core Principles

### 1.1 Level-Addressable Optimization

Optimization may be multi-objective, but it may not collapse all learning pressure into one opaque outcome scalar.

Every active optimization program must preserve:

- level identity,
- failure-taxonomy identity,
- verifier-conditioned attribution,
- auditability of which surfaces are being optimized.

### 1.2 State-Transition Learning Comes Before Outcome-Only Tuning

The trunk must learn to form, patch, and use State IR, not merely to emit benchmark-shaped outputs.

Outcome-side gains are valid only when they remain compatible with:

- State IR integrity,
- controllable search/recovery,
- verifier evidence,
- contamination discipline.

### 1.3 Learned Control and Recovery Remain First-Class

Control, stopping, backtracking, reparse choice, retrieval usage, and repair targeting must remain learned surfaces.

Optimization that only trains final answers while leaving control behavior as untrained plumbing is non-compliant with the active target.

### 1.4 Verifier-Conditioned Learning Is Explicit

Verifier-conditioned learning is allowed and expected, but it must remain explicit about:

- evidence quality,
- evidence provenance,
- whether the signal is training-time weak supervision or eval-time acceptance evidence,
- which level surfaces the signal is meant to improve.

### 1.5 Phase-Appropriate Activation

Earlier phases may mount fewer objective families, but they may not smuggle late-phase benchmark chasing into the program through undeclared objective bundles.

---

## 2. Canonical Objective Families

The active objective families are grouped by the level/responsibility surfaces they primarily train.

### 2.1 `obj.rep.state_construction`

Primary levels:

- `L0`
- `L1`

Primary surfaces:

- `PF`
- `SY`
- `CG`
- source anchors
- document-grounding fidelity

Typical signals:

- parser-aligned structuring labels,
- symbol binding labels,
- scope/type consistency labels,
- constraint reconstruction targets,
- document-grounding or anchor-alignment supervision.

### 2.2 `obj.proc.frontier_induction`

Primary level:

- `L2`

Primary surfaces:

- `FR.branch`
- `FR.subgoal`
- `FR.obligation`
- `FR.strategy_candidate`

Typical signals:

- strategy-family labels,
- frontier expansion quality,
- branch diversity pressure,
- non-collapse preference signals across plausible approaches.

### 2.3 `obj.search.control_and_recovery`

Primary level:

- `L3`

Primary surfaces:

- `CS.selected_action`
- `CS.runtime_status`
- budget allocation
- repair targeting

Typical signals:

- verifier-conditioned control targets,
- targeted recovery success/failure,
- stop-vs-continue preferences,
- budget-allocation or branch-choice supervision,
- failure-credit-targeted replay.

### 2.4 `obj.mem.applicability`

Primary level:

- `L4`

Primary surfaces:

- `LM.binding_map`
- `LM.applicability_audit`
- retrieval rejection/acceptance behavior

Typical signals:

- applicability labels,
- mismatch contrast pairs,
- safe-reuse vs unsafe-reuse preference signals,
- provenance-aware retrieval audits.

### 2.5 `obj.abs.compression`

Primary level:

- `L5`

Primary surfaces:

- `LM` entries with `memory_kind = derived_abstraction` or equivalent,
- branch-scoped `FR.hypothesis` / `FR.strategy_candidate` abstractions,
- `CG` invariant relations emitted from abstraction.

Typical signals:

- invariant usefulness,
- abstraction reuse,
- compression-vs-override tradeoff,
- verifier-compatible abstraction preference,
- expansion-then-recompression replay.

Rules:

1. L5 objectives may not reward compression that erases proof conditions.
2. A behavior-affecting abstraction is only optimization-valid if it lands canonically in State IR.

### 2.6 `obj.eval.verification_and_calibration`

Primary level:

- `L6`

Primary surfaces:

- `VS`
- `failure.credit`
- calibrated confidence
- false-accept / false-reject behavior

Typical signals:

- local-validity labels,
- proof-gap labels,
- counterexample contrast pairs,
- calibration losses,
- failure-credit supervision or preference targets.

### 2.7 `obj.task.outcome`

Primary levels:

- cross-level, interpreted through `L6` and the active task adjudication policy

Primary surfaces:

- final answer/proof/formal artifact validity,
- accepted-vs-rejected task outcome,
- outcome confidence under the declared task family

Typical signals:

- final-answer validity,
- proof-validity or formal-check success,
- construction/witness acceptance,
- abstain-vs-invalid preference.

Rules:

1. Outcome objectives may not be the only active objective family in a program that claims IRIS-math v2 alignment.
2. Benchmark-facing outcome optimization must remain subordinate to the benchmark and tier rules in `docs/07` and `docs/15`.

---

## 3. Executable Optimization Contract

Every active training program should resolve to a `learning_objective_bundle/v1`.

Minimum fields:

| Field | Requirement |
| --- | --- |
| `learning_objective_bundle_id` | immutable bundle identity |
| `profile_id` | active profile family such as `P1`, `P2`, `P3`, or `P4` |
| `phase` | active `A-E` phase |
| `objective_families[]` | active objective families with weights or bands plus activation status |
| `level_surface_map` | which State IR / diagnostic surfaces each objective family is allowed to update |
| `control_learning_mode` | how `L3` is trained without collapsing into deterministic policy |
| `verifier_conditioning_mode` | how verifier signals participate in optimization |
| `failure_replay_policy` | how `failure.credit` is used for targeted replay or contrast sampling |
| `curriculum_policy_summary` | declared curriculum and activation boundaries |
| `benchmark_visibility_guardrails` | declaration that benchmark-aware optimization remains inside family/tier rules |
| `partial_mount_overrides` | any objective families constrained by mounted vs `partial_mount` vs stub realities |

Rules:

1. A run may not silently change objective families, weights, or benchmark visibility without a new `learning_objective_bundle_id`.
2. `control_learning_mode` and `verifier_conditioning_mode` are behavior-affecting surfaces and therefore reproducibility-relevant.
3. A bundle may temporarily downweight a surface, but it may not claim learned maturity for a level whose objective family remains inactive or stub-constrained.

### 3.1 Resolution Order

An active training run must resolve to exactly one authoritative `learning_objective_bundle/v1` before the first optimizer step.

Canonical resolution order:

1. explicit run-manifest `learning_objective_bundle_id`,
2. otherwise the declared default bundle for the active `profile_id + phase`,
3. otherwise the run is non-compliant.

The `profile_id + phase -> learning_objective_bundle_id` default mapping must be published in the active training-profile registry or run-configuration registry and versioned with the codebase.

Rules:

1. Runtime config may not merge multiple bundles at execution time.
2. Ad hoc CLI, notebook, or environment overrides to objective-family weights, activation states, benchmark visibility, or verifier-conditioning semantics are forbidden once a bundle is resolved.
3. If an experiment needs different optimization behavior, it must publish a new bundle id rather than patching a resolved bundle in memory.

### 3.2 Inheritance and Versioning

Bundles may declare lineage, but active resolution is always by exact bundle object.

Optional lineage fields:

- `parent_bundle_id`
- `lineage_note`

Rules:

1. `parent_bundle_id` is documentary lineage only; runtime may not reconstruct the active bundle by parent-plus-delta merge.
2. A new `learning_objective_bundle_id` is required whenever any behavior-affecting field changes, including:
   - objective-family membership,
   - activation status,
   - numeric weights or bands,
   - level-surface permissions,
   - control-learning mode,
   - verifier-conditioning mode,
   - failure-replay policy,
   - benchmark-visibility guardrails,
   - `partial_mount_overrides`.
3. Non-behavioral prose may be revised outside the bundle object without changing the id, but the executable object keyed by the id must remain immutable.
4. Profile-default changes take effect only by publishing a new bundle id and updating the profile/phase default mapping.

### 3.3 Registry and Replay Surface

Each `learning_objective_bundle/v1` must live in an immutable registry surface resolvable from `learning_objective_bundle_id`.

Canonical engineering rules:

1. If the registry object is version-controlled with the codebase, `learning_objective_bundle_id + code_version_hash` is sufficient for replay.
2. If the registry object lives outside the codebase, the run artifact must also preserve:
   - immutable object ref or fetch ref,
   - content digest,
   - retrieval provenance sufficient for audit.
3. A journal, checkpoint, or readiness packet may not cite a bundle id that cannot be resolved at review time.
4. Bundle registries may be implemented in JSON, TOML, YAML, or code-generated manifests, but the resolved object must be schema-stable and auditable.

### 3.4 Minimal Example Payload

Illustrative minimum payload:

```json
{
  "schema": "learning_objective_bundle/v1",
  "learning_objective_bundle_id": "p2-phase-c-bundle-v1",
  "profile_id": "P2",
  "phase": "C",
  "objective_families": [
    {"family": "obj.rep.state_construction", "status": "active", "weight_band": "high"},
    {"family": "obj.search.control_and_recovery", "status": "active", "weight_band": "medium"},
    {"family": "obj.task.outcome", "status": "active", "weight_band": "low"}
  ],
  "level_surface_map": {
    "obj.rep.state_construction": ["PF", "SY", "CG"],
    "obj.search.control_and_recovery": ["CS.selected_action", "CS.runtime_status"],
    "obj.task.outcome": ["accepted_outcome"]
  },
  "control_learning_mode": "verifier_conditioned_learned_policy",
  "verifier_conditioning_mode": "explicit_eval_evidence_plus_training_labels",
  "failure_replay_policy": "credit_targeted_replay_v1",
  "curriculum_policy_summary": "phase-c-default",
  "benchmark_visibility_guardrails": "registered-family-only",
  "partial_mount_overrides": [],
  "parent_bundle_id": "p2-phase-b-bundle-v3"
}
```

---

## 4. Phase Activation and Curriculum Rules

| Phase | Required emphasis | Blocked shortcuts |
| --- | --- | --- |
| `A` | instrumentation, `obj.rep.state_construction`, bootstrap verifier labels, observability surfaces | benchmark-heavy outcome optimization, hidden solver heuristics as primary policy |
| `B` | explicit data/tier wiring, limited `obj.task.outcome`, curriculum and benchmark visibility declaration | undeclared Tier 1 mixing, Tier 2/Tier 3-driven tuning |
| `C` | full `L0-L6` objective addressability, `obj.search.control_and_recovery`, `obj.mem.applicability`, `obj.eval.verification_and_calibration` all live enough for attribution | control-only black-box optimization, opaque single-scalar blame |
| `D` | document robustness, paired reformulation, stronger `obj.abs.compression`, verifier-conditioned recovery quality | cosmetic document parsing, abstraction without canonical landing |
| `E` | frontier-facing outcome objectives, proof-validity emphasis, hard verifier/calibration discipline, strict governance replayability | leaderboard-only optimization, hidden Tier 3 adaptation, frontier claims without verifier maturity |

Curriculum rules:

1. Curriculum may change emphasis only through declared objective bundles and benchmark visibility rules.
2. Failure-targeted replay is allowed only when tied to explicit `failure.credit` or other declared attribution surfaces.
3. Weak supervision may shape early training, but it may not silently redefine final proof truth.

---

## 5. Allowed and Forbidden Optimization Patterns

Allowed patterns:

- supervised losses,
- contrastive or pairwise objectives,
- decomposed reward or preference objectives,
- verifier-conditioned self-training,
- failure-targeted replay,
- abstraction expand/compress replay,
- calibration objectives,
- curriculum changes tied to declared policy bundles.

Forbidden patterns:

- opaque single-scalar reward as the sole active objective surface,
- hidden benchmark-family-specific objective shaping outside declared tier policy,
- permanent deterministic teacher forcing for control/recovery semantics,
- learning-signal routing that ignores level identity,
- optimizing `obj.abs.compression` without canonical State IR landing.

---

## 6. Required Optimization Artifacts

A run that changes learning behavior must preserve at least:

- `learning_objective_bundle_id`,
- `learning_objective_bundle_resolution_source`,
- active objective-family summary,
- per-level objective activation summary,
- control-learning mode summary,
- verifier-conditioning mode summary,
- failure-replay summary,
- benchmark visibility summary,
- external bundle ref and digest when the registry is out-of-tree,
- any `TEMPORARY TECHNICAL DEBT` guardrails that constrain learning.

These artifacts are reproducibility-relevant and must be preserved alongside the normal governance packet.

---

## 7. Explicit Non-Goals

This document does not:

- freeze one optimizer or scheduler brand,
- define numeric loss weights for every run,
- force one curriculum recipe across all profiles,
- replace the benchmark or data constitutions,
- authorize outcome-only training as a substitute for learned State IR behavior.

---

## 8. Related Documents

- `docs/01_Architecture_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/07_Data_Constitution.md`
- `docs/08_Training_Run_Governance.md`
- `docs/14_Verifier_and_Formalization_Stack.md`
- `docs/17_Runtime_and_Task_Adjudication_Semantics.md`
