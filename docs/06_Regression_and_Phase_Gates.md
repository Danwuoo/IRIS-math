# Regression and Phase Gates

**Document Type:** Canonical Binding Workflow  
**Metric Vocabulary Source:** `docs/05_Eval_Metrics_Spec.md`  
**Scope:** Regression suites `S1-S8`, phase activation `A-E`, gate semantics, tolerance policy, required artifacts, and promotion criteria for IRIS-math v2  
**Companion authority:** `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`, `docs/16_Verifier_and_Formalization_Stack.md`, `docs/17_Scaling_Promotion_and_Readiness.md`, `docs/18_Optimization_and_Learning_Contract.md`, `docs/19_Runtime_and_Task_Adjudication_Semantics.md`

---

## 0. Scope and Authority

This document governs how architecture-, data-, training-, parser-, verifier-, and eval-impacting changes are validated.

If a change improves aggregate score but violates an active regression gate, the change is rejected.

The repository is in a documentation-first transition.
Legacy ARC-family probes may still appear as compatibility signals, but they are not sufficient by themselves to define v2 readiness.
Benchmark-family governance details live in `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.
Verifier evidence policy lives in `docs/16_Verifier_and_Formalization_Stack.md`.
Capability-readiness promotion between `P1-P4` lives in `docs/17_Scaling_Promotion_and_Readiness.md`.
Learning-objective bundle semantics live in `docs/18_Optimization_and_Learning_Contract.md`.
Task-family and terminal adjudication semantics live in `docs/19_Runtime_and_Task_Adjudication_Semantics.md`.

---

## 1. Regression Philosophy

### 1.1 Regression Is Not Just Accuracy

Regression answers:

- did we preserve the right capability structure,
- did failure attribution remain meaningful,
- did contamination discipline remain intact,
- did verifier-grounded behavior stay calibrated.

### 1.2 Named Failure Is Still the Atomic Unit

All regression results must be explainable in canonical failure-taxonomy terms.

### 1.3 Process and Governance Stability Come First

Primary priorities:

- failure attribution,
- calibration,
- strategy diversity,
- contamination control,
- provenance coverage,
- resume consistency.

Outcome gains do not waive primary regressions.

---

## 2. Phase Definitions (`A-E`)

### Phase A

Diagnostics bootstrap:

- parser / verifier / provenance instrumentation,
- failure tags,
- no solver heuristics as primary policy.

### Phase B

Data-constitution and benchmark-tier plumbing:

- Tier 1 disclosure,
- homologous split setup,
- paired reformulation plumbing,
- declared default `learning_objective_bundle/v1`,
- no uncontrolled benchmark mixing.

### Phase C

Minimal closed loop:

- all level interfaces `L0-L6` present,
- credit routing live,
- active optimization bundle resolution is auditable,
- outcome-facing eval surfaces resolve task family and task adjudication policy,
- v2 docs active,
- baseline implementation still allowed to lag explicitly.

### Phase D

Document-grounded diagnostic maturity:

- document / OCR / diagram reformulation checks,
- concept isolation,
- contamination audit becomes first-class.

### Phase E

Frontier regression and verifier hardening:

- strict held-out frontier evaluation,
- strong verifier evidence,
- resume consistency and governance artifacts fully required.

---

## 3. Tolerance and Baseline Policy

Each regression run must declare:

- `phase`
- `baseline_id`
- `tolerance_profile_id`

Rules:

1. Tolerances are fixed per `phase + tolerance_profile_id`.
2. Tolerances cannot silently relax inside the same profile.
3. Baseline changes must be explicitly documented.
4. The bootstrap readiness defaults for `tp_p1_bootstrap` through `tp_p4_bootstrap` live in `docs/05_Eval_Metrics_Spec.md`; stricter packets may override them, looser packets must be explicitly named and justified.

---

## 4. Canonical Failure Taxonomy

Regression keys remain:

- `F_REP`
- `F_PROC`
- `F_SEARCH`
- `F_MEM`
- `F_ABS`
- `F_EVAL`

The semantics come from `docs/04_Credit_Assignment_and_Recovery.md`.

---

## 5. Regression Axes

### 5.1 Data Axis

Runs must, where applicable, cover:

- core pool behavior,
- document-native inputs,
- formal / semi-formal tasks,
- declared benchmark tiers,
- archive compatibility probes if still retained.

### 5.2 Reformulation Axis

Check equivalence across:

- textual restatements,
- OCR / clean-text variants,
- diagram / text variants,
- homologous benchmark variants,
- paired tasks retained from legacy probes.

### 5.3 Level Axis

For each `L0-L6`, track:

- invocation behavior,
- diagnostic entropy,
- credited mass,
- mounted vs stub distinction.

### 5.4 Control and Governance Axis

Track:

- budget usage,
- termination behavior,
- recovery quality,
- contamination audit status,
- provenance coverage,
- resume stability.

### 5.5 Contract-Coverage Axis

Track:

- `learning_objective_bundle/v1` resolution,
- task-family resolution coverage,
- `task_adjudication_policy/v1` coverage on outcome-facing eval,
- canonical `runtime_status` / `adjudication_status` usage,
- benchmark-family adjudication overlay usage where relevant.

---

## 6. Required Regression Suites (`S1-S8`)

### S1: Smoke Regression

Purpose:

- detect crash, parser failure, verifier failure, missing artifacts.

Hard block on:

- runtime crash,
- broken parser / verifier path,
- missing required output artifact,
- missing active `learning_objective_bundle_id` on training-impacting runs,
- missing resolved `task_adjudication_policy_id` on outcome-facing eval runs.

### S2: Structural Regression

Purpose:

- ensure the active contracts are still structurally respected.

Checks:

- all `L0-L6` interfaces exist,
- no uncontrolled State IR drift,
- no hard-coded semantic control path replaces learned policy,
- data constitution / profile references are internally consistent,
- `learning_objective_bundle/v1` resolution is unambiguous,
- task-family resolution and task adjudication policy attachment are unambiguous where outcome-facing evaluation is active,
- persisted `runtime_status` and `adjudication_status` use canonical vocabularies.

### S3: Failure-Profile Regression

Purpose:

- detect silent redistribution of errors across the canonical taxonomy.

### S4: Concept Isolation Regression

Purpose:

- ensure math concepts remain usable in isolation and under composition.

Accepted probes may include:

- active math concept buckets,
- document-grounded concept buckets,
- legacy compatibility probes retained during transition.

### S5: Paired Representation Regression

Purpose:

- detect reformulation brittleness.

Examples:

- OCR vs clean text,
- theorem statement restatement,
- diagram vs textual variant,
- legacy paired tasks kept for compatibility.

### S6: Credit Routing Regression

Purpose:

- ensure `failure.credit` remains meaningful and does not collapse.

### S7: Pretraining Diagnostics Regression

Purpose:

- keep process, governance, and contamination signals stable.

Monitored examples include:

- `failure.credit.collapse_rate`
- `eval.calibration_error`
- `prog.diversity`
- `search.termination_margin`
- `rep.tokenizer.ir_fragmentation_rate`
- `contam.strict_holdout_leakage_score`
- `provenance.parser_coverage`

### S8: Resume Consistency Regression

Purpose:

- prevent semantic drift between uninterrupted and resumed runs.

Crash classes remain:

- execute crash,
- pre-commit crash,
- post-commit crash.

---

## 7. Suite Activation Matrix

Activation states:

- `ON` = blocking
- `OBSERVE` = report-only
- `OFF` = not required

| Suite | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| `S1` | ON | ON | ON | ON | ON |
| `S2` | ON | ON | ON | ON | ON |
| `S3` | OBSERVE | OBSERVE | ON | ON | ON |
| `S4` | OFF | OBSERVE | ON | ON | ON |
| `S5` | OFF | OBSERVE | ON | ON | ON |
| `S6` | OBSERVE | OBSERVE | ON | ON | ON |
| `S7` | OBSERVE | OBSERVE | ON | ON | ON |
| `S8` | OFF | OFF | ON | ON | ON |

---

## 8. Hard-Gate Semantics

When a relevant suite is `ON`, the following are hard blocks:

- architectural invariant violation,
- uncontrolled State IR drift,
- deletion or bypass of a level interface,
- deterministic control replacing learned policy,
- verifier non-functionality,
- undeclared or unresolved `learning_objective_bundle/v1` on a training-impacting change,
- outcome-facing evaluation without resolved task family or `task_adjudication_policy/v1`,
- non-canonical persisted `runtime_status` / `adjudication_status`,
- credit collapse,
- calibration degradation beyond tolerance,
- contamination leakage beyond tolerance,
- missing required provenance coverage,
- unexplained failure-distribution drift,
- resume drift beyond tolerance.

Outcome gains do not waive these failures.

---

## 9. Required Artifacts

Each regression run must produce:

1. summary report,
2. failure-profile diff,
3. concept / reformulation breakdown,
4. credit-routing diff,
5. contamination / provenance audit summary when relevant,
6. resume consistency packet when `S8` is active.

Required metadata includes:

- `phase`
- `baseline_id`
- `tolerance_profile_id`
- mandatory docs consulted
- `learning_objective_bundle_id` when training behavior changes
- task-family and adjudication-policy resolution summary when outcome-facing evaluation is active
- runtime/adjudication status breakdown when outcome-facing evaluation is active

---

## 10. Promotion Criteria

These criteria govern **phase promotion** only.
They do not define hardware/profile promotion across `P1-P4`.

### `A -> B`

- parser / verifier / provenance instrumentation is usable,
- `S1` / `S2` stable.

### `B -> C`

- benchmark tiering and contamination disclosure exist,
- `L0-L6` interfaces are wired,
- `failure.credit` is emitted,
- default `learning_objective_bundle/v1` is declared,
- outcome-facing eval surfaces can resolve task family and adjudication policy without ambiguity.

### `C -> D`

- reformulation and document-grounded diagnostics are stable,
- contamination audit is active,
- `S4` / `S5` are stable under fixed tolerances.

### `D -> E`

- strict held-out frontier evaluation is alive,
- verifier evidence is strong,
- `S8` and governance artifacts are stable,
- terminal adjudication artifacts are stable on active held-out surfaces.

---

## 11. Final Rule

If a change cannot explain its effect in terms of failure taxonomy, verifier-grounded diagnostics, and governance discipline, it is not ready for promotion.
