# Eval Metrics Spec

**Document Type:** Canonical Binding Vocabulary  
**Audience:** Model / Tool / Evaluation / Training Pipeline  
**Scope:** Metric names, field semantics, and required metadata for logging/artifacts  
**Boundary:** This file defines metric vocabulary and semantics only. Regression suites, phase activation, and gate workflow are defined in `docs/06_Regression_and_Phase_Gates.md`.  
**Source lineage:** Consolidated from a legacy metrics spec (removed on 2026-02-27).

---

## 0. Purpose and Non-Goals

### 0.1 Purpose

This document defines a **single, unified metrics vocabulary** for the entire system, covering:

- Failure taxonomy (semantic, Level-addressable)
- Diagnostic signals emitted by each Level
- Verifier- and controller-consumable scores
- Regression and gate input metrics

The metrics defined here are the **only allowed basis** for:

- Credit routing (L6 → others)
- Recovery strategy selection (via L3)
- Benchmark interpretation (ConceptARC, arc-agi-benchmarking)
- Architecture/training change validation (“did we break something?”)

### 0.2 Explicit Non-Goals

This document does **not**:

- Define loss functions
- Define optimization algorithms
- Define training schedules
- Define regression suites, suite activation matrices, or phase gates (see `docs/06_Regression_and_Phase_Gates.md`)
- Replace Level Contracts or System Invariants

Metrics describe **what is observed**, not **how learning happens**.

---

## 1. Metric Taxonomy Overview

All metrics fall into one of four classes:

1. **Outcome Metrics (Secondary)** – Was the final result acceptable?
2. **Failure Taxonomy Metrics (Primary)** – *Why* did it fail, semantically?
3. **Process / Diagnostic Metrics (Primary)** – What happened internally?
4. **Regression Gate Input Metrics (Hard-Gate Inputs)** – What the gate consumes when activated

Only these four classes are permitted.

Optimization priority is fixed:

- **Primary**: Failure Taxonomy + Process/Diagnostic
- **Secondary**: Outcome (success/score/cost)

---

## 2. Outcome Metrics (Global, Secondary)

### 2.1 Task-Level Outcome

| Metric | Type | Description |
| --- | --- | --- |
| `task.success` | bool | Final output accepted by verifier |
| `task.validity_score` | float ∈ [0,1] | Verifier validity (continuous) |
| `task.confidence` | float ∈ [0,1] | Calibrated confidence (L6) |

Rules:

- `task.success` MUST be derived from verifier logic, not dataset labels directly.
- Confidence MUST be separable from correctness.
- Outcome metrics are probe signals; they are not the primary pretraining objective.

### 2.2 Cost / Efficiency (Secondary)

| Metric | Type | Notes |
| --- | --- | --- |
| `cost.total_steps` | int | Total reasoning cycles |
| `cost.program_proposals` | int | Total L2 proposals |
| `cost.rollout_steps` | int | L1 unroll depth (sum) |
| `cost.retrieval_calls` | int | L4 reads |

These metrics **must never** be used alone to judge model quality.

---

## 3. Failure Taxonomy (Canonical)

Failure taxonomy is **semantic**, **Level-addressable**, and **exclusive** in definition (but credit may be distributed).

### 3.1 Failure Category Codes

| Code | Category | Primary Level |
| --- | --- | --- |
| `F_REP` | Representation Failure | L0 / L1 |
| `F_PROC` | Procedural Failure | L2 |
| `F_SEARCH` | Search / Budget Failure | L3 |
| `F_MEM` | Memory Failure | L4 |
| `F_ABS` | Abstraction Failure | L5 |
| `F_EVAL` | Evaluation / Calibration Failure | L6 |

These codes are **mandatory**. No ad-hoc categories are allowed.

### 3.2 Failure Attribution Vector (Mandatory)

For every failed (or low-confidence) attempt, the system MUST emit:

```text
failure.credit = {
  L0: c0,
  L1: c1,
  L2: c2,
  L3: c3,
  L4: c4,
  L5: c5,
  L6: c6
}
```

Constraints:

- `ck ∈ [0,1]`
- `Σ ck = 1`
- Produced by **L6 Credit Router**
- Consumed by **L3 recovery policy** and training

Hard attribution (single Level) is **not allowed**.

---

## 4. Level-Specific Diagnostic Metrics

### 4.1 Level 0–1 (Representation & Dynamics)

| Metric | Type | Description |
| --- | --- | --- |
| `rep.object.count` | int | Number of object tokens |
| `rep.relation.count` | int | Number of relation tokens |
| `rep.event.count` | int | Number of event tokens |
| `rep.object.entropy` | float | Slot / assignment uncertainty |
| `dyn.violation_score` | float | Constraint / energy violation |
| `dyn.uncertainty` | float | Predictive uncertainty |
| `rep.tokenizer.unk_rate` | float | Fraction of `UNK` tokens in text inputs (0 if none) |
| `rep.tokenizer.ir_fragmentation_rate` | float | Protected IR/control strings split into >1 token |

Interpretation:

- High entropy + downstream failure → `F_REP`
- Low entropy but wrong → likely upstream masking (invalid)
- Non-zero `rep.tokenizer.ir_fragmentation_rate` → control/markup instability (`F_REP` or downstream `F_PROC`)

### 4.2 Level 2 (Program Induction & Execution)

| Metric | Type | Description |
| --- | --- | --- |
| `prog.count` | int | Programs proposed |
| `prog.diversity` | float | Embedding dispersion |
| `prog.exec.success_rate` | float | Partial execution viability |
| `prog.exec.instability` | float | Sensitivity to small perturbations |
| `prog.score.spread` | float | Score variance |

Interpretation:

- Low diversity + failure → premature convergence
- High exec instability → executor semantics problem (`F_PROC`)

### 4.3 Level 3 (Search & Control)

| Metric | Type | Description |
| --- | --- | --- |
| `search.depth.max` | int | Max depth used |
| `search.termination_margin` | float | Stop confidence margin |
| `search.retry_count` | int | Number of retries |
| `search.budget_pressure` | float | Learned budget saturation |

Interpretation:

- Early stop + low confidence → `F_SEARCH`
- Excessive retries → masking upstream failures (flag)

### 4.4 Level 4 (Memory)

| Metric | Type | Description |
| --- | --- | --- |
| `mem.read.k` | int | Retrieved items |
| `mem.read.similarity` | float | Avg similarity |
| `mem.write.gate` | float | Write probability |
| `mem.consolidation.action` | enum | `merge` / `new` / `ignore` |

Interpretation:

- High similarity but failure → stale memory (`F_MEM`)
- Frequent writes → memory pollution risk

### 4.5 Level 5 (Abstraction)

| Metric | Type | Description |
| --- | --- | --- |
| `abs.macro.count` | int | Active macro tokens |
| `abs.granularity` | float | Micro ↔ Macro scale |
| `abs.override_rate` | float | Macro ignored downstream |

Interpretation:

- Macro present but ignored → underpowered abstraction
- Macro dominates failures → over-abstraction (`F_ABS`)

### 4.6 Level 6 (Verification & Calibration)

| Metric | Type | Description |
| --- | --- | --- |
| `eval.false_accept_rate` | float | Invalid accepted |
| `eval.false_reject_rate` | float | Valid rejected |
| `eval.calibration_error` | float | ECE / similar |
| `eval.disagreement` | float | Internal verifier variance |

Interpretation:

- High false accept → `F_EVAL` critical
- High disagreement → unreliable credit routing

---

## 5. Pretraining-Primary Process Metrics

### 5.1 Primary Gate Signals (Pretraining-First)

| Metric | Type | Gate Intent |
| --- | --- | --- |
| `failure.credit.collapse_rate` | float | Credit routing must not collapse |
| `eval.calibration_error` | float | Calibration must not degrade |
| `prog.diversity` | float | Program proposal diversity must persist |
| `concept.leakage_score` | float | Concept leakage must not increase |
| `paired.invariance.gap` | float | Paired invariance must not worsen |
| `search.termination_margin` | float | Avoid failure-masking early stop |
| `process.failure_distribution_entropy` | float | Keep failure diagnostics informative |
| `rep.tokenizer.ir_fragmentation_rate` | float | Protected IR/control strings must remain atomic |

### 5.2 Priority Rule

If an update improves outcome metrics but worsens any primary gate signal, it is treated as **regression**.

---

## 6. ConceptARC-Specific Metrics

ConceptARC is treated as a **diagnostic instrument**, not a leaderboard.

### 6.1 Per-Concept Bucket Metrics

For each concept bucket:

| Metric | Description |
| --- | --- |
| `concept.success_rate` | Accuracy within bucket |
| `concept.isolation_score` | Independence from other concepts |
| `concept.leakage_score` | Performance degradation when mixed |
| `concept.failure_profile` | Distribution over failure codes |

Regression is defined as **worsening isolation or increased leakage**, even if global accuracy improves.

---

## 7. Regression Output Schema (Vocabulary)

Regression suite activation (`ON` / `OBSERVE` / `OFF`) and gate semantics are defined in `docs/06_Regression_and_Phase_Gates.md`.
This section defines **only** the canonical output/logging fields.

### 7.1 Gate Output Schema

```text
regression.status = PASS | FAIL
regression.violations = [
  { metric, delta, phase, suspected_level }
]
```

Rules:

- `phase` MUST be one of `A|B|C|D|E` and MUST match the active run profile.
- No silent passes are allowed: if a suite is executed, it must emit an explicit status record.

---

## 8. Logging and Storage Requirements

- All metrics MUST be serializable (JSON/YAML).
- Per-attempt logs MUST include:
  - State ID
  - Phase
  - Dataset / tool source
  - Full failure credit vector
  - `segment_id` and `optimizer_step_id` (if segmented training is active)
  - `dataset_slice_id` and `data_seed` (if replayable data path is active)
  - `journal_status` (`PENDING|APPLIED`) and `journal_head_hash` (if segment journal is active)
  - `rng_hash_pre` / `rng_hash_post` at segment boundary (if `S8` is active)
  - `resume_path_id` (`uninterrupted|execute_crash|pre_commit_crash|post_commit_crash`) for `S8` runs
- Run metadata MUST include (minimum):
  - `baseline_id`
  - `tolerance_profile_id`
  - `runtime_lock_manifest_id`
  - `runtime_lock_manifest_sha256`
  - `code_version_hash`
  - `config_hash`
  - `tokenizer.vocab_size` (when text pipeline is active)
- Aggregation MUST NOT discard tail failures.

---

## 9. Final Invariants

- No metric may bypass Level identity.
- No failure may be recorded without a taxonomy code.
- No regression may be waived without explicit annotation.

> If a behavior cannot be expressed in these metrics, it does not exist for the system.
