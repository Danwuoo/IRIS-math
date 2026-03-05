# Regression and Phase Gates

**Document Type:** Canonical Binding Workflow  
**Metric Vocabulary Source:** `docs/05_Eval_Metrics_Spec.md`  
**Scope:** Regression suites (S1–S8), phase activation (A–E), gate semantics, tolerance/baseline policy, required artifacts, and promotion criteria  
**Source lineage:** Consolidated from legacy regression/phase-gate docs (removed on 2026-02-27).

---

> Purpose question: **「我修了 A，是否悄悄毀了 B？」**

---

## 0. Scope and Authority

This document is **normative for development workflow**.

Any architectural, training, or evaluation-impacting change **MUST** be evaluated against this regression harness **before** being considered valid.

If a change improves aggregate outcome but violates a regression gate defined here, the change is **rejected**.

This workflow is subordinate only to:

- System invariants and non-negotiables
- Trunk and State IR contracts
- Level contracts (L0–L6)
- Credit assignment and failure recovery contract
- Learnable routing/gating/control contract

---

## 1. Regression Philosophy (Non-Negotiable)

### 1.1 Regression ≠ Accuracy

Regression testing answers **stability questions**, not performance questions.

We care about:

- Capability preservation
- Failure localization stability
- Credit routing consistency
- Concept isolation

A model that is *better on average* but *worse in attribution or isolation* is considered **regressed**.

### 1.2 Named Failure Is the Atomic Unit

All regression signals are defined over **failure types**, not tasks.

If a regression cannot be mapped to a known failure category, that indicates:

- Missing failure taxonomy, or
- Insufficient verifier/diagnostic signal

The correct response is **to improve diagnostics**, not to waive regression.

### 1.3 Pretraining-First Priority

Regression prioritizes **process/diagnostic stability** over outcome gains:

- Primary: failure attribution, credit routing, calibration, concept isolation, paired invariance
- Secondary: task success rate / aggregate benchmark score

If secondary improves while primary degrades, the change is rejected.

---

## 2. Phase Definitions (A–E)

Phases define phase-appropriate scope boundaries and gate activation expectations.

### Phase A

Diagnostics-first bootstrap:

- Verifier signals
- Failure tags
- Trace/logging skeleton
- No solver heuristics as primary policy

### Phase B

Tool generation and alignment:

- Failure-tag alignment
- Paired-task plumbing
- No correctness encoded into tool internals

### Phase C

Minimal closed loop in `src/`:

- All Level interfaces L0–L6 present (mounted or stubbed)
- Credit/failure attribution live
- Pretraining-first process diagnostics enforced

### Phase D

Concept diagnostics:

- Concept isolation/leakage monitoring through ConceptARC
- Attribution and diagnostic stability prioritized over leaderboard score

### Phase E

Regression and verifier harness hardening:

- `arc-agi-benchmarking` as regression/verifier probe harness
- No benchmark-shaped architectural shortcuts

---

## 3. Tolerance and Baseline Policy

Each regression run must declare:

- `phase`: one of `A`, `B`, `C`, `D`, `E`
- `baseline_id`: immutable baseline reference
- `tolerance_profile_id`: identifier for epsilon/tolerance configuration

Rules:

1. Tolerances (`epsilon`, calibration tolerance, leakage tolerance) must be fixed per `phase + tolerance_profile_id`.
2. Tolerances cannot be relaxed within the same profile after baseline freeze.
3. Profile changes require explicit annotation in regression artifacts.

---

## 4. Canonical Failure Taxonomy (Regression Keys)

Regression diffs and gates are indexed by canonical failure categories and Level responsibility.

Canonical codes live in `docs/05_Eval_Metrics_Spec.md`:

- `F_REP` (L0/L1)
- `F_PROC` (L2)
- `F_SEARCH` (L3)
- `F_MEM` (L4)
- `F_ABS` (L5)
- `F_EVAL` (L6)

---

## 5. Regression Axes

Every regression run evaluates orthogonal axes. A change is accepted only if it does not regress on any axis, unless explicitly waived.

### 5.1 Dataset Axis

Minimum required coverage (phase-dependent activation):

- MiniARC (sanity & edge cases)
- ARC-AGI-1 (baseline stability)
- ARC-AGI-2 (core stress)
- re-arc paired tasks (representation invariance)
- ConceptARC (concept isolation)
- arc-agi-benchmarking (probe/regression harness only)

### 5.2 Concept Axis

Measured per concept bucket (ConceptARC):

- Single-concept success
- Cross-concept interference
- Concept leakage rate

### 5.3 Level Axis

For each Level (L0–L6):

- Invocation frequency
- Diagnostic signal entropy
- Credit assignment mass

### 5.4 Control Axis

- Budget stability (search depth, beam, rollout)
- Termination consistency
- Sensitivity to noise / seed

---

## 6. Required Regression Suites (S1–S8)

In this section, “mandatory” means mandatory once activated by the current phase profile (see Section 7).

### S1: Smoke Regression (Fast)

Purpose: Detect catastrophic breakage.

Coverage:

- MiniARC (full)
- ARC-AGI-1 (small fixed subset)

Checks:

- Parser validity
- Verifier executable
- No NaNs / crashes
- Basic success rate within tolerance

Gate:

- Any crash or parser failure → **hard block**

### S2: Structural Regression

Purpose: Ensure architectural contracts remain intact.

Checks:

- All Levels instantiated (stub acceptable)
- No token schema drift (State IR ordering and token types)
- Tokenizer protected IR/control strings remain atomic (`rep.tokenizer.ir_fragmentation_rate == 0` once configured)
- No new hard-coded control paths
- Routing outputs remain learnable (non-degenerate)

Gate:

- Any invariant violation → **hard block**

### S3: Failure-Profile Regression

Purpose: Detect silent redistribution of errors.

Method:

- Compare failure taxonomy histograms before/after change
- Measure divergence per failure category (e.g., KL)

Gate:

- Large unexplained shift in failure distribution → **block**
- Shift explained by explicit design goal → **allowed with annotation**

### S4: Concept Isolation Regression

Purpose: Ensure concepts remain independently usable.

Method:

- Run ConceptARC per concept bucket
- Measure:
  - Concept isolation score
  - Concept leakage score

Gate:

- Any concept whose isolation decreases beyond tolerance → **block**

### S5: Paired Representation Regression

Purpose: Detect representation overfitting.

Data:

- re-arc paired tasks (A/B with semantic invariance)

Checks:

- A success but B failure → representation sensitivity
- Divergent internal programs for invariant pairs

Gate:

- Increased asymmetry rate → **block**

### S6: Credit Routing Regression

Purpose: Ensure diagnosis logic remains stable.

Method:

- Compare L6 credit distributions for identical failure cases

Gate:

- Collapse to single-Level blame
- Excessive entropy loss

Either condition → **block**

### S7: Pretraining Diagnostics Regression

Purpose: Keep pretraining process metrics stable across updates.

Method:

- Compare before/after on:
  - `failure.credit.collapse_rate`
  - `eval.calibration_error`
  - `prog.diversity`
  - `rep.tokenizer.ir_fragmentation_rate` (text/IR-control pipelines only)
  - `search.termination_margin` (failure-masking checks)

Gate:

- Credit collapse rate increase beyond tolerance → **block**
- Calibration degradation beyond tolerance → **block**
- Program diversity collapse → **block**
- Any increase in `rep.tokenizer.ir_fragmentation_rate` beyond tolerance → **block**
- Cost gain caused by failure masking → **block**

### S8: Resume Consistency Regression

Purpose: Prevent semantic drift between uninterrupted and resumed training.

Method:

- Fix runtime lock manifest, seed, and dataset slice identity.
- Path A: run `N` segments without interruption.
- Path B1: inject crash during `execute` and resume.
- Path B2: inject crash after `execute` and before checkpoint commit (`pre-commit`).
- Path B3: inject crash after checkpoint commit (`post-commit`) with journal reconciliation.
- For each resumed path, compare segment-aligned distributions at identical boundaries using `task.validity_score`, `task.confidence`, `failure.credit`.
- Emit drift diagnosis labels: `runtime_drift`, `rng_drift`, `data_slice_drift`, `optimizer_state_drift`.

Gate:

- Any drift above epsilon defined in tolerance profile → **block**
- Missing or inconsistent segment boundary alignment → **block**
- Missing crash-class coverage (`execute`, `pre-commit`, `post-commit`) when S8 is ON → **block**
- Missing drift diagnosis labels/artifacts for failed S8 → **block**

---

## 7. Suite Activation Matrix (A–E)

Suites are activated by phase. Activation states:

- `ON` = blocking gate
- `OBSERVE` = collect/report only, non-blocking
- `OFF` = not required in this phase

| Suite ID | Suite Name | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Smoke Regression | ON | ON | ON | ON | ON |
| S2 | Structural Regression | ON | ON | ON | ON | ON |
| S3 | Failure-Profile Regression | OBSERVE | OBSERVE | ON | ON | ON |
| S4 | Concept Isolation Regression | OFF | OBSERVE | ON | ON | ON |
| S5 | Paired Representation Regression | OFF | OBSERVE | ON | ON | ON |
| S6 | Credit Routing Regression | OBSERVE | OBSERVE | ON | ON | ON |
| S7 | Pretraining Diagnostics Regression | OBSERVE | OBSERVE | ON | ON | ON |
| S8 | Resume Consistency Regression | OFF | OFF | ON | ON | ON |

---

## 8. Gate Semantics

Regression gates are **binary** unless explicitly marked otherwise.

### 8.1 Hard Gates (Non-Waivable When Suite Is `ON`)

Hard gates apply when their corresponding suites/metrics are `ON` for the current phase profile:

- System invariant violation
- Token schema drift
- Removal or bypass of a Level interface (including via deleting its I/O contract/stub)
- Hard-coded control replacing learned routing/gating
- Verifier non-functional
- Credit attribution collapse (`failure.credit.collapse_rate` beyond tolerance)
- Calibration degradation (`eval.calibration_error` beyond tolerance)
- Concept leakage increase (`concept.leakage_score` beyond tolerance)
- Paired invariance regression (`paired.invariance.gap` beyond tolerance)
- Failure distribution drift not explained by declared intent
- Any failure category rate increases by > ε without compensating decrease elsewhere (ε from `tolerance_profile_id`)
- Cost decreases only by masking failures (e.g., early termination)
- When S8 is `ON`: resume drift exceeds tolerance, crash-class coverage is incomplete, or resume provenance is missing

### 8.2 Soft Gates (Waivable With Justification)

Soft gate waivers are permitted only with explicit annotation:

- Failure category affected
- Short-term reason
- Intended long-term fix
- Removal criterion

Soft gate examples:

- Minor performance loss with structural gain
- Temporary increase in search cost
- Known diagnostic rebalancing

---

## 9. Required Artifacts (Regression Outputs)

Each regression run **must** produce:

1. **Summary report**
   - Pass/fail per suite
   - Block reasons (if any)
2. **Failure profile diff**
   - Before/after histograms
3. **Concept breakdown**
   - Per-concept isolation & leakage
4. **Credit routing diff**
   - L6 distributions comparison
5. **Resume consistency packet** (required when S8 is ON)
   - Segment journal diff (`PENDING/APPLIED` reconciliation)
   - Checkpoint lineage for last applied segment
   - Runtime lock manifest id/sha used by each run path
   - Drift diagnosis label summary (`runtime/rng/data/optimizer`)

Required run metadata includes (minimum):

- `phase`
- `baseline_id`
- `tolerance_profile_id`

Artifacts are stored under a versioned directory and must be retained.

---

## 10. Promotion Criteria

### A → B

- S1/S2 stable
- Verifier and failure taxonomy logging available

### B → C

- L0–L6 interfaces wired (stub allowed)
- Credit routing vector emitted and serialized
- S3/S6/S7 promoted from observe to blocking

### C → D

- Concept leakage/isolation instrumentation stable
- S4/S5 active and stable under fixed tolerance profile

### D → E

- Resume consistency (S8) stable
- Full regression artifact pipeline retained across runs

---

## 11. Change Classification (Required Declaration)

Every change must declare its expected regression impact:

- Pure refactor (no behavior change expected)
- Targeted fix (which failure category?)
- Capability expansion (which concepts?)

Undeclared changes that cause regressions are **automatically rejected**.

---

## 12. When Regression Fails

If a regression gate blocks a change:

1. Identify **which failure category regressed**
2. Identify **which Level is implicated**
3. Decide one of:
   - Fix the regression
   - Narrow the change scope
   - Improve diagnostics (if attribution is unclear)

Ignoring a regression is **never** acceptable.

---

## 13. Migration Note

Legacy `DevelopmentPlan.md` is intentionally retired (removed on 2026-02-27).
Any previous references must point to this document.

---

## 14. Explicit Non-Goals

This regression harness does **not**:

- Optimize leaderboard scores
- Guarantee monotonic accuracy gains
- Provide human-readable explanations

Its sole role is to preserve architectural integrity over time.

---

## 15. Final Rule

> If you cannot explain why a regression is acceptable in terms of Level responsibility and failure taxonomy, it is not acceptable.
