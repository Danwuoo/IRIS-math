# Credit Assignment and Failure Recovery

**Document Type:** Canonical Specification (Normative)  
**Effective date:** 2026-03-10  
**Authority:** Defines canonical failure taxonomy, credit-routing semantics, and targeted recovery behavior for IRIS-math v2.

---

## 1. Purpose and Scope

This document ensures that:

1. failures are attributed to semantically meaningful levels,
2. verification-driven recovery is targeted instead of blind rerun,
3. learning pressure remains level-addressable,
4. the first-round v2 migration keeps external `L0-L6` stable.

During the first-round v2 design:

- `L3` absorbs recovery / repair policy responsibilities,
- `L6` absorbs learning-signal routing responsibilities.

No new external level ids are introduced in this round.

---

## 2. Core Principles

### 2.1 Semantic Credit, Not Gradient Convenience

Credit assignment is based on the meaning of the failure, not on whichever parameters are easiest to optimize.

### 2.2 Top-Down Diagnosis

Global failure is diagnosed from the verifier side and routed downward.
Lower levels do not declare overall success on their own.

### 2.3 Recovery Is Targeted

The system must not respond to failure by defaulting to "rerun everything."
Recovery must correspond to the suspected failure location.

### 2.4 Recovery and Learning Remain Learned

Recovery selection and learning-signal routing are model behaviors, not engineering-only fallbacks.

---

## 3. Canonical Failure Taxonomy

The external failure codes remain stable.

| Code | Category | Primary Levels | v2 Interpretation |
| --- | --- | --- | --- |
| `F_REP` | Representation Failure | `L0/L1` | Parse, grounding, symbol binding, document-structure or constraint-structure failure |
| `F_PROC` | Procedural Failure | `L2` | Strategy induction or frontier-construction failure |
| `F_SEARCH` | Search / Recovery Failure | `L3` | Budget, branching, backtracking, repair-policy failure |
| `F_MEM` | Memory Failure | `L4` | Retrieval or lemma-applicability failure |
| `F_ABS` | Abstraction Failure | `L5` | Invariant, macro, or compression-granularity failure |
| `F_EVAL` | Evaluation Failure | `L6` | Verification, calibration, or diagnosis failure |

No new top-level failure codes are introduced in this round.

---

## 4. Credit Routing Flow

### 4.1 Directionality

Canonical order:

1. `L6` evaluates outcome validity and local/global risk
2. `L6` emits failure-credit distribution
3. `L3` chooses targeted recovery policy
4. credited levels are re-invoked or adjusted

### 4.2 Failure Credit Vector

`L6` must emit:

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
- hard single-level blame is not the canonical output

---

## 5. Canonical Recovery Actions

| Credited Level | Canonical Recovery Family |
| --- | --- |
| `L0` | reparse document regions, revisit OCR/layout grounding, re-anchor formulas or diagrams |
| `L1` | rebuild symbol table, rescope assumptions, repair constraint graph |
| `L2` | resample strategy families, reopen or rewrite the frontier |
| `L3` | increase or reallocate budget, backtrack, switch strategy, choose targeted repair ordering |
| `L4` | retrieve different lemmas/examples, tighten applicability checks, reject unsafe memory reuse |
| `L5` | adjust abstraction granularity, re-expand compressed reasoning, revise invariants |
| `L6` | re-verify, run stronger contradiction or counterexample probes, recalibrate confidence |

This table defines the allowed family of actions, not a hard-coded flowchart.

---

## 6. Multi-Level Failures

Multi-level credit is allowed and expected.

Examples:

- `F_REP + F_EVAL`: bad document grounding plus verifier overconfidence
- `F_PROC + F_SEARCH`: weak strategy proposal plus poor branch allocation
- `F_MEM + F_ABS`: retrieved lemma mismatch plus over-aggressive abstraction

`L3` decides staged or parallel recovery order using learned policy.

---

## 7. Training-Time Credit Assignment

Training losses must remain level-addressable.

Allowed patterns:

- level-scoped supervision,
- verifier-conditioned self-consistency signals,
- decomposed reward or preference signals,
- targeted replay or contrast training tied to `failure.credit`.

Forbidden patterns:

- opaque single-scalar blame,
- controller-only optimization that treats all other levels as black boxes,
- learning-signal routing that ignores level identity.

---

## 8. Responsibility Boundaries

1. Lower levels must not encode hidden solutions to compensate for higher-level weakness.
2. Higher levels must not mask lower-level defects by infinite retries or looser acceptance thresholds.
3. Retrieval cannot stand in for reasoning without applicability audit.
4. Verification cannot become a post-hoc excuse layer that accepts unsupported proofs.

---

## 9. Observable Signals Required

Every level must expose enough diagnostics for `L6` to reason about:

- uncertainty,
- disagreement,
- collapse indicators,
- calibration,
- recovery usefulness.

Opaque levels are non-compliant.

