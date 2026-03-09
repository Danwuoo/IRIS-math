# Benchmark Training and Eval Tiering

**Document Type:** Change Proposal (Requires Approval)  
**Purpose:** Proposed benchmark tiering, contamination control, and held-out
evaluation policy for the IRIS-math transition  
**Non-Override Clause:** This proposal does not approve benchmark training
inclusion or evaluation-policy changes until explicitly approved.

---

## 1. Scope

This proposal covers:

- Benchmark tier definitions
- Train/eval boundary rules
- Contamination and decontamination requirements
- Held-out evaluation planning
- Migration away from benchmark-locked baseline assumptions where necessary

---

## 2. Proposed Direction

IRIS-math benchmark policy should move toward explicit tiers instead of a single
"regression probe only" bucket.

Candidate tiers:

- Regression-only benchmarks
- Diagnostic-development benchmarks
- Candidate train-time benchmark corpora subject to strict decontamination
- Held-out promotion and release benchmarks

Each tier must define:

- Allowed use
- Forbidden use
- Required artifacts
- Contamination reporting expectations

---

## 3. Approved Surface

The following work is approved now:

- Defining benchmark tiers and allowed-use matrices
- Documenting contamination risks and decontamination plans
- Designing held-out evaluation protocols
- Reclassifying existing benchmark assets into proposed tiers
- Planning regression and verifier harness evolution without changing active
  inclusion rules

---

## 4. Blocked Surface

The following work is blocked until approval:

- Moving any benchmark corpus into training
- Relaxing held-out boundaries
- Using benchmark performance as the sole product objective
- Treating decontamination as optional for train-time benchmark use
- Claiming a benchmark tier is active without an approval record

---

## 5. Required Migration Artifacts

This proposal expects the following artifacts before approval:

- Dataset inventory with source and split provenance
- Decontamination protocol and reporting format
- Held-out evaluation plan
- Regression artifact updates for any tier change
- Approval record identifying active tiers and blocked tiers

---

## 6. Open Questions

Label unresolved items as **不確定**, especially for:

- Which benchmark families remain regression-only
- What contamination thresholds are acceptable
- Which verifier-development datasets can be used without harming held-out
  validity
- Promotion rules from proposal to active tier
