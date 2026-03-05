# Documentation / Decisions Log (IRIS / Codex)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Keep a durable record of decisions, assumptions (`不確定`), and per-change completion checklists for Codex runs  
**Non-Override Clause:** This file does not override normative contracts.

---

## 0) Current Status (keep short)

```text
current_phase: C
current_milestone: M5 completed (Phase A->C skeleton baseline in place)
next_action: expand mounted (non-stub) level implementations while preserving S1/S2 stability
```

---

## 1) Assumptions / Unknowns (label 不確定)

```text
- 不確定: JAX + Flax NNX GPU runtime is available in this Windows environment.
  Evidence: local Python (3.10) does not currently have jax/flax installed; smoke runs used numpy backend.
  Fallback: keep all smoke/structural/train checks runnable on CPU numpy path; use --device gpu command path as compatibility hook.
```

---

## 2) Decision Records

## [2026-02-28] Decision: Build a contract-first Phase C baseline skeleton

Decision:
- Implemented `src/iris/` skeleton covering State IR schema, L0-L6 interfaces/stubs, single trunk, canonical metrics helpers, and toy segmented training with resume journal.

Rationale:
- Satisfies Phase C requirements in `docs/06_Regression_and_Phase_Gates.md` (interfaces present, attribution/logging live, structural checks runnable).

Alternatives considered:
- Build only partial modules first. Rejected because it would leave Level interface and regression scaffolding incomplete.

Contract impact (docs/01..04):
- Preserves single-trunk constraint.
- Preserves canonical State IR token set `{T,G,O,R,X,M}` and order.
- Keeps L0-L6 interfaces present with explicit stub behavior and diagnostics.
- Uses learnable control logits from trunk outputs; no deterministic semantic scheduler introduced.

Workflow impact (docs/05..08):
- Adds canonical metric-name logging stubs including `failure.credit`.
- Adds S1/S2 scripts and minimal resume journal semantics (`PENDING`/`APPLIED`).
- Keeps benchmark data out of training path (toy synthetic only).

Follow-ups:
- Add richer mounted level implementations and regression artifacts for S3/S6/S7 when phase scope expands.

---

## 3) Per-Change Completion Checklist (copy into PR / run notes)

```text
Mandatory docs consulted:
- docs/10_Glossary_and_Normative_Status.md
- docs/01_Architecture_Constitution.md
- docs/02_State_IR_Spec.md
- docs/03_Level_Contracts_L0-L6.md
- docs/04_Credit_Assignment_and_Recovery.md
Policy docs consulted (if applicable):
- docs/05_Eval_Metrics_Spec.md
- docs/06_Regression_and_Phase_Gates.md
- docs/07_Data_Mixture_and_Ingestion.md
- docs/08_Training_Run_Governance.md
- docs/09_Training_Profile_SingleH100_3B.md

Change class: capability_expansion
Expected failure-category / metric impact (use canonical names): improve attribution/readiness signals for F_REP, F_PROC, F_SEARCH, F_EVAL without benchmark-optimized claims
Technical debt guardrails introduced (with removal criteria): none
Regression status (what ran / what is expected to pass): pytest, S1 smoke, S2 structural, toy train/eval, crash+resume replay demonstration
Termination: Done
```
