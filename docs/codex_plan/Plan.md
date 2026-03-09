# Execution Plan (IRIS-math v2)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Living execution plan for the IRIS-math v2 migration and alignment work

---

## 0. Run Header

```text
run_id: codex-2026-03-10-docs-v2-migration-01
date: 2026-03-10
change_class: capability_expansion
phase_target: C
target_failure_categories: F_REP, F_PROC, F_SEARCH, F_MEM, F_ABS, F_EVAL
baseline_id: docs-first-transition-v2
tolerance_profile_id: transition-default
```

---

## 1. Milestones

### M0 — Authority Reset

Goal:

- rewrite `AGENTS.md`, `docs/00_INDEX.md`, and `docs/10_Glossary_and_Normative_Status.md`
- establish active target = IRIS-math v2
- declare documentation-first transition

Status: `Done`

### M1 — Data Constitution + Profile Matrix

Goal:

- replace the fixed mixture baseline with `docs/07_Data_Constitution.md`
- replace the single-H100 note with `docs/09_Training_Profiles_and_Scaling.md`
- update training governance references

Status: `Done`

### M2 — Contract Rewrite

Goal:

- rewrite `docs/02` through `docs/06`
- keep external `L0-L6`, `F_*`, and `S1-S8` stable
- switch semantics to math-native v2

Status: `Done`

### M3 — Architecture Alignment

Goal:

- revise `docs/01_Architecture_Constitution.md`
- preserve single trunk and learnable control invariants
- update benchmark/tool/document positioning

Status: `Done`

### M4 — Operational Guidance + Small Code Sync

Goal:

- update `docs/codex_plan/*`
- update small code surfaces that hardcode mandatory docs or renamed paths

Status: `Done`

### M5 — Next Wave

Goal:

- align `src/` State IR, metrics, and regression logic to the active v2 contracts
- retire explicit baseline adapters once replacements are ready

Status: `Pending`

---

## 2. Explicit Non-Goals for This Wave

- no expansion of external level numbering beyond `L0-L6`
- no new failure taxonomy codes
- no full `src/` semantic rewrite
- no attempt to claim final scaling behavior from the `3B` institution profile

---

## 3. Closure Template

```text
status: Done | Blocked | Cancelled
what_changed: ...
expected_failure_metric_impact: ...
technical_debt_guardrails: ...
regression_status: ...
termination: ...
```
