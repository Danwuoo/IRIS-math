# Execution Plan (IRIS-math Transition / Codex)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Living execution plan for the IRIS-math transition  
**Non-Override Clause:** This plan does not approve conflicting changes by
itself. Approval still comes from approved transition specs.

---

## 0) Run Header (fill per Codex run)

```text
run_id:
date:
change_class: pure_refactor | targeted_fix | capability_expansion | contract_migration_proposal
active_workstream: control_plane_realignment | document_parsing_expansion | data_policy_redesign | verifier_search_upgrade | benchmark_tiering_eval_redesign | hardware_profile_routing
active_direction_doc: docs/13_IRIS_Math_v2_Charter.md
relevant_transition_docs:
proposal_or_approved_status:
impacted_baseline_docs:
contamination_eval_risk:
hardware_target_profile:
termination_state:
```

---

## 1) Milestones

### T0 - Control-plane realignment

**Goal:** Make the active control documents point to IRIS-math instead of the
historical baseline.

**Deliverables**

- `AGENTS.md` rewritten for IRIS-math transition control
- `docs/00_INDEX.md` and `docs/10_Glossary_and_Normative_Status.md` synced to
  the new authority map
- `docs/13..16` seed docs created
- `docs/codex_plan/Prompt.md`, `Plan.md`, and `Documentation.md` rewritten
- `docs/codex_plan/Implement.md` marked as historical baseline

**Acceptance**

- Active control docs use consistent authority labels
- Mandatory-reading links resolve
- Active docs no longer present the baseline skeleton as current status

### T1 - Transition proposal approval pass

**Goal:** Convert the seed proposals into decision-ready transition surfaces.

**Deliverables**

- Approval reviews or follow-on approved specs for data, benchmark, and parser
  surfaces
- Explicit approved and blocked surface records for each transition area

**Acceptance**

- Each transition area has a named approval state
- Blocked surfaces are explicit and traceable

### T2 - Data and parse groundwork

**Goal:** Prepare document-native and math-native data handling under approved
surfaces only.

**Deliverables**

- Canonical sidecar design work
- Provenance and contamination reporting templates
- Migration notes from the historical baseline data path

**Acceptance**

- No hidden parser bypasses remain undocumented
- Data and parse work cites the relevant approved or proposal surface

### T3 - Verifier/search upgrade

**Goal:** Advance verifier-centered proof and search work without undocumented
control drift.

**Deliverables**

- Verified workstream plan for verifier/search upgrades
- Attribution, evaluation-risk, and hardware-target notes for each change

**Acceptance**

- Every change cites active specs and workflow docs
- No change relies on hard-coded semantic control as routine policy

### T4 - Benchmark tiering and decontamination

**Goal:** Replace the single baseline benchmark posture with explicit tiers and
held-out discipline.

**Deliverables**

- Tier definitions
- Decontamination plan
- Held-out evaluation plan
- Regression artifact update plan

**Acceptance**

- Benchmark usage is classified by tier
- Any train-time use remains blocked until approved

### T5 - Hardware-scale routing

**Goal:** Make scaling and training plans explicit across the available hardware
profiles.

**Deliverables**

- Profile routing notes for `1x H100 80GB`, `1-8x H200 NVL`, `16x H200 SXM`,
  and `1-8x B200`
- Mapping from current work to the `3B -> 7B -> 14B -> 30B -> 70B -> 120B`
  path

**Acceptance**

- No substantial plan assumes a single hardware profile by default
- Every scaling-sensitive change declares its target profile

---

## 2) Latest Execution Status (2026-03-09)

```text
T0 status: Done
T1 status: Pending
T2 status: Pending
T3 status: Pending
T4 status: Pending
T5 status: Pending
next_active_milestone: T1
```

---

## 3) Explicit Non-Goals for the Transition

- Do not treat AGENTS or a design note as approval for conflicting changes.
- Do not rebuild the baseline skeleton as the default milestone.
- Do not assume benchmark-aware means benchmark-locked.
- Do not assume single-H100 is the only hardware path.

---

## 4) Run Closure Template

```text
status: Done | Blocked | Cancelled
change_class:
active_workstream:
active_specs_consulted:
proposal_or_approved_status:
impacted_baseline_docs:
expected_failure_or_eval_risk_impact:
contamination_eval_risk:
hardware_target_profile:
what_changed:
validation:
next_step:
```
