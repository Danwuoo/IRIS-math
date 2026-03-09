# Glossary and Normative Status

**Document Type:** Design Note (Non-normative)  
**Purpose:** Shared terminology and authority labels to prevent document drift

---

## 1. Normative Status Labels

Use these labels consistently:

- **Authoritative Specification (Normative):** system-level contracts; violation is invalid architecture.
- **Normative Contract:** binding behavioral/interface constraints.
- **Canonical Binding (Workflow/Vocabulary):** binding policy for metrics/regression/gating vocabulary and process.
- **Design Note (Non-normative):** implementation guidance; cannot override contracts.
- **Change Proposal (Requires Approval):** candidate changes to contracts/policies; not active until approved.

Conflict rule:

- Normative contracts override design notes and planning notes.

---

## 1.1 This Repo’s Authority Map (Canonical)

**Normative (architecture contracts):**

- `docs/01_Architecture_Constitution.md` (Authoritative Specification)
- `docs/02_State_IR_Spec.md` (Canonical Specification)
- `docs/03_Level_Contracts_L0-L6.md` (Normative Contract)
- `docs/04_Credit_Assignment_and_Recovery.md` (Canonical Specification)

**Canonical binding (process/vocabulary):**

- `docs/05_Eval_Metrics_Spec.md` (metric vocabulary + logging schema)
- `docs/06_Regression_and_Phase_Gates.md` (suites, phase activation, gates, artifacts, promotion criteria)
- `docs/07_Data_Mixture_and_Ingestion.md` (stability-critical training mixture + ingestion policy)

**Policy-binding (engineering governance):**

- `docs/08_Training_Run_Governance.md` (segment transactions, resume exactly-once, runtime lock, S8 drift diagnostics)

**Design notes:**

- `docs/00_INDEX.md` (navigation)
- `docs/09_Training_Profile_SingleH100_3B.md` (single-card fixed values; non-normative profile)
- `docs/10_Glossary_and_Normative_Status.md` (this file)

---

## 2. Core Terms

- **Trunk:** primary parameter-bearing cognitive substrate.
- **Second trunk:** competing high-capacity parallel semantic brain (forbidden).
- **State IR:** canonical internal representation with fixed token categories.
- **Program IR:** procedural representation outside State IR.
- **Level (L0-L6):** semantic responsibility interfaces in the architecture.
- **Stub:** disabled-mode interface-preserving no-op/low-capacity adapter.
- **Router/Gate:** learnable control signals for invocation/fusion/selection.
- **Guardrail:** non-semantic safety/resource bound.
- **Technical debt guardrail:** temporary hard control with removal criterion.
- **Failure taxonomy:** canonical failure categories (`F_REP`, `F_PROC`, `F_SEARCH`, `F_MEM`, `F_ABS`, `F_EVAL`).
- **Credit vector:** L6-emitted failure responsibility distribution.
- **Phase A-E:** lifecycle stage used for suite activation and gate policy.
- **S1-S8:** canonical regression suites.

---

## 3. Boundary Rules

1. Contracts are written once and referenced, not redefined by divergent wording.
2. Metrics spec defines vocabulary, not suite activation logic.
3. Regression/phase doc defines suite activation and gate behavior, not metric definitions.
4. Training governance defines transaction/resume/runtime lock rules, not profile fixed values.
5. Profile doc defines fixed run values, not global governance semantics.
