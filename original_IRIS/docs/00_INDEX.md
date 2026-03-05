# IRIS Documentation Index

**Document Type:** Design Note (Non-normative)  
**Purpose:** Single entrypoint for the consolidated 10-12 document set  
**Non-Override Clause:** This index does not override authoritative contracts.

---

## 1. Consolidated Main Set (11 Docs)

1. `docs/00_INDEX.md`
2. `docs/01_Architecture_Constitution.md`
3. `docs/02_State_IR_Spec.md`
4. `docs/03_Level_Contracts_L0-L6.md`
5. `docs/04_Credit_Assignment_and_Recovery.md`
6. `docs/05_Eval_Metrics_Spec.md`
7. `docs/06_Regression_and_Phase_Gates.md`
8. `docs/07_Data_Mixture_and_Ingestion.md`
9. `docs/08_Training_Run_Governance.md`
10. `docs/09_Training_Profile_SingleH100_3B.md`
11. `docs/10_Glossary_and_Normative_Status.md`

---

## 2. Reading Paths

### 2.1 New Engineer Path

1. `docs/10_Glossary_and_Normative_Status.md`
2. `docs/01_Architecture_Constitution.md`
3. `docs/02_State_IR_Spec.md`
4. `docs/03_Level_Contracts_L0-L6.md`
5. `docs/06_Regression_and_Phase_Gates.md`

### 2.2 Research/Architecture Path

1. `docs/01_Architecture_Constitution.md`
2. `docs/02_State_IR_Spec.md`
3. `docs/03_Level_Contracts_L0-L6.md`
4. `docs/04_Credit_Assignment_and_Recovery.md`
5. `docs/05_Eval_Metrics_Spec.md`

### 2.3 Training/Operations Path

1. `docs/08_Training_Run_Governance.md`
2. `docs/09_Training_Profile_SingleH100_3B.md`
3. `docs/07_Data_Mixture_and_Ingestion.md`
4. `docs/06_Regression_and_Phase_Gates.md`
5. `docs/05_Eval_Metrics_Spec.md`

---

## 3. Legacy/Removed → Consolidated Mapping

Entries marked **REMOVED (2026-02-27)** no longer exist as files; they are listed only to help migrate older links/notes.

| Legacy Source | Consolidated Destination |
| --- | --- |
| **REMOVED (2026-02-27)** `docs/System Invariants & Non-Negotiables.md` | `docs/01_Architecture_Constitution.md` |
| **REMOVED (2026-02-27)** `docs/What This Model Is Explicitly NOT.md` | `docs/01_Architecture_Constitution.md` |
| **REMOVED (2026-02-27)** `docs/Single Trunk Contract & Allowed Variations.md` | `docs/01_Architecture_Constitution.md` |
| **REMOVED (2026-02-27)** `docs/Routing, Gating, and Control Are Learnable.md` | `docs/01_Architecture_Constitution.md` |
| **REMOVED (2026-02-27)** `docs/State IR Canonical Spec.md` | `docs/02_State_IR_Spec.md` |
| **REMOVED (2026-02-27)** `docs/State IR Examples & Edge Cases.md` | `docs/02_State_IR_Spec.md` |
| **REMOVED (2026-02-27)** `docs/Level Contracts/Level 0–1 Contract.md` | `docs/03_Level_Contracts_L0-L6.md` |
| **REMOVED (2026-02-27)** `docs/Level Contracts/Level 2 Contract.md` | `docs/03_Level_Contracts_L0-L6.md` |
| **REMOVED (2026-02-27)** `docs/Level Contracts/Level 3–4 Contract.md` | `docs/03_Level_Contracts_L0-L6.md` |
| **REMOVED (2026-02-27)** `docs/Level Contracts/Level 5–6 Contract.md` | `docs/03_Level_Contracts_L0-L6.md` |
| **REMOVED (2026-02-27)** `docs/Credit Assignment & Failure Recovery Model.md` | `docs/04_Credit_Assignment_and_Recovery.md` |
| **REMOVED (2026-02-27)** `docs/harness/legacy/metrics.md` | `docs/05_Eval_Metrics_Spec.md` |
| **REMOVED (2026-02-27)** `docs/harness/legacy/regression.md` | `docs/06_Regression_and_Phase_Gates.md` |
| **REMOVED (2026-02-27)** `docs/harness/legacy/phase-gate-policy.md` | `docs/06_Regression_and_Phase_Gates.md` |
| **REMOVED (2026-02-27)** `docs/Data Mixture & Ingestion Specification.md` | `docs/07_Data_Mixture_and_Ingestion.md` |
| **REMOVED (2026-02-27)** `docs/Training Segment and Resume Rules (Design Note).md` | `docs/08_Training_Run_Governance.md` |
| **REMOVED (2026-02-27)** `docs/Pretraining Objective Spec.md` | `docs/08_Training_Run_Governance.md` |
| **REMOVED (2026-02-27)** `docs/plan/reproducibility-and-resume-exactly-once-spec.md` | `docs/08_Training_Run_Governance.md` |
| **REMOVED (2026-02-27)** `docs/plan/runtime-stack-lock-jax-flax-nnx.md` | `docs/08_Training_Run_Governance.md` |
| **REMOVED (2026-02-27)** `docs/plan/single-h100-3b-training-profile.md` | `docs/09_Training_Profile_SingleH100_3B.md` |
| **REMOVED (2026-02-27)** `docs/plan/single-h100-3b-open-decisions.md` | `docs/09_Training_Profile_SingleH100_3B.md` |
| **REMOVED (2026-02-27)** `docs/模型架構設計.md` | `docs/01_Architecture_Constitution.md` (appendix) |

---

## 4. Authority Reminder

- The authoritative normative contracts are:
  - `docs/01_Architecture_Constitution.md`
  - `docs/02_State_IR_Spec.md`
  - `docs/03_Level_Contracts_L0-L6.md`
  - `docs/04_Credit_Assignment_and_Recovery.md`
- Metrics vocabulary and regression workflow are binding and live in `docs/05_Eval_Metrics_Spec.md` and `docs/06_Regression_and_Phase_Gates.md`.

---

## 5. Phase Execution Notes (Non-normative)

- `docs/11_Phase_D_Diagnostics_Design_Note.md`
- `docs/12_Phase_E_Execution_Design_Note.md`
