# IRIS-math Documentation Index

**Document Type:** Design Note (Non-normative)  
**Purpose:** Single entrypoint for the active IRIS-math transition control set,
workflow docs, and historical baseline references  
**Non-Override Clause:** This index does not override approved transition specs,
canonical workflow bindings, or approved proposals.

---

## 1. Active Transition Control Set

These docs define the current project direction and transition surfaces:

1. `docs/00_INDEX.md`
2. `docs/10_Glossary_and_Normative_Status.md`
3. `docs/13_IRIS_Math_v2_Charter.md`
4. `docs/14_IRIS_Math_Data_Constitution_v2.md`
5. `docs/15_Benchmark_Training_and_Eval_Tiering.md`
6. `docs/16_Document_Math_Parse_Canonical_Format.md`

Current status:

- `docs/13_IRIS_Math_v2_Charter.md` is the active direction charter.
- `docs/14..16` are transition proposals and do not by themselves approve
  conflicting implementation work.
- No approved IRIS-math replacement spec has yet superseded the historical
  baseline architecture contracts.

---

## 2. Canonical Workflow and Governance Docs Still in Use

These remain active workflow references unless an approved transition spec
explicitly supersedes a surface:

1. `docs/05_Eval_Metrics_Spec.md`
2. `docs/06_Regression_and_Phase_Gates.md`
3. `docs/08_Training_Run_Governance.md`

---

## 3. Historical Baseline References

These docs describe the baseline IRIS stack and are retained for compatibility
checks, migration impact analysis, and blocked-surface traceability:

1. `docs/01_Architecture_Constitution.md`
2. `docs/02_State_IR_Spec.md`
3. `docs/03_Level_Contracts_L0-L6.md`
4. `docs/04_Credit_Assignment_and_Recovery.md`
5. `docs/07_Data_Mixture_and_Ingestion.md`
6. `docs/09_Training_Profile_SingleH100_3B.md`
7. `docs/11_Phase_D_Diagnostics_Design_Note.md`
8. `docs/12_Phase_E_Execution_Design_Note.md`

These are not the default build target for new IRIS-math work.

---

## 4. Operational Planning Bundle

The active Codex planning bundle lives under `docs/codex_plan/`:

- `docs/codex_plan/Prompt.md`
- `docs/codex_plan/Plan.md`
- `docs/codex_plan/Documentation.md`

Historical baseline runbook:

- `docs/codex_plan/Implement.md`

---

## 5. Reading Paths

### 5.1 IRIS-math Control-Plane Path

1. `docs/10_Glossary_and_Normative_Status.md`
2. `docs/13_IRIS_Math_v2_Charter.md`
3. `docs/14_IRIS_Math_Data_Constitution_v2.md`
4. `docs/15_Benchmark_Training_and_Eval_Tiering.md`
5. `docs/16_Document_Math_Parse_Canonical_Format.md`
6. Relevant workflow docs: `docs/05`, `docs/06`, `docs/08`
7. Historical baseline docs only as needed

### 5.2 Data / Parser / Eval Path

1. `docs/13_IRIS_Math_v2_Charter.md`
2. `docs/14_IRIS_Math_Data_Constitution_v2.md`
3. `docs/15_Benchmark_Training_and_Eval_Tiering.md`
4. `docs/16_Document_Math_Parse_Canonical_Format.md`
5. `docs/05_Eval_Metrics_Spec.md`
6. `docs/06_Regression_and_Phase_Gates.md`
7. `docs/08_Training_Run_Governance.md`

### 5.3 Historical Compatibility Path

1. `docs/10_Glossary_and_Normative_Status.md`
2. Relevant active transition docs
3. Relevant historical baseline docs for impact analysis

---

## 6. Authority Reminder

- AGENTS or a design note cannot by themselves legalize conflicting
  architecture, data, parser, or evaluation changes.
- Approved transition specs are the only documents that can replace historical
  baseline authority for a given IRIS-math surface.
- Transition proposals control planning and migration behavior but are not
  approval on their own.
- Canonical workflow docs remain active unless explicitly superseded.

---

## 7. Historical Mapping Note

The repo was previously organized around a consolidated baseline IRIS stack
(`docs/01..12`). That baseline material remains in the repo for reference, but
the active direction is now governed by the transition-control set in Section
1.
