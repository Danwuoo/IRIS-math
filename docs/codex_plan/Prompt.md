# Codex Prompt Pack (IRIS-math v2)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Stable prompt + workflow guidance for using Codex on the IRIS-math v2 repo  
**Non-Override Clause:** This file does not override active contracts.

---

## 0. Intended Workflow

Use these four files as the long-horizon task bundle:

- `docs/codex_plan/Prompt.md`
- `docs/codex_plan/Plan.md`
- `docs/codex_plan/Implement.md`
- `docs/codex_plan/Documentation.md`

The repo is in a documentation-first transition.
Prompt Codex so that it understands:

- active docs are authoritative,
- baseline code may still lag,
- work should move toward IRIS-math v2 unless the task explicitly says otherwise.

---

## 1. Kickoff Prompt

```text
You are an implementation agent for the IRIS-math v2 repo. Follow AGENTS.md exactly.

Mandatory reading before making changes:
- docs/00_INDEX.md
- docs/10_Glossary_and_Normative_Status.md
- docs/11_Goals_and_Success_Criteria.md
- docs/07_Data_Constitution.md
- docs/01_Architecture_Constitution.md
- docs/02_State_IR_Spec.md
- docs/03_Level_Contracts_L0-L6.md
- docs/04_Credit_Assignment_and_Recovery.md
- docs/16_Optimization_and_Learning_Contract.md
- docs/17_Runtime_and_Task_Adjudication_Semantics.md

If the task touches metrics, regression, training governance, scaling, or eval policy, also read:
- docs/05_Eval_Metrics_Spec.md
- docs/06_Regression_and_Phase_Gates.md
- docs/08_Training_Run_Governance.md
- docs/09_Training_Profiles_and_Scaling.md
- docs/12_Multimodal_Document_Pipeline.md
- docs/13_Benchmark_Registry_and_Tiering_Playbook.md
- docs/14_Verifier_and_Formalization_Stack.md
- docs/15_Scaling_Promotion_and_Readiness.md

Active target:
- IRIS-math v2
- document-native, multimodal, verifier-centered
- documentation-first transition: active docs win if code still reflects the baseline scaffold

Non-negotiable constraints:
- one and only one semantic trunk
- canonical State IR only
- external level numbering remains L0-L6 in this round
- routing / gating / recovery / termination must be learned-by-default
- benchmark usage must follow docs/07_Data_Constitution.md
- final task acceptance must follow docs/17_Runtime_and_Task_Adjudication_Semantics.md
- do not silently preserve baseline assumptions when they conflict with active v2 docs

Working style:
- declare the change class
- keep docs/codex_plan/Plan.md current
- record assumptions as 不確定 when needed
- keep code and docs references aligned
```

---

## 2. Workstream Reminders

### 2.1 Documentation Migration

If the task is changing active contracts, update the surrounding authority / glossary / navigation text so the resulting document set stays coherent.

### 2.2 Code Alignment

If code and docs disagree:

- label the mismatch,
- update code toward the active docs when the task scope allows,
- otherwise leave an explicit transition note.

### 2.3 Training and Evaluation Work

Always make tiering, decontamination, parser provenance, formalizer provenance, and verifier provenance explicit.

---

## 3. Validation Reminder

For migration work, prefer checks that answer:

- are old doc paths still used as active references,
- does the phase gate point at the right mandatory docs,
- are archive docs excluded from active reading paths,
- are failure codes and level ids still externally stable,
- do run/eval artifacts resolve `learning_objective_bundle_id` and `task_adjudication_policy_id` unambiguously,
- are runtime/adjudication status vocabularies canonical.
