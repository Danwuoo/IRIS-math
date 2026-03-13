# AGENTS.md (for IRIS-math v2)
Project: IRIS-math v2

## 0) Non-Negotiable Operating Mode
You are an implementation agent for **IRIS-math v2**.
The active target is a **document-native, multimodal, verifier-centered mathematical foundation model**.

This repository is in a **documentation-first transition**:

- Active v2 docs are the source of truth.
- Existing `src/` code may still reflect baseline IRIS scaffolding.
- When baseline code or archived notes conflict with active v2 docs, you must **explicitly label the conflict** and follow the active v2 track.
- You must not silently preserve the baseline worldview just because the code still does.

Before any development work, complete the mandatory reading in Section 1 and the relevant policy reading in Section 2.

When uncertain, explicitly label **不確定**.

Conflict rule:

- System-level invariants and active authoritative contracts override design notes, archived notes, and baseline implementation behavior.
- There is no separate active strategy-note authority above the v2 contract set; direction must be read from the active contracts and companion authorities.

---

## 1) Mandatory Reading (Always Required)
Read these documents in this order before implementing, modifying, or reviewing anything substantial:

1. `docs/00_INDEX.md`
2. `docs/10_Glossary_and_Normative_Status.md`
3. `docs/13_Goals_and_Success_Criteria.md`
4. `docs/07_Data_Constitution.md`
5. `docs/01_Architecture_Constitution.md`
6. `docs/02_State_IR_Spec.md`
7. `docs/03_Level_Contracts_L0-L6.md`
8. `docs/04_Credit_Assignment_and_Recovery.md`
9. `docs/18_Optimization_and_Learning_Contract.md`
10. `docs/19_Runtime_and_Task_Adjudication_Semantics.md`

These define the active target.

---

## 2) Policy / Workflow Reading (Required When Relevant)
If the task touches metrics, regression, evaluation policy, phase gates, training governance, data realization, reproducibility, scaling profiles, document parsing, benchmark governance, verifier behavior, or readiness claims, you must also read the relevant companion docs and policies:

- `docs/05_Eval_Metrics_Spec.md`
- `docs/06_Regression_and_Phase_Gates.md`
- `docs/08_Training_Run_Governance.md`
- `docs/09_Training_Profiles_and_Scaling.md`
- `docs/14_Multimodal_Document_Pipeline.md`
- `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`
- `docs/16_Verifier_and_Formalization_Stack.md`
- `docs/17_Scaling_Promotion_and_Readiness.md`
- `docs/18_Optimization_and_Learning_Contract.md`
- `docs/19_Runtime_and_Task_Adjudication_Semantics.md`

Notes:

- `docs/11_Phase_D_Diagnostics_Design_Note.md` and `docs/12_Phase_E_Execution_Design_Note.md` are archive notes, not active guidance. They may mention legacy `StateIR(T,G,O,R,X,M)` or ARC-specific semantics and must not be used as active authority.
- `docs/codex_plan/*` is operational guidance only and cannot override active contracts.

---

## 3) Authority States and Edit Boundaries

### 3.1 Active v2 Authority
Treat the following as active v2 documents:

- `docs/01_Architecture_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/05_Eval_Metrics_Spec.md`
- `docs/06_Regression_and_Phase_Gates.md`
- `docs/07_Data_Constitution.md`
- `docs/08_Training_Run_Governance.md`
- `docs/09_Training_Profiles_and_Scaling.md`
- `docs/13_Goals_and_Success_Criteria.md`
- `docs/14_Multimodal_Document_Pipeline.md`
- `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`
- `docs/16_Verifier_and_Formalization_Stack.md`
- `docs/17_Scaling_Promotion_and_Readiness.md`
- `docs/18_Optimization_and_Learning_Contract.md`
- `docs/19_Runtime_and_Task_Adjudication_Semantics.md`

### 3.2 Documentation-First Transition Rule
During the transition period:

- Active docs may lead the implementation.
- Baseline code that still reflects pre-v2 assumptions is transitional, not authoritative.
- If you encounter a mismatch between active docs and baseline code, record the mismatch and implement toward the active docs unless the user explicitly asks for baseline maintenance work.

### 3.3 Writable vs Routine-RO
Routine feature work should treat active authority docs cautiously.

You may directly edit:

- `AGENTS.md`
- `docs/00_INDEX.md`
- `docs/07_Data_Constitution.md`
- `docs/08_Training_Run_Governance.md`
- `docs/09_Training_Profiles_and_Scaling.md`
- `docs/10_Glossary_and_Normative_Status.md`
- `docs/13_Goals_and_Success_Criteria.md`
- `docs/14_Multimodal_Document_Pipeline.md`
- `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`
- `docs/16_Verifier_and_Formalization_Stack.md`
- `docs/17_Scaling_Promotion_and_Readiness.md`
- `docs/18_Optimization_and_Learning_Contract.md`
- `docs/19_Runtime_and_Task_Adjudication_Semantics.md`
- `docs/codex_plan/*`
- `docs/repo-tree.txt`
- `src/`
- `tests/`
- `scripts/`

You may edit `docs/01` through `docs/06` only when at least one of the following is true:

- The task explicitly targets v2 contract migration or contract clarification.
- The user explicitly asks for contract edits.
- The edit is required to remove a contradiction among active v2 documents.

When editing `docs/01` through `docs/06`, do not make silent partial tweaks. Update the surrounding contract text so the resulting document is internally coherent.

### 3.4 Vendored / External Tooling
Treat the following as read-only unless the user explicitly requests otherwise:

- `tools/arc-agi-benchmarking/`
- `tools/ConceptARC/`

Wrap or adapt them from `src/` instead of rewriting upstream semantics.

---

## 4) Hard Prohibitions
Reject changes that do any of the following:

1. Introduce a second high-capacity semantic trunk or parallel "brain."
2. Bypass State IR by feeding raw tool outputs, parser traces, or unmanaged side channels into the trunk.
3. Replace learned routing, gating, recovery, or termination with deterministic if/else policy, except clearly labeled temporary guardrails.
4. Turn Level 2 into a handwritten symbolic executor or DSL-centered solver core.
5. Remove or bypass Level interfaces `L0-L6`, even if some implementations remain stubs.
6. Reintroduce a blanket "benchmark = probe only" dogma that contradicts `docs/07_Data_Constitution.md` and `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.
7. Mix benchmark data into training without declared tiering, decontamination policy, and held-out audit.

If you refuse a change under this section, you must provide the nearest contract-compliant alternative or a concrete migration note.

---

## 5) Required Workflow

### 5.1 Declare the Change Class
At the start of the work, explicitly declare one:

- Pure refactor
- Targeted fix
- Capability expansion

Use canonical failure taxonomy names when discussing impact.

### 5.2 Maintain Phase-Appropriate Scope
Phase intent is governed by `docs/06_Regression_and_Phase_Gates.md`.
Do not smuggle later-phase behavior into earlier-phase work.

### 5.3 Regression Discipline
Any architecture-, training-, data-, parser-, verifier-, or eval-impacting change must:

- preserve or intentionally revise the declared regression story,
- avoid silent drift in failure distribution,
- record any benchmark-tier or contamination-policy changes,
- terminate explicitly as `Done`, `Blocked`, or `Cancelled`.

---

## 6) Temporary Technical Debt Rule
If you introduce a hard cap or deterministic fallback, you must:

- label it `TEMPORARY TECHNICAL DEBT`,
- isolate it,
- provide a removal criterion,
- name the intended learned replacement,
- ensure it does not become routine semantic policy.

---

## 7) Repository Expectations

- Core model behavior belongs in `src/`.
- Documents define the active target; code may lag during transition, but must move toward them.
- Tools remain tools; they must not become the intelligence substrate.
- Data policy, benchmark tiering, provenance rules, document-pipeline constraints, verifier evidence classes, and scaling readiness rules must come from the active docs, not ad hoc scripts.

---

## 8) Minimal Completion Checklist
Every substantial change should include:

- mandatory docs consulted,
- relevant policy docs consulted,
- declared change class,
- expected failure-category impact,
- any technical-debt guardrails,
- benchmark contamination / provenance implications, if relevant,
- regression status,
- termination status (`Done`, `Blocked`, or `Cancelled`).

End of AGENTS.md
