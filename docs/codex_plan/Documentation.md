# Documentation / Decisions Log (IRIS-math v2)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Durable record of migration decisions, assumptions, and completion checklists

---

## 0. Current Status

```text
current_mode: documentation-first transition
current_focus: theory-layer and document-layer engineering contracts completed through optimization bundles, adjudication policies, and regression/readiness binding
next_action: audit src and eval implementation gaps against the now-closed document contract set
```

---

## 1. Assumptions / Unknowns

```text
- 不確定: the current src implementation still reflects baseline T/G/O/R/X/M-oriented scaffolding in several places.
  Impact: docs are active truth; code-alignment work remains the next migration wave.

- 不確定: legacy ARC-family probes may remain temporarily useful as archive compatibility signals.
  Impact: allowed during transition, but insufficient alone for Phase D/E promotion evidence.
```

---

## 2. Decision Records

## [2026-03-10] Decision: move the repo to IRIS-math v2 authority

Decision:
- Rewrote authority, navigation, data policy, profile policy, and v2 contracts so the active target is IRIS-math v2.

Rationale:
- Baseline authority and baseline data policy were blocking the new direction.

## [2026-03-10] Decision: keep external `L0-L6`, `F_*`, and `S1-S8` stable in round one

Decision:
- The first v2 rewrite changes semantics but not external numbering or suite ids.

Rationale:
- This limits transition blast radius while documents lead the implementation.

## [2026-03-10] Decision: treat `docs/11` and `docs/12` as archive

Decision:
- They remain in the repo but are no longer active reading requirements.

Rationale:
- They describe baseline phase notes, not the active v2 target.

## [2026-03-13] Decision: retire the separate strategy-note authority and close the remaining theory gaps with `docs/18` and `docs/19`

Decision:
- Removed `docs/數學模型建議.md` from the active authority map.
- Added an optimization/learning contract and a runtime/task adjudication contract.
- Bound `L5` abstraction outputs canonically into State IR.

Rationale:
- Direction must now come from the active contract set itself, not from a missing strategy note.
- The repo needed explicit authority for learning-objective semantics, runtime stopping/adjudication semantics, and `L5` landing surfaces before engineering requirement review.

## [2026-03-13] Decision: close the remaining document-layer engineering requirements around executable objects and adjudication

Decision:
- Bound `learning_objective_bundle/v1` to explicit resolution order, immutability, lineage, and replay rules.
- Bound `task_adjudication_policy/v1` to explicit resolution order, canonical default policy ids, and auditable resolution-source fields.
- Connected benchmark-family registry, verifier evidence bundles, regression gates, and profile-readiness packets to the new adjudication/objective contracts.

Rationale:
- Without these bindings, the new contracts would still leave implementation-level ambiguity around which object wins, how policy ids are resolved, and what readiness/regression must actually check.

---

## 3. Per-Change Completion Checklist

```text
Mandatory docs consulted:
- docs/00_INDEX.md
- docs/10_Glossary_and_Normative_Status.md
- docs/13_Goals_and_Success_Criteria.md
- docs/07_Data_Constitution.md
- docs/01_Architecture_Constitution.md
- docs/02_State_IR_Spec.md
- docs/03_Level_Contracts_L0-L6.md
- docs/04_Credit_Assignment_and_Recovery.md
- docs/18_Optimization_and_Learning_Contract.md
- docs/19_Runtime_and_Task_Adjudication_Semantics.md

Policy docs consulted:
- docs/05_Eval_Metrics_Spec.md
- docs/06_Regression_and_Phase_Gates.md
- docs/08_Training_Run_Governance.md
- docs/09_Training_Profiles_and_Scaling.md
- docs/14_Multimodal_Document_Pipeline.md
- docs/15_Benchmark_Registry_and_Tiering_Playbook.md
- docs/16_Verifier_and_Formalization_Stack.md
- docs/17_Scaling_Promotion_and_Readiness.md

Change class: capability_expansion
Expected failure-category impact: reduces theory-layer ambiguity around F_ABS, F_SEARCH, and F_EVAL while preserving stable external F_* / L0-L6 / S1-S8 surfaces
Technical debt guardrails introduced: none in this documentation wave
Benchmark contamination / provenance implications: tiering, decontamination, parser/formalizer/verifier provenance remain mandatory policy surfaces
Regression status: targeted phase-gate regression checks required
Termination: Done
```
