# Documentation / Decisions Log (IRIS-math v2)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Durable record of migration decisions, assumptions, and completion checklists

---

## 0. Current Status

```text
current_mode: documentation-first transition
current_focus: v2 authority + contract rewrite completed
next_action: align src State IR / metrics / regression implementation to active v2 docs
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

---

## 3. Per-Change Completion Checklist

```text
Mandatory docs consulted:
- docs/數學模型建議.md
- docs/00_INDEX.md
- docs/10_Glossary_and_Normative_Status.md
- docs/07_Data_Constitution.md
- docs/01_Architecture_Constitution.md
- docs/02_State_IR_Spec.md
- docs/03_Level_Contracts_L0-L6.md
- docs/04_Credit_Assignment_and_Recovery.md

Policy docs consulted:
- docs/05_Eval_Metrics_Spec.md
- docs/06_Regression_and_Phase_Gates.md
- docs/08_Training_Run_Governance.md
- docs/09_Training_Profiles_and_Scaling.md

Change class: capability_expansion
Expected failure-category impact: establishes v2 semantics while preserving stable external F_* / L0-L6 / S1-S8 surfaces
Technical debt guardrails introduced: none in this documentation wave
Benchmark contamination / provenance implications: tiering, decontamination, parser/formalizer/verifier provenance are now mandatory policy surfaces
Regression status: targeted phase-gate regression checks required
Termination: Done
```
