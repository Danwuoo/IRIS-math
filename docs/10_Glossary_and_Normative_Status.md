# Glossary and Normative Status

**Document Type:** Design Note (Non-normative)  
**Purpose:** Shared authority labels, transition terminology, and vocabulary to
prevent control-plane drift

---

## 1. Normative Status Labels

Use these labels consistently:

- **Approved Transition Spec (Normative):** approved IRIS-math replacement or
  superseding spec for a specific surface.
- **Canonical Binding Workflow/Vocabulary:** binding process or metric
  vocabulary until superseded by an approved transition spec.
- **Active Direction (Control, Non-normative):** active project direction and
  prioritization; does not by itself authorize conflicting implementation.
- **Transition Proposal (Requires Approval):** candidate changes to contracts,
  data policy, benchmark tiering, parser format, or workflow; not active until
  approved.
- **Historical Baseline Reference:** retained baseline IRIS contract or profile
  document used for compatibility checks and migration analysis, not as the
  default build target for new IRIS-math work.
- **Design Note (Non-normative):** implementation guidance or navigation help;
  cannot override approved specs or canonical workflow bindings.

Conflict rules:

1. Approved transition specs override historical baseline references for their
   approved surface.
2. Canonical workflow docs remain active unless an approved transition spec
   explicitly supersedes them.
3. Transition proposals guide planning and proposal work but do not approve a
   conflicting change by themselves.
4. AGENTS, design notes, and charters cannot alone legalize conflicting
   architecture, data, parser, or evaluation changes.

---

## 1.1 This Repo's Authority Map (Current)

**Active direction docs**

- `docs/13_IRIS_Math_v2_Charter.md`

**Transition proposals**

- `docs/14_IRIS_Math_Data_Constitution_v2.md`
- `docs/15_Benchmark_Training_and_Eval_Tiering.md`
- `docs/16_Document_Math_Parse_Canonical_Format.md`

**Canonical binding workflow/vocabulary**

- `docs/05_Eval_Metrics_Spec.md`
- `docs/06_Regression_and_Phase_Gates.md`
- `docs/08_Training_Run_Governance.md`

**Historical baseline references**

- `docs/01_Architecture_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/07_Data_Mixture_and_Ingestion.md`
- `docs/09_Training_Profile_SingleH100_3B.md`
- `docs/11_Phase_D_Diagnostics_Design_Note.md`
- `docs/12_Phase_E_Execution_Design_Note.md`

**Navigation and planning**

- `docs/00_INDEX.md`
- `docs/codex_plan/Prompt.md`
- `docs/codex_plan/Plan.md`
- `docs/codex_plan/Documentation.md`
- `docs/codex_plan/Implement.md` (historical baseline runbook)

Current approval note:

- No approved transition spec currently supersedes the historical baseline
  architecture documents.
- Therefore, any conflicting architecture, data, parser, or evaluation change
  still requires proposal work and explicit approval before implementation.

---

## 2. Control-Plane Terms

- **Active direction:** the current project target and prioritization. In this
  repo, that is IRIS-math.
- **Historical baseline:** the previous IRIS control stack preserved for
  reference and migration analysis.
- **Approved surface:** the portion of a proposal or spec that is explicitly
  allowed to guide implementation right now.
- **Blocked surface:** the portion that remains proposal-only or requires an
  approval step before implementation.
- **Migration artifact:** the document, decision record, decontamination note,
  or compatibility report required to move a blocked surface toward approval.
- **Contamination risk:** the risk that benchmark, held-out, or evaluation data
  leaks into training or model-shaping policy.
- **Hardware target profile:** the declared compute envelope for the current
  task, for example `1x H100 80GB`, `1-8x H200 NVL`, `16x H200 SXM`, or
  `1-8x B200`.

---

## 3. Architecture and Workflow Terms

- **Trunk:** primary parameter-bearing cognitive substrate.
- **Second trunk:** competing high-capacity parallel semantic network
  (forbidden).
- **State IR:** canonical internal representation referenced by the historical
  baseline; any change requires explicit migration discipline.
- **Level (L0-L6):** semantic responsibility interfaces from the historical
  baseline architecture.
- **Guardrail:** non-semantic safety or resource bound.
- **Technical debt guardrail:** temporary hard control with a removal criterion.
- **Failure taxonomy:** canonical failure categories
  (`F_REP`, `F_PROC`, `F_SEARCH`, `F_MEM`, `F_ABS`, `F_EVAL`).
- **Credit vector:** L6-emitted failure responsibility distribution in the
  baseline workflow.

---

## 4. Boundary Rules

1. Direction is written once and referenced, not redefined by conflicting notes.
2. Proposal docs must mark approved and blocked surfaces explicitly.
3. Historical baseline references remain readable for compatibility analysis
   even when they are no longer the default target.
4. Workflow docs define process and vocabulary, not project identity.
5. A missing approved transition spec is a blocker for conflicting
   implementation, not a license to fall back silently or to improvise.
