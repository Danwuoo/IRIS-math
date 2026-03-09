# AGENTS.md (for gpt-5.4)
Project: IRIS-math

## 0) Non-Negotiable Operating Mode
You are an implementation agent for the IRIS-math repo.

This repository is transitioning from the historical baseline IRIS contract stack
to a math-native, document-native, verifier-centered reasoning system. Optimize
for the active IRIS-math direction while preserving explicit migration
discipline and avoiding undocumented drift.

Before any development work (code changes, refactors, new modules, eval harness,
data-policy updates, parser changes, or control-doc rewrites), you MUST complete
the mandatory reading steps in Section 1 and follow the permissions in Section
3.

If there is any conflict between documents:
- Approved transition specs override historical baseline references for
  IRIS-math work.
- Canonical binding workflow docs apply unless an approved transition spec
  explicitly supersedes them.
- AGENTS.md, design notes, and project-direction statements do not by
  themselves legalize conflicting architecture, data, or evaluation changes.
- Change proposals are not active until approved.

When uncertain, explicitly label **不確定**.

If uncertainty can be resolved by consulting the documents listed in Section 1
or 2, you MUST resolve it before defaulting to not making the change.

If a task depends on a conflicting change and only a proposal exists, do not
implement the conflicting portion. Update or produce the proposal, name the
blocked surface, and stop there.

Only unresolved uncertainty after mandatory consultation should block
implementation.

---

## 1) Mandatory Reading (Always Required)
Before you implement or modify anything, read these documents in this order:

### 1.1 Always Read
1. `AGENTS.md`
2. `docs/00_INDEX.md`
3. `docs/10_Glossary_and_Normative_Status.md`

### 1.2 Active IRIS-math Direction
4. `docs/13_IRIS_Math_v2_Charter.md`
5. `docs/14_IRIS_Math_Data_Constitution_v2.md`
6. `docs/15_Benchmark_Training_and_Eval_Tiering.md`
7. `docs/16_Document_Math_Parse_Canonical_Format.md`

### 1.3 Current Workflow and Governance References
Read the relevant workflow docs for the task:

- `docs/05_Eval_Metrics_Spec.md` for canonical metrics vocabulary
- `docs/06_Regression_and_Phase_Gates.md` for the currently active regression
  workflow unless superseded by an approved transition spec
- `docs/08_Training_Run_Governance.md` for resume, reproducibility, runtime
  lock, and S8 topics

### 1.4 Historical Baseline References
Read the historical baseline docs only when you need compatibility checks,
migration impact analysis, or blocked-surface traceability:

- `docs/01_Architecture_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/07_Data_Mixture_and_Ingestion.md`
- `docs/09_Training_Profile_SingleH100_3B.md`
- `docs/11_Phase_D_Diagnostics_Design_Note.md`
- `docs/12_Phase_E_Execution_Design_Note.md`

---

## 2) Active Direction vs Historical Baseline
This repository is no longer pursuing the baseline IRIS objective alone. The
active direction is IRIS-math: a math-native, document-native,
verifier-centered reasoning system.

Treat the documents as follows:

- `docs/13..16` define the active direction and transition-control surfaces.
- `docs/01..04`, `docs/07`, `docs/09`, `docs/11`, and `docs/12` are
  historical baseline references, not the default build target for new
  IRIS-math work.
- `docs/05`, `docs/06`, and `docs/08` remain usable workflow references unless
  an approved transition spec explicitly supersedes part of them.

When baseline references conflict with the active IRIS-math direction:
1. Consult `docs/10_Glossary_and_Normative_Status.md` for the authority map.
2. Check whether an approved transition spec exists for the conflicting area.
3. If an approved transition spec exists, follow it and cite the impacted
   baseline docs.
4. If only a proposal exists, update or create the proposal and do not
   implement the conflicting change.

Never silently force the historical baseline merely because it is older or more
concrete.

---

## 3) Permissions and Authority Boundaries (Read/Write Rules)
This section defines what you may modify. Treat it as repository policy.

### 3.1 Historical Baseline References (RO by Default)
You MUST NOT edit these files as part of routine IRIS-math development:

- `docs/01_Architecture_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/07_Data_Mixture_and_Ingestion.md`
- `docs/09_Training_Profile_SingleH100_3B.md`
- `docs/11_Phase_D_Diagnostics_Design_Note.md`
- `docs/12_Phase_E_Execution_Design_Note.md`

If you believe a historical baseline document is wrong, incomplete, or blocks
the IRIS-math direction, do one of the following:

- Update the relevant active transition doc if the change is already approved.
- Write or update a `Change Proposal (Requires Approval)` that explains the
  conflict, implications, and migration path.
- Do not silently reactivate the baseline as the controlling authority.

### 3.2 Tooling Vendored Code (RO by Default)
The following are treated as externally sourced or "vendored":

- `tools/arc-agi-benchmarking/`
- `tools/ConceptARC/`

Do not rewrite tool internals as a shortcut for IRIS-math core work.

Allowed:
- Add wrappers or adapters in `src/` that consume these tools without altering
  their upstream semantics.

### 3.3 Writable (W): Active Direction, Workflow, and New Notes
You MAY edit or add under:

- `docs/00_INDEX.md`
- `docs/05_*.md`
- `docs/06_*.md`
- `docs/08_*.md`
- `docs/10_*.md`
- `docs/13_*.md` through `docs/16_*.md`
- `docs/codex_plan/`
- `docs/repo-tree.txt`

You MAY also add new documents under `docs/` that are explicitly labeled as:

- `Design Note (Non-normative)`, or
- `Change Proposal (Requires Approval)`

and that clearly state whether they are active direction, workflow guidance, or
proposal material.

### 3.4 Writable (W): Core Implementation
You MAY implement and refactor core system code under:

- `src/`

You MUST keep:

- Core model behavior in `src/`, not in `tools/` or `data/`
- Trunk responsibility attributable to learned model behavior
- Any document/parser/benchmark integration aligned with approved transition
  docs and canonical interfaces

---

## 4) Hard Prohibitions (Reject Changes That Do This)
You must refuse to implement changes that:

1. Introduce silent State IR schema drift, canonical ordering drift, or hidden
   document-parse schema drift without a versioned approved transition spec or
   an explicit proposal plus migration note.
2. Bypass canonical interfaces by sending raw tensors, raw parser outputs,
   benchmark labels, tool traces, or ad-hoc latent channels directly into the
   trunk.
3. Include benchmark data in training without documented tiering,
   decontamination, held-out evaluation, and artifact plans.
4. Replace learned routing, gating, search, or termination with deterministic
   if/else policy, except clearly labeled temporary guardrails with removal
   criteria.
5. Turn verifier, search, parser, or math execution tooling into the primary
   intelligence substrate.
6. Add a secondary high-capacity network that competes with the trunk.
7. Remove, collapse, or bypass Level interfaces without a documented migration
   contract and approved replacement surface.
8. Claim that a conflicting architecture, data, or evaluation change is allowed
   solely because AGENTS.md or a design note says the repo targets IRIS-math.

When refusing a change under this section, you MUST:

- Propose the closest compliant alternative, or
- Produce or update a minimal Change Proposal stub that identifies the blocking
  clause, the blocked surface, and a compliant migration path.

Pure refusal without a next-step artifact is not allowed.

---

## 5) Required Workflow for Any Change
### 5.1 Declare the Change Class and Workstream
At the start of your work, explicitly declare:

- Change class:
  - `Pure refactor`
  - `Targeted fix`
  - `Capability expansion`
  - `Contract migration proposal`
- Active workstream:
  - `Control-plane realignment`
  - `Document parsing expansion`
  - `Data-policy redesign`
  - `Verifier/search upgrade`
  - `Benchmark tiering/eval redesign`
  - `Hardware profile routing`

Use canonical metrics vocabulary when naming expected impact. Do not invent new
failure labels ad hoc.

### 5.2 Active-Spec Discipline
Every substantial change must record:

- Active spec(s) consulted
- Proposal vs approved status for the relevant surface
- Impacted historical baseline docs
- Contamination or evaluation risk, if benchmarks or datasets are involved
- Hardware target profile
- Explicit termination state: `Done`, `Blocked`, or `Cancelled`

If the task touches planning, evaluation policy, metrics, regression process,
phase gates, benchmark inclusion, parser contracts, or data mixture, read the
relevant active transition docs first and then cite the workflow docs they
interact with.

### 5.3 Regression Discipline (Always-On)
Any architecture, training, evaluation, parser, or data-impacting change must:

- Preserve regression expectations and required artifacts, unless an approved
  transition spec explicitly changes them
- Avoid silent shifts in failure distributions or evaluation semantics unless
  the shift is intended, documented, and attributable

No open-ended or implicitly ongoing work is allowed.

---

## 6) "Technical Debt" Rule for Hard Control (Only as Guardrail)
If you must introduce a hard cap (for example max steps, parser limits, or
budget ceilings), you MUST:

- Label it clearly as `TEMPORARY TECHNICAL DEBT`
- Isolate it so it can be removed
- Provide a removal criterion and the intended learned replacement
- Ensure it does not become routine semantic policy

---

## 7) Hardware Routing Expectations
Known hardware targets for this repo:

- Current training resource: `1x H100 80GB`
- Borrowable profiles: `1-8x H200 NVL`, `16x H200 SXM`, `1-8x B200`

Rules:

- Do not assume `1x H100` is the only valid planning target.
- Pick the smallest hardware profile that answers the current task.
- For scaling, training, or runtime decisions, state which profile you are
  targeting and why.
- If a plan is intended to scale from `3B -> 7B -> 14B -> 30B -> 70B -> 120B`,
  say where the current change sits on that path.

---

## 8) Repository Structure Expectations (Do Not Violate)
- Core model behavior belongs in `src/`.
- Tools remain in `tools/` and should not become the intelligence substrate.
- Datasets remain under `data/` and are not a place to encode semantics.
- Transition control docs belong in `docs/`, not in code comments or ad-hoc
  task notes.

---

## 9) Minimal Completion Checklist (Attach to Each PR/Change)
You must include:

- Which mandatory docs you consulted
- The change class and active workstream
- The active spec or proposal status for the touched surface
- Impacted historical baseline docs
- The expected failure-category or evaluation-risk impact
- Contamination or eval-risk notes, if applicable
- Hardware target profile
- Any introduced technical-debt guardrails with removal criteria
- Regression status or doc-validation status
- Termination state: `Done`, `Blocked`, or `Cancelled`

End of AGENTS.md
