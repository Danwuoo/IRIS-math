# AGENTS.md (for gpt-5.3-codex)
Project: IRIS (Integrated Reasoning via Internal State)

## 0) Non-Negotiable Operating Mode
You are an implementation agent. You must not introduce architectural drift.
Before *any* development work (code changes, refactors, new modules, eval harness), you MUST complete the mandatory reading steps in Section 1 and follow the permissions in Section 3.

If there is any conflict between documents:
- **System-level invariants and normative contracts override everything else.**
- "Draft" or "design notes" never override normative/authoritative contracts.

When uncertain, explicitly label **不確定**.

If uncertainty can be resolved by consulting binding documents listed in Section 1 or 2,
you MUST resolve it before defaulting to not making the change.

Only unresolved uncertainty after mandatory consultation should block implementation.

---

## 1) Mandatory Reading (Always Required)
Before you implement or modify anything, read these documents in this order:

Recommended entrypoint: `docs/00_INDEX.md` (non-normative), then read the binding contracts below.

1. `docs/10_Glossary_and_Normative_Status.md` (Authority map + vocabulary)
2. `docs/01_Architecture_Constitution.md` (Authoritative architecture invariants + non-goals + trunk + learnable control)
3. `docs/02_State_IR_Spec.md` (Canonical State IR spec + examples; closed token types and ordering)
4. `docs/03_Level_Contracts_L0-L6.md` (Level interface contracts; stub/observability/prohibitions)
5. `docs/04_Credit_Assignment_and_Recovery.md` (Failure taxonomy + credit routing + recovery semantics)

These are binding. If your change would violate any item above, **do not implement**; propose an alternative that complies.

---

## 2) Planning/Eval Policy References (Required When Task Touches Planning/Eval Policy)
If your task mentions or implies work on planning, evaluation policy, metrics, regression process, or phase gates, you MUST read:

- `docs/05_Eval_Metrics_Spec.md` (canonical metrics vocabulary and field semantics)
- `docs/06_Regression_and_Phase_Gates.md` (canonical suites, phase activation, gates, artifacts, promotion criteria)
- `docs/08_Training_Run_Governance.md` (required if the task touches resume/repro/runtime lock/S8)

Notes:
- Legacy `docs/harness/legacy/*` and `docs/plan/*` were removed during docs consolidation; do not reintroduce them.
- Policy docs do not override Section 1 normative contracts.

---

## 3) Permissions and Authority Boundaries (Read/Write Rules)
This section defines what you may modify. Treat this as a repository policy.

### 3.1 Read-Only (RO): Binding Contracts and Invariants
You MUST NOT edit these files as part of routine development:
- `docs/01_Architecture_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/04_Credit_Assignment_and_Recovery.md`

If you believe a RO document is wrong or incomplete, you may:
- Write a proposal in a *new* document (see 3.3) explaining the conflict, implications, and migration plan.
- Do not silently change contracts.

### 3.2 Tooling Vendored Code (RO by Default)
The following are treated as externally sourced or "vendored":
- `tools/arc-agi-benchmarking/` (Do not modify tool internals as part of IRIS core work)
- `tools/ConceptARC/` (Do not rewrite the dataset/tool logic for convenience)

Allowed: add adapters/wrappers in `src/` that consume these tools without altering their upstream semantics.

### 3.3 Writable (W): Plans, Metrics, Regression, New Notes
You MAY edit/add under:
- `docs/00_INDEX.md`, `docs/05_*.md` … `docs/10_*.md`, and `docs/repo-tree.txt`, provided you do not contradict RO contracts
- New documents under `docs/` that are explicitly labeled as:
  - `Design Note (Non-normative)` OR
  - `Change Proposal (Requires Approval)`
  and that clearly state they do not override contracts.

### 3.4 Writable (W): Core Implementation
You MAY implement and refactor core system code under:
- `src/` (the only place where core model behavior should live in Phase C and beyond)

You MUST keep:
- State IR schema enforcement in `src/schema/` aligned with `docs/02_State_IR_Spec.md`.
- Trunk implementation consistent with `docs/01_Architecture_Constitution.md` (no hidden second trunk, and no architecture-specific bypass that violates the contract).

---

## 4) Hard Prohibitions (Reject Changes That Do This)
You must refuse to implement changes that:
1. Add new State IR token categories or change canonical ordering without a versioned spec revision (not allowed in normal development).
2. Bypass State IR by sending raw tensors, tool outputs, or program traces directly into the trunk.
3. Replace learned routing/gating/termination with deterministic if/else policy (except explicitly labeled guardrail technical debt with removal criteria).
4. Turn Level 2 into a "neural proposer + symbolic executor" split or a Python DSL interpreter as the core executor.
5. Add a secondary high-capacity network that competes with the trunk ("second trunk" in disguise).
6. Remove, collapse, or bypass any Level interface L0–L6 (including by deleting its I/O contract or stub behavior). Implementations may be disabled only if the interface contract remains intact.

When refusing any change under this section, you MUST:
- Propose the closest contract-compliant alternative, OR
- Produce a minimal Change Proposal stub that identifies the blocking clause and a compliant migration path.

Pure refusal without a next-step artifact is not allowed.

---

## 5) Required Workflow for Any Change
### 5.1 Declare the Change Class
At the start of your work, explicitly declare one:
- Pure refactor (no behavior change expected)
- Targeted fix (must name failure category / suspected Level)
- Capability expansion (must name concepts / expected impact)

Use the failure taxonomy / metrics vocabulary; do not invent new labels ad hoc.

### 5.2 Maintain "Phase-Appropriate" Scope
- Phase A: diagnostics, verifier signals, trace/logging skeleton only (no solver heuristics).
- Phase B: tool generation alignment (failure tags, paired tasks), do not encode correctness rules into tools.
- Phase C: minimal closed loop in `src/` with all Level interfaces present (mounted or stubbed); failures must be attributable.
- Phase D: ConceptARC as diagnostic harness; output isolation/leakage/attribution metrics (not leaderboard tuning).
- Phase E: arc-agi-benchmarking as regression & verifier harness; no benchmark hacks.

Phase definitions, suite activation states, and promotion requirements are governed by `docs/06_Regression_and_Phase_Gates.md`.

### 5.3 Regression Discipline (Always-On)
Any architectural/training/eval-impacting change must:
- Preserve the regression harness expectations and artifacts.
- Avoid silent shifts in failure distributions unless explicitly intended and documented.

All declared changes or plans MUST terminate explicitly in one of:
- Done
- Blocked (with blocking contract cited)
- Cancelled (with reason)

No open-ended or implicitly ongoing work is allowed.

---

## 6) "Technical Debt" Rule for Hard Control (Only as Guardrail)
If you must introduce a hard cap (e.g., max steps), you MUST:
- Label it clearly as TEMPORARY TECHNICAL DEBT.
- Isolate it so it can be removed.
- Provide a removal criterion and the intended learned replacement.
- Ensure it does not become routine policy.

---

## 7) Repository Structure Expectations (Do Not Violate)
- Core model behavior belongs in `src/`.
- Tools remain in `tools/` and should not become the intelligence substrate.
- Datasets are under `data/` and are not a place to encode semantics.

---

## 8) Minimal Completion Checklist (Attach to Each PR/Change)
You must include:
- Which mandatory docs you consulted (Section 1 + all relevant Section 2 policy docs).
- The change class (refactor / targeted fix / expansion).
- The expected failure-category impact (using canonical metrics).
- Any introduced technical debt guardrails (with removal criteria), if applicable.
- Regression status: what suites are expected to pass / what artifacts are updated.

End of AGENTS.md
