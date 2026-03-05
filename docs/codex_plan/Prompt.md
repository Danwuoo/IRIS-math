# Codex Prompt Pack (IRIS)

**Document Type:** Design Note (Non-normative)  
**Purpose:** A stable, copy/pasteable prompt + workflow for using Codex to build IRIS without architectural drift  
**Non-Override Clause:** This document does not override normative contracts in `docs/01..04` or workflow bindings in `docs/05..08`.

---

## 0) Intended Workflow (4-file loop)

Keep these four files as the canonical "long-horizon task bundle":

- `docs/codex_plan/Prompt.md` (this file): the spec + kickoff prompt you paste into Codex
- `docs/codex_plan/Plan.md`: the living execution plan (milestones, acceptance, next actions)
- `docs/codex_plan/Implement.md`: runbook (commands, environment notes, how to validate)
- `docs/codex_plan/Documentation.md`: decisions, assumptions (不確定), and completion checklists

This follows the OpenAI Cookbook "long horizon tasks" pattern:

- `https://developers.openai.com/cookbook/articles/codex_exec_plans`

---

## 1) Kickoff Prompt (copy/paste into Codex)

```text
You are an implementation agent for the IRIS repo. Follow AGENTS.md exactly.

Mandatory reading BEFORE changing code (in order):
- docs/10_Glossary_and_Normative_Status.md
- docs/01_Architecture_Constitution.md
- docs/02_State_IR_Spec.md
- docs/03_Level_Contracts_L0-L6.md
- docs/04_Credit_Assignment_and_Recovery.md

If the task touches metrics/regression/training ops/data mixture, also read:
- docs/05_Eval_Metrics_Spec.md
- docs/06_Regression_and_Phase_Gates.md
- docs/07_Data_Mixture_and_Ingestion.md
- docs/08_Training_Run_Governance.md
- docs/09_Training_Profile_SingleH100_3B.md

Change class: capability expansion (model-dev system scaffolding; no contract changes).

Goal:
Build an IRIS "model development system + baseline model" skeleton that is contract-compliant and can run end-to-end on a toy workload, with clear hooks for scaling to 1x H100.

Non-negotiable constraints (treat as hard requirements):
- One and only one trunk (no second trunk / dual-brain).
- Canonical State IR only: token types {T,G,O,R,X,M} in fixed order; no new categories.
- All Level interfaces L0–L6 must exist; disabled levels must be stubs preserving I/O + observability.
- Routing/gating/control must be learnable-by-default; hard control only as TEMPORARY TECHNICAL DEBT guardrails with removal criteria.
- Do not turn L2 into a symbolic executor/DSL core.
- Benchmarks are regression probes only (no ARC-family data in training mixture).

Deliverables (Phase A→C baseline skeleton):
- src/iris/schema/: State IR data structures + validator aligned to docs/02.
- src/iris/levels/: L0..L6 interfaces + stubs + minimum diagnostics fields.
- src/iris/trunk/: a minimal trunk (Flax NNX preferred) that consumes State IR and emits updates + control logits.
- src/iris/metrics/: logging helpers using canonical metric names from docs/05.
- src/iris/train/: minimal training loop with segment journal + exactly-once resume semantics (docs/08) on a tiny synthetic dataset.
- scripts/: entrypoints for smoke train/eval and S1/S2-style structural checks.

Validation:
- Provide commands to run a fast smoke test on CPU and (if available) GPU.
- Add minimal unit tests for State IR ordering/type enforcement and Level stub behavior.

Working style:
- Use docs/codex_plan/Plan.md as the single source of truth for milestones + acceptance criteria.
- Record decisions/assumptions (label 不確定 when needed) in docs/codex_plan/Documentation.md.
- Keep patches small; run the narrowest tests first.
- Do not edit normative RO docs; if you must, write a Change Proposal document instead.
```

---

## 2) Hardware Profiles (how to phrase constraints to Codex)

Codex works best if you **explicitly choose one** profile per run:

- **Local RTX 3050 (dev/smoke):** run small configs, prioritize `S1/S2` + unit tests; if using JAX GPU, prefer Linux/WSL2.
- **Notebook H100 (scale target):** align to `docs/09_Training_Profile_SingleH100_3B.md`; enforce `docs/08` segment journal + resume exactly-once.

If you want both, tell Codex:
1) implement a tiny CPU/GPU smoke config first, then  
2) add an H100 profile only after the smoke path is stable.

---

## 3) Dataset Policy Reminder (so prompts don’t accidentally violate docs)

When you instruct Codex to "use HuggingFace datasets":

- Use HF for **Pure LM** text and **IR-aligned synthetic** data pipelines.
- Do **not** include ARC-family / ConceptARC / arc-agi-benchmarking datasets in training mixture (regression probes only).
- Make dataset choice/config **data-driven** (config file), not hard-coded.

Source of truth: `docs/07_Data_Mixture_and_Ingestion.md`.

---

## 4) Prompting Tips (Codex-specific)

- Keep the goal statement short; put details in files (Plan/Implement/Documentation).
- Always include: exact commands to run + what "done" means.
- Ask for incremental milestones (e.g., "make S2 pass") rather than "build everything".

References:

- `https://developers.openai.com/cookbook/guides/gpt-5-codex_prompting_guide`
