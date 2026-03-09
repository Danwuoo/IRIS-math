# Codex Prompt Pack (IRIS-math Transition)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Stable kickoff prompt and workflow for using Codex on the active
IRIS-math transition  
**Non-Override Clause:** This document does not override approved transition
specs, canonical workflow bindings, or proposal approval rules.

---

## 0) Intended Workflow

Use this active planning bundle:

- `docs/codex_plan/Prompt.md` (this file): kickoff prompt and workflow guardrails
- `docs/codex_plan/Plan.md`: living transition milestones and acceptance
- `docs/codex_plan/Documentation.md`: decisions, assumptions (`不確定`), and
  completion checklists

Historical baseline runbook:

- `docs/codex_plan/Implement.md`

Do not treat `Implement.md` as the active source of truth for IRIS-math work
unless the task explicitly targets historical baseline compatibility.

---

## 1) Kickoff Prompt (copy/paste into Codex)

```text
You are an implementation agent for the IRIS-math repo. Follow AGENTS.md exactly.

Mandatory reading BEFORE changing code or docs (in order):
- AGENTS.md
- docs/00_INDEX.md
- docs/10_Glossary_and_Normative_Status.md
- docs/13_IRIS_Math_v2_Charter.md
- docs/14_IRIS_Math_Data_Constitution_v2.md
- docs/15_Benchmark_Training_and_Eval_Tiering.md
- docs/16_Document_Math_Parse_Canonical_Format.md

Then read the relevant workflow docs:
- docs/05_Eval_Metrics_Spec.md
- docs/06_Regression_and_Phase_Gates.md
- docs/08_Training_Run_Governance.md

Read docs/01..04, docs/07, docs/09, docs/11, and docs/12 only when you need
historical baseline compatibility analysis or migration traceability.

Before implementation, declare:
- change_class = pure_refactor | targeted_fix | capability_expansion | contract_migration_proposal
- active_workstream = control_plane_realignment | document_parsing_expansion | data_policy_redesign | verifier_search_upgrade | benchmark_tiering_eval_redesign | hardware_profile_routing

Core rule:
- AGENTS.md, this prompt pack, and design notes do not by themselves approve
  conflicting architecture, data, parser, or evaluation changes.
- If the task depends on a conflicting surface and only a proposal exists,
  update or create the proposal and stop short of the conflicting implementation.

Goal:
Advance IRIS-math as a math-native, document-native, verifier-centered reasoning
system with explicit migration discipline.

Non-negotiable constraints:
- No undocumented drift in architecture, data policy, benchmark policy, or parse format.
- No raw parser outputs, benchmark labels, or tool traces bypassing canonical interfaces into the trunk.
- No benchmark training inclusion without tiering, decontamination, and held-out evaluation planning.
- No hard-coded semantic routing/search/termination as routine policy.
- No tools, datasets, or parsers becoming the primary intelligence substrate.

Working style:
- Use docs/codex_plan/Plan.md as the transition milestone source of truth.
- Record decisions, assumptions, and blocked surfaces in docs/codex_plan/Documentation.md.
- Treat docs/codex_plan/Implement.md as a frozen historical baseline runbook unless the task explicitly targets that baseline.
- Keep patches small and validate links, authority labels, and active-vs-historical wording after edits.
```

---

## 2) Hardware Profiles

Codex should choose and state an explicit hardware target for each substantial
task.

Known profiles:

- Local dev/smoke profile
- `1x H100 80GB` current training resource
- `1-8x H200 NVL` borrowable scale profile
- `16x H200 SXM` borrowable scale profile
- `1-8x B200` borrowable scale profile

Rules:

- Do not assume `1x H100` is the only long-range target.
- Pick the smallest profile that answers the current task.
- If the task is on the `3B -> 7B -> 14B -> 30B -> 70B -> 120B` path, say which
  segment it belongs to.

---

## 3) Data and Benchmark Reminder

- Benchmark data is not automatically approved for training just because the
  project is benchmark-aware.
- If a task needs benchmark training inclusion, decontamination changes, or
  held-out policy changes, work through
  `docs/15_Benchmark_Training_and_Eval_Tiering.md` first.
- If a task needs new parser-derived model-facing interfaces, work through
  `docs/16_Document_Math_Parse_Canonical_Format.md` first.
- If a task needs new mixture or ingestion policy, work through
  `docs/14_IRIS_Math_Data_Constitution_v2.md` first.

---

## 4) Prompting Tips

- Keep the goal statement short and put durable policy in docs.
- Always include: exact task, active workstream, target hardware profile, and
  what counts as `Done`.
- Prefer milestones and proposal closures over open-ended "explore more" tasks.
