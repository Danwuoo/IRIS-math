# Documentation / Decisions Log (IRIS-math Transition / Codex)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Durable record of decisions, assumptions (`不確定`), and
completion checklists for the IRIS-math transition  
**Non-Override Clause:** This file does not approve conflicting changes by
itself.

---

## 0) Current Status

```text
current_mode: IRIS-math transition control
current_milestone: T0 completed
next_action: move docs/14..16 from seed proposals toward approved transition surfaces
```

---

## 1) Assumptions / Unknowns (label 不確定)

```text
- 不確定: no approved transition spec yet supersedes the historical baseline architecture documents.
- 不確定: benchmark training inclusion remains blocked until docs/15 gains explicit approval.
- 不確定: the first approved document-math parse sidecar schema may require multiple iterations before model-facing use is allowed.
- Assumed hardware routing for planning: 1x H100 80GB currently available, with borrowable 1-8x H200 NVL, 16x H200 SXM, and 1-8x B200 profiles.
```

---

## 2) Decision Records

## [2026-03-09] Decision: Move the repo control plane to IRIS-math transition mode

Decision:

- Rewrote the active control docs around IRIS-math as the primary direction.
- Reclassified the baseline IRIS doc stack as historical baseline references for
  migration and compatibility analysis.
- Seeded transition docs `docs/13..16`.

Rationale:

- The repo previously instructed Codex to optimize for a baseline skeleton,
  single-H100 profile, and regression-only benchmark posture.
- That control surface no longer matched the active project direction.

Guardrail:

- AGENTS, design notes, and charters do not by themselves approve conflicting
  architecture, data, parser, or evaluation changes.
- Conflicting work still requires approved transition specs.

Follow-ups:

- Review and approve the blocked surfaces in `docs/14..16`.
- Replace the frozen historical baseline runbook with an IRIS-math runbook when
  the first approved transition surfaces exist.

---

## 3) Per-Change Completion Checklist

```text
Mandatory docs consulted:
- AGENTS.md
- docs/00_INDEX.md
- docs/10_Glossary_and_Normative_Status.md
- docs/13_IRIS_Math_v2_Charter.md
- relevant docs/14..16
- relevant workflow docs (docs/05, docs/06, docs/08)
- impacted historical baseline docs, if needed

Change class:
Active workstream:
Active spec or proposal status:
Impacted baseline docs:
Expected failure-category or eval-risk impact:
Contamination/eval risk note:
Hardware target profile:
Technical debt guardrails introduced (if any):
Validation run:
Termination: Done | Blocked | Cancelled
```
