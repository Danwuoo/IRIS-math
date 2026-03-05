# Execution Plan (IRIS / Codex)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Living, check-boxable execution plan for Codex runs (milestones + acceptance)  
**Non-Override Clause:** Does not override any contracts in `docs/01..04` or bindings in `docs/05..08`.

---

## 0) Run Header (fill per Codex run)

```text
run_id: codex-2026-02-28-phase-c-skeleton-01
date: 2026-02-28
phase_target: C
change_class: capability_expansion
target_failure_categories: F_REP, F_PROC, F_SEARCH, F_EVAL (attribution scaffolding focus; no benchmark optimization claims)
baseline_id (if regression-relevant): toy-baseline
tolerance_profile_id (if regression-relevant): toy-default
```

---

## 1) Milestones (recommended order)

### M0 — Skeleton package + CLI

**Goal:** Create the minimal project structure so everything else has a stable home under `src/`.

**Deliverables**

- `src/iris/` package created
- `scripts/` entrypoints created (thin wrappers; logic stays in `src/iris/`)
- A single "smoke" command that imports the package and prints versions/config

**Acceptance**

- `python -c "import iris; print('ok')"` succeeds

---

### M1 — State IR schema enforcement (docs/02)

**Goal:** Enforce the closed token type set `{T,G,O,R,X,M}` and canonical ordering at runtime.

**Deliverables**

- `src/iris/schema/` implements:
  - data structure(s) for State IR
  - validation that rejects unknown token categories
  - canonical concatenation ordering helper
- Unit tests covering:
  - ordering enforcement
  - rejection of undefined token types
  - empty sections are allowed (but classes must exist)

**Acceptance**

- `python -m pytest -q` passes for schema tests

---

### M2 — Level interfaces L0–L6 + stubs (docs/03)

**Goal:** Ensure all Level interfaces exist and can run in stub mode with observability.

**Deliverables**

- `src/iris/levels/` defines:
  - interface for each Level (`L0`..`L6`)
  - stub implementations satisfying:
    - `state_out = state_in` (or minimal normalization)
    - neutral `control_out`
    - `diagnostics` includes "disabled" marker + basic summary stats
- Unit tests verifying stub behavior and that every Level is instantiable

**Acceptance**

- "structural check" script runs and prints one diagnostic record per Level

---

### M3 — Minimal single trunk (docs/01 + docs/02)

**Goal:** Provide exactly one trunk that consumes canonical State IR and produces updates + control surfaces.

**Deliverables**

- `src/iris/trunk/` with a minimal model (Flax NNX preferred) that:
  - takes State IR tokens with type embeddings
  - outputs:
    - updated tokens (same categories; no new token types)
    - learnable routing/gating logits (even if unused initially)
- A tiny forward-pass smoke test (CPU)

**Acceptance**

- forward pass runs on CPU with deterministic seed

---

### M4 — Training loop + segment journal + resume exactly-once (docs/08)

**Goal:** A minimal training pipeline that is resumable and produces policy-relevant artifacts.

**Deliverables**

- `src/iris/train/` implements:
  - toy dataset (synthetic, small) to validate wiring
  - segment transaction model (execute/apply) + append-only journal (`iris.segment_journal/v1`)
  - checkpoint save/load with the minimum required metadata fields
- `docs/codex_plan/Implement.md` updated with exact run commands

**Acceptance**

- Run `N` segments, interrupt mid-segment, resume:
  - no double-apply
  - segment replay occurs when last event is `PENDING`

---

### M5 — Metrics logging + S1/S2-style checks (docs/05 + docs/06)

**Goal:** Produce canonical metrics names and run fast regression-style structural checks.

**Deliverables**

- `src/iris/metrics/` helpers that emit:
  - `failure.credit` vector shape and constraints
  - Level diagnostic metrics stubs (even if zeroed)
- A script that implements:
  - **S1 Smoke**: no crash/NaN + basic output
  - **S2 Structural**: Levels exist (stubs ok), State IR schema stable, no extra token types

**Acceptance**

- `scripts/s1_smoke.ps1` or `scripts/s1_smoke.sh` runs in < 2 minutes on CPU
- `scripts/s2_structural.ps1` or `scripts/s2_structural.sh` passes

---

## 1.1) Latest Execution Status (2026-02-28)

```text
M0 status: Done
M1 status: Done
M2 status: Done
M3 status: Done
M4 status: Done
M5 status: Done
```

Acceptance evidence:

```text
python -m pytest -q
python scripts/s1_smoke.py --device cpu
python scripts/s2_structural.py
python scripts/train_toy.py --segments 2 --device cpu --output-dir artifacts/toy_train_cpu
python scripts/eval_toy.py --output-dir artifacts/toy_train_cpu
python scripts/train_toy.py --segments 1 --device gpu --output-dir artifacts/toy_train_gpu
```

Resume replay demonstration:

```text
python scripts/train_toy.py --segments 1 --output-dir artifacts/toy_resume --crash-point pre_commit --crash-segment 0
python scripts/train_toy.py --segments 1 --output-dir artifacts/toy_resume
```

Expected replay result:

```text
1) First run interrupts after a PENDING journal event.
2) Resume replays the same segment_id.
3) Exactly one APPLIED event exists for that segment_id.
```

---

## 2) Explicit Non-Goals (for early runs)

- No benchmark-specific solver logic (ARC is regression instrumentation only).
- No hard-coded routing/termination policy (guardrails only, labeled technical debt).
- No State IR schema changes.

---

## 3) Run Closure Template (must fill)

```text
status: Done
blocking_contract (if Blocked): N/A
what_changed: Added contract-compliant baseline skeleton under src/iris (schema, levels, trunk, metrics, train), scripts, and minimal unit tests.
expected_failure_metric_impact: Improves readiness to measure and diagnose F_REP/F_PROC/F_SEARCH/F_EVAL via canonical metrics and failure.credit instrumentation.
technical_debt_guardrails (if any, with removal criteria): None introduced as semantic policy.
regression_status (what ran / expected to pass): pytest pass; S1 pass; S2 pass; toy train/eval pass; crash+resume replay check pass.
```
