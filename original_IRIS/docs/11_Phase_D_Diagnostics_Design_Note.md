# Phase D ARC Diagnostics Design Note

**Document Type:** Design Note (Non-normative)  
**Normative Boundaries:** This note does not override `docs/01`~`docs/06` or `docs/08`.

## 1. Scope

Phase D implementation upgrades S4/S5/S3 diagnostics from dataset IO proxies to model-inference-driven evaluation:

- Use canonical `StateIR(T,G,O,R,X,M)` encoding for ARC tasks.
- Run mounted L0-L6 + single trunk inference path.
- Decode predicted grid and confidence from model outputs.
- Attribute failures using canonical taxonomy `F_REP/F_PROC/F_SEARCH/F_MEM/F_ABS/F_EVAL`.

Out of scope:

- No edits to `tools/ConceptARC/` or `tools/arc-agi-benchmarking/`.
- No State IR token-category or ordering changes.
- No secondary trunk or hard-coded solver core.

## 2. Leakage Protocol

S4 uses two modes per ConceptARC task:

- `concept.isolation`: original task train context.
- `concept.leakage`: train context replaced by cross-concept examples.

Leakage score is computed as:

`concept.leakage_score = max(0, concept.isolation_score - mixed_success_rate)`

Per-concept diagnostics include:

- `concept.success_rate`
- `concept.isolation_score`
- `concept.leakage_score`
- `concept.failure_profile` (taxonomy histogram)

## 3. re_arc Pairing Policy

Current policy is `adjacent pairing`:

- `[0,1], [2,3], ...` within each task example sequence.
- Policy value is recorded in artifacts as `pairing_policy`.

Rationale:

- Deterministic and reproducible without changing vendored dataset logic.
- Replaceable if future binding docs define official pair mapping.

## 4. S3 Taxonomy Drift Semantics

S3 no longer uses S8 failure-credit KL as proxy.

Phase D S3 uses taxonomy histogram drift:

- `failure_profile.current_histogram`
- `failure_profile.baseline_histogram`
- `failure_profile.delta`
- `failure_profile.l1_distance`
- `failure_profile.kl_divergence`

## 5. Technical Debt Guardrail

`max_reasoning_cycles` is explicitly managed as:

- **TEMPORARY TECHNICAL DEBT**
- Purpose: avoid runaway reasoning loops before termination calibration stabilizes.
- Removal criterion: remove hard cap after 3 consecutive full-runs with stable termination calibration.

## 6. Artifact/Baseline Workflow

Phase D baseline semantics:

1. Freeze `phase-d-v1` baseline artifacts with current S3/S4/S5 v2 schema.
2. Subsequent runs diff against frozen baseline.
3. Keep `phase`, `baseline_id`, `tolerance_profile_id` explicit in summary report.
