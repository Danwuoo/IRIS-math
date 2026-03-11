# Eval Metrics Spec

**Document Type:** Canonical Binding Policy  
**Audience:** Model, training, evaluation, parser, formalizer, and verifier pipelines  
**Scope:** Canonical metric names, field semantics, and required metadata for IRIS-math v2  
**Companion authority:** `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`, `docs/16_Verifier_and_Formalization_Stack.md`, `docs/17_Scaling_Promotion_and_Readiness.md`

---

## 0. Purpose and Boundary

This document defines the canonical metrics vocabulary used to describe:

- mathematical outcome quality,
- failure taxonomy,
- level-specific diagnostics,
- contamination and provenance health,
- regression gate inputs.

This document defines **what is measured**, not which suites are active.
Suite activation lives in `docs/06_Regression_and_Phase_Gates.md`.
Benchmark-family-specific governance lives in `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.
Verifier evidence classes and false accept / false reject policy live in `docs/16_Verifier_and_Formalization_Stack.md`.
Profile-readiness promotion claims live in `docs/17_Scaling_Promotion_and_Readiness.md`.

---

## 1. Metric Classes

All canonical metrics fall into one of these classes:

1. outcome metrics,
2. failure-taxonomy metrics,
3. process / diagnostic metrics,
4. governance metrics.

Optimization priority remains:

- **Primary:** failure taxonomy + process / diagnostic + governance integrity
- **Secondary:** aggregate success / benchmark score / cost

---

## 2. Outcome Metrics

| Metric | Type | Description |
| --- | --- | --- |
| `task.success` | bool | Final output accepted by the active verifier policy |
| `task.validity_score` | float | Continuous validity score |
| `task.confidence` | float | Calibrated confidence |
| `task.proof_validity_score` | float | Proof-validity score when proof-bearing output exists |
| `task.document_grounding_score` | float | Quality of source-grounded reasoning when document anchors exist |

Rules:

- `task.success` must come from verifier logic or an explicitly declared evaluation policy, not raw label lookup.
- `task.confidence` must remain separable from correctness.

---

## 3. Failure Taxonomy

The canonical codes remain:

| Code | Category | Primary Level |
| --- | --- | --- |
| `F_REP` | Representation Failure | `L0/L1` |
| `F_PROC` | Procedural Failure | `L2` |
| `F_SEARCH` | Search / Recovery Failure | `L3` |
| `F_MEM` | Memory Failure | `L4` |
| `F_ABS` | Abstraction Failure | `L5` |
| `F_EVAL` | Evaluation Failure | `L6` |

Every failed or low-confidence attempt must emit `failure.credit` over `L0..L6`.

---

## 4. Level-Specific Diagnostic Metrics

### 4.1 L0 / L1 Representation and Structuring

| Metric | Description |
| --- | --- |
| `rep.document.parse_completeness` | Fraction of required document structure successfully parsed |
| `rep.symbol.binding_error_rate` | Symbol binding or scope error rate |
| `rep.diagram.grounding_score` | Diagram-to-state grounding quality |
| `rep.constraint.coverage` | Coverage of explicit mathematical constraints |
| `rep.tokenizer.ir_fragmentation_rate` | Fragmentation of protected IR/control strings |

### 4.2 L2 Strategy Induction

| Metric | Description |
| --- | --- |
| `prog.diversity` | Diversity of strategy proposals or frontier expansions |
| `proc.frontier.branch_quality` | Quality of frontier candidates |
| `proc.strategy.consistency` | Stability of strategy family under small perturbations |

### 4.3 L3 Search Control and Recovery

| Metric | Description |
| --- | --- |
| `search.depth.max` | Maximum search / reasoning depth |
| `search.budget_pressure` | Budget saturation signal |
| `search.backtrack_rate` | Backtrack frequency |
| `search.recovery_precision` | Fraction of recovery actions hitting the credited failure locus |
| `search.termination_margin` | Margin between stop decision and uncertainty / failure risk |

### 4.4 L4 Memory / Retrieval Binding

| Metric | Description |
| --- | --- |
| `mem.read.similarity` | Retrieval similarity summary |
| `mem.applicability_precision` | Fraction of retrieved items that pass applicability audit |
| `mem.applicability_reject_rate` | Fraction explicitly rejected as unsafe |
| `mem.write.gate` | Write probability or consolidation gate |

### 4.5 L5 Abstraction / Compression

| Metric | Description |
| --- | --- |
| `abs.granularity` | Micro-to-macro abstraction scale |
| `abs.invariant.reuse_rate` | Reuse rate of abstraction outputs |
| `abs.override_rate` | Fraction ignored or overridden downstream |

### 4.6 L6 Verification and Diagnosis

| Metric | Description |
| --- | --- |
| `eval.false_accept_rate` | Invalid solutions or proofs accepted |
| `eval.false_reject_rate` | Valid solutions or proofs rejected |
| `eval.calibration_error` | Calibration error |
| `eval.counterexample_hit_rate` | Rate at which contradiction / counterexample probes succeed when failure exists |
| `eval.disagreement` | Internal verification disagreement |

---

## 5. Representation-Robustness and Concept Metrics

These metrics remain canonical during the transition:

| Metric | Description |
| --- | --- |
| `paired.invariance.gap` | Performance gap across semantically equivalent reformulations |
| `paired.asymmetry_rate` | One-sided success / failure rate for paired variants |
| `concept.success_rate` | Per-concept success rate |
| `concept.isolation_score` | Ability to use a concept without cross-concept interference |
| `concept.leakage_score` | Leakage or interference under mixed conditions |

Use these metrics for document variants, OCR variants, cross-modal variants, and homologous problem restatements, not only legacy ARC-style pairings.

---

## 6. Governance and Data Metrics

| Metric | Description |
| --- | --- |
| `contam.train_visible_overlap_rate` | Estimated overlap between train-visible material and held-out eval |
| `contam.strict_holdout_leakage_score` | Leakage estimate against strict held-out sets |
| `benchmark.tier1.weight` | Realized Tier 1 share |
| `benchmark.tier2.generalization_gap` | Gap between train-visible-adjacent and train-hidden homologous eval |
| `benchmark.tier3.frontier_success_rate` | Success rate on strict held-out frontier eval |
| `provenance.parser_coverage` | Fraction of relevant examples with parser provenance |
| `provenance.formalizer_coverage` | Fraction with formalizer provenance when applicable |
| `provenance.verifier_coverage` | Fraction with verifier build provenance when applicable |

---

## 7. Primary Gate Signals

These remain the highest-priority gate inputs:

| Metric | Gate Intent |
| --- | --- |
| `failure.credit.collapse_rate` | Credit routing must not collapse |
| `eval.calibration_error` | Calibration must not degrade |
| `prog.diversity` | Strategy diversity must persist |
| `search.termination_margin` | Avoid failure-masking early stop |
| `process.failure_distribution_entropy` | Failure diagnostics must remain informative |
| `rep.tokenizer.ir_fragmentation_rate` | Protected control strings must remain stable |
| `contam.strict_holdout_leakage_score` | Strict held-out leakage must stay bounded |
| `provenance.parser_coverage` | Parser provenance coverage must not silently disappear |

If outcome improves while primary gate signals regress, the change is treated as a regression.

---

## 8. Logging and Metadata Requirements

Per-attempt or per-segment logging must preserve:

- `phase`
- `baseline_id`
- `tolerance_profile_id`
- `failure.credit`
- benchmark tier id where relevant
- parser provenance id where relevant
- formalizer version where relevant
- verifier build id where relevant
- dataset slice id / seed when replayable
- resume path id when `S8` is active

Aggregation must not discard tail failures.

---

## 9. Final Invariants

1. No metric may bypass level identity.
2. No failed attempt may omit failure taxonomy.
3. No governance-relevant data source may omit required provenance.
4. No regression may be justified by benchmark score alone.
