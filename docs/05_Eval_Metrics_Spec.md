# Eval Metrics Spec

**Document Type:** Canonical Binding Policy  
**Audience:** Model, training, evaluation, parser, formalizer, and verifier pipelines  
**Scope:** Canonical metric names, field semantics, and required metadata for IRIS-math v2  
**Companion authority:** `docs/13_Benchmark_Registry_and_Tiering_Playbook.md`, `docs/14_Verifier_and_Formalization_Stack.md`, `docs/15_Scaling_Promotion_and_Readiness.md`, `docs/17_Runtime_and_Task_Adjudication_Semantics.md`

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
Benchmark-family-specific governance lives in `docs/13_Benchmark_Registry_and_Tiering_Playbook.md`.
Verifier evidence classes and false accept / false reject policy live in `docs/14_Verifier_and_Formalization_Stack.md`.
Profile-readiness promotion claims live in `docs/15_Scaling_Promotion_and_Readiness.md`.
Task-family acceptance and terminal adjudication semantics live in `docs/17_Runtime_and_Task_Adjudication_Semantics.md`.

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
| `task.success` | bool | Final output accepted by the active task adjudication policy |
| `task.validity_score` | float | Continuous validity score |
| `task.confidence` | float | Calibrated confidence |
| `task.proof_validity_score` | float | Proof-validity score when proof-bearing output exists |
| `task.document_grounding_score` | float | Quality of source-grounded reasoning when document anchors exist |

Rules:

- `task.success` must come from verifier logic interpreted through the active `task_adjudication_policy/v1` or an explicitly declared evaluation policy, not raw label lookup.
- `task.confidence` must remain separable from correctness.
- Any persisted accepted or rejected attempt must use the canonical `runtime_status` and `adjudication_status` vocabularies defined in `docs/17_Runtime_and_Task_Adjudication_Semantics.md`.

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

### 7.1 Bootstrap Profile Hard-Gate Surface Sets

This subsection names the canonical metric bundles that profile-readiness documents may treat as hard-blocking surfaces.
It does **not** activate suites or assign numeric tolerances on its own.

Rules:

1. A declared `baseline_id` and `tolerance_profile_id` must stay fixed across the compared promotion packet.
2. No profile may satisfy its bundle by silently substituting benchmark score for the listed supporting surfaces.
3. If a listed surface is not applicable, the run artifact must state why it is inapplicable and what authoritative replacement surface was used.

| Profile | Minimum hard-gate surfaces | Required directionality |
| --- | --- | --- |
| `P1` | `rep.document.parse_completeness`, `task.document_grounding_score`, `failure.credit.collapse_rate`, `eval.false_accept_rate`, `eval.calibration_error`, `contam.strict_holdout_leakage_score`, `provenance.parser_coverage`, `provenance.verifier_coverage` | parsing and grounding must show material utility; false accept and calibration may not regress; leakage and provenance must remain within declared tolerance |
| `P2` | all `P1` surfaces plus `prog.diversity`, `paired.invariance.gap`, `paired.asymmetry_rate`, `benchmark.tier2.generalization_gap`, `eval.counterexample_hit_rate` | strategy diversity must rise or remain stable; paired robustness may not regress; homologous held-out gap must not widen beyond tolerance; counterexample probing must become more useful, not less |
| `P3` | all `P2` surfaces plus `task.proof_validity_score`, `search.recovery_precision`, `search.termination_margin`, `mem.applicability_precision`, `provenance.formalizer_coverage` when formal bridge is active | proof validity, recovery targeting, and applicability audit must improve or stay stable on harder held-out tasks; early-stop masking and provenance loss are blockers |
| `P4` | all `P3` surfaces plus `task.validity_score`, `benchmark.tier3.frontier_success_rate`, `eval.false_reject_rate`, and frontier-slice `task.document_grounding_score` on document-native held-out tasks | strict held-out frontier success may rise only while all supporting verifier/governance surfaces remain green; false accept and false reject must both remain actionable and bounded by the declared tolerance profile |

If outcome improves while primary gate signals regress, the change is treated as a regression.

### 7.2 Bootstrap Tolerance Profiles

This subsection supplies the default numeric tolerance profiles used by routine readiness review.
These defaults apply only when no stricter benchmark-family, verifier-family, or phase packet rule has been declared.

Normalization rule:

- unless a metric family declares otherwise, rates and scores are interpreted on a normalized `[0, 1]` scale,
- default deltas below are expressed in absolute percentage points (`pp`) on that normalized scale,
- a `+1.0 pp` improvement therefore means `+0.01` absolute.

`tp_p1_bootstrap`

- `rep.document.parse_completeness`: must remain `>= 0.97` and may not drop by more than `0.5 pp` from baseline.
- `task.document_grounding_score`: must improve by at least `+1.0 pp` over baseline on the declared document-native slice.
- `failure.credit.collapse_rate`: must remain `<= 0.02` and may not worsen by more than `0.5 pp`.
- `eval.false_accept_rate`: may not worsen by more than `0.25 pp`.
- `eval.calibration_error`: may not worsen by more than `0.5 pp`.
- `contam.strict_holdout_leakage_score`: may not worsen beyond the declared audit uncertainty band; absent a stricter family rule, treat any increase greater than `0.1 pp` as blocking.
- `provenance.parser_coverage`: must remain `>= 0.95` and may not drop by more than `1.0 pp`.
- `provenance.verifier_coverage`: when verifier-conditioned evaluation is active, must remain `>= 0.90` and may not drop by more than `1.0 pp`.

`tp_p2_bootstrap`

- inherits all `tp_p1_bootstrap` constraints,
- `prog.diversity`: may not worsen by more than `1.0 pp`,
- `paired.invariance.gap`: may not widen by more than `1.0 pp`,
- `paired.asymmetry_rate`: may not worsen by more than `1.0 pp`,
- `benchmark.tier2.generalization_gap`: may not widen by more than `1.0 pp`,
- `eval.counterexample_hit_rate`: must improve by at least `+1.0 pp`.

`tp_p3_bootstrap`

- inherits all `tp_p2_bootstrap` constraints,
- `task.proof_validity_score`: must improve by at least `+1.0 pp`,
- `search.recovery_precision`: must improve by at least `+1.0 pp`,
- `search.termination_margin`: may not worsen by more than `0.5 pp`,
- `mem.applicability_precision`: may not worsen by more than `1.0 pp`,
- `provenance.formalizer_coverage`: when the formal bridge is active, must remain `>= 0.80` and may not drop by more than `1.0 pp`.

`tp_p4_bootstrap`

- inherits all `tp_p3_bootstrap` constraints,
- `task.validity_score`: must improve by at least `+1.0 pp`,
- `benchmark.tier3.frontier_success_rate`: must improve by at least `+1.0 pp`,
- `eval.false_reject_rate`: may not worsen by more than `0.5 pp`,
- frontier-slice `task.document_grounding_score`: must improve by at least `+1.0 pp` on declared document-native held-out tasks.

If a program needs looser thresholds than these bootstrap defaults, the looser profile must be explicitly named, justified, and retained as a separate `tolerance_profile_id`.

---

## 8. Logging and Metadata Requirements

Per-attempt or per-segment logging must preserve:

- `phase`
- `baseline_id`
- `tolerance_profile_id`
- `failure.credit`
- `task_family` where runtime adjudication is active
- `task_adjudication_policy_id` where runtime adjudication is active
- `task_adjudication_policy_resolution_source` where runtime adjudication is active
- `runtime_status` and `adjudication_status` where runtime adjudication is active
- `task_family_resolution_source` where mixed-family benchmark or eval surfaces are active
- `benchmark_family_override_ref` where benchmark-family adjudication tightening is active
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
