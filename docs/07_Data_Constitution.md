# Data Constitution

**Document Type:** Canonical Binding Policy  
**Scope:** Training data pools, benchmark tiering, contamination control, multimodal parsing policy, and provenance requirements for IRIS-math v2  
**Gate policy:** Data-policy changes require regression review under `docs/06_Regression_and_Phase_Gates.md`

---

## 1. Purpose

This document replaces the old fixed-mixture baseline.
It defines the governing rules for:

- which data classes are core,
- which are auxiliary,
- which are weak supervision only,
- which are evaluation only,
- how benchmark data may or may not enter training,
- how multimodal document data is normalized,
- how provenance and contamination are audited.

This is a constitution, not a single frozen recipe.
Exact realized ratios belong to active profiles and run manifests.

---

## 2. Pool Taxonomy

The active pool system is organized around five primary pools.

| Pool | Name | Default Role | Primary Use |
| --- | --- | --- | --- |
| A | Mathematical Knowledge Bedrock | `core` | Definitions, theorem style, symbol sense, domain language |
| B | Procedural Problem-Solving Data | `auxiliary` + `weak_supervision_only` | Stepwise solution traces and strategy exemplars |
| C | Formal / Semi-Formal Data | `core` | Verifier-friendly reasoning and formal alignment |
| D | Document-Native Data | `core` | Long-context document reading, cross-page grounding, layout-aware reasoning |
| E | Synthetic Mechanism Data | `auxiliary` | Recovery, strategy switching, counterexample generation, verifier contrast |

### 2.1 Pool A: Mathematical Knowledge Bedrock

Typical sources:

- textbooks,
- lecture notes,
- mathematical encyclopedias,
- preliminaries / background sections from papers,
- human-readable parts of formal libraries.

This pool is for:

- definition precision,
- statement parsing,
- symbol disambiguation,
- theorem-style familiarity.

### 2.2 Pool B: Procedural Problem-Solving Data

Typical sources:

- olympiad solutions,
- contest writeups,
- worked examples,
- high-quality forum or StackExchange solutions,
- proof-oriented teaching materials.

Constraints:

- treat as weak procedural evidence, not guaranteed gold proofs,
- do not let template-heavy or jump-heavy solutions dominate,
- do not treat final-answer correctness as proof correctness.

### 2.3 Pool C: Formal / Semi-Formal Data

Typical sources:

- Lean / Isabelle / Coq style proofs,
- theorem / lemma / tactic traces,
- natural-to-formal aligned pairs,
- semi-formal proof skeletons.

This pool is critical for:

- verifier grounding,
- proof validity signals,
- formalization transfer.

### 2.4 Pool D: Document-Native Data

Typical sources:

- PDFs,
- lecture notes,
- textbook chapters,
- papers,
- scanned notes with OCR + layout parsing,
- diagram-bearing materials.

This pool is for:

- long-range document reasoning,
- cross-page reference resolution,
- theorem-proof chain reconstruction,
- formula / table / diagram grounding.

### 2.5 Pool E: Synthetic Mechanism Data

Synthetic data exists to reinforce mechanisms, not to pour in answers.

Preferred uses:

- failure recovery,
- strategy switching,
- equivalent reformulation,
- counterexample generation,
- state tracking,
- verifier contrast pairs,
- correct vs subtly invalid proof discrimination.

---

## 3. Benchmark Tiering Policy

Benchmark usage is governed by three tiers.

| Tier | Visibility | Allowed Use | Default Role |
| --- | --- | --- | --- |
| 1 | train-visible | structural signals, process fragments, curriculum anchors | `auxiliary` |
| 2 | train-hidden | homologous held-out evaluation | `eval_only` |
| 3 | strict held-out | frontier evaluation | `eval_only` |

### 3.1 Tier 1: Train-Visible Benchmark Pool

Allowed uses:

- problem-type labels,
- difficulty bands,
- solution-family tags,
- early curriculum anchoring,
- process fragments,
- verifier contrast pairs,
- representation or first-step structure signals.

Not allowed:

- letting benchmark full problems plus full gold solutions become the dominant supervised corpus,
- using leaderboard-style data as the main training distribution,
- hiding train-visible benchmark usage.

### 3.2 Tier 2: Train-Hidden Homologous Eval

Tier 2 exists to answer:

- did the model generalize beyond seen benchmark items,
- did it memorize template families,
- did train-visible benchmark exposure distort local generalization.

### 3.3 Tier 3: Strict Held-Out Frontier Eval

Tier 3 must remain unseen during training and curriculum tuning.

Use this tier for:

- frontier math evaluation,
- adversarial reformulations,
- cross-modal variants,
- document-grounded theorem tracing,
- proof repair evaluation.

---

## 4. Contamination and Decontamination Rules

Every training program must declare a contamination-control plan.

Minimum requirements:

1. source-level hashing and provenance retention,
2. document-level deduplication,
3. problem- and theorem-level normalized fingerprints where applicable,
4. homologous split policy for benchmark-adjacent corpora,
5. explicit disclosure of which benchmark families are Tier 1 train-visible,
6. strict separation of Tier 2 / Tier 3 from training,
7. reformulation and near-duplicate checks for document, OCR, and diagram variants.

If contamination cannot be measured, the run is not governance-complete.

---

## 5. Multimodal Parse Canonical Artifact

Document-derived and multimodal sources must be normalized into a canonical parse artifact before downstream packing.

Canonical artifact name:

`math_document_record/v1`

Minimum fields:

- `record_id`
- `source_id`
- `source_uri_or_snapshot`
- `content_sha256`
- `pages`
- `reading_order`
- `clean_text_blocks`
- `formula_blocks`
- `table_blocks`
- `diagram_regions`
- `section_structure`
- `math_semantic_units` (`definition`, `theorem`, `lemma`, `proof`, `example`, `exercise`, `remark`, `figure`)
- `cross_reference_edges`
- `parse_confidence`
- `parser_provenance_id`
- `ocr_layout_extractor_version`
- `formula_parser_version`

This artifact is not State IR.
It is the canonical ingestion object from which State IR-aligned supervision or inputs are later derived.

---

## 6. Provenance Requirements

Every pool realization must retain auditable provenance.

Required dimensions include:

- dataset or corpus identity,
- dataset revision or snapshot id,
- sampling policy id,
- parser provenance id,
- OCR/layout extractor version,
- formalizer version when used,
- verifier build provenance when used,
- license / attribution coverage.

Document-derived, formalized, or verifier-labeled data without provenance is non-compliant.

---

## 7. Mixture Realization Rules

This document intentionally does **not** freeze a universal `90 / 10 / 0` recipe.

Instead:

- each active training profile must declare the realized pool weights,
- each run must declare the realized benchmark tier usage,
- each run must declare whether a pool is used as `core`, `auxiliary`, `weak_supervision_only`, or `eval_only`.

Non-negotiable rules:

1. Pool roles must be explicit.
2. Tier 2 and Tier 3 remain evaluation-only.
3. Tier 1 train-visible benchmark usage must be disclosed and contamination-audited.
4. Benchmark data must never dominate the full supervised mixture.
5. Synthetic data must reinforce mechanisms, not replace real mathematical distributions.
6. Multimodal document data must enter through the canonical parse artifact path.

---

## 8. Monitoring and Gate Inputs

Training and data pipelines must track at least:

- realized pool ratios,
- realized benchmark-tier ratios,
- parser provenance coverage,
- formalizer provenance coverage,
- verifier provenance coverage,
- contamination audit status,
- document parse quality,
- weak-supervision share,
- strict held-out leakage score.

Significant drift or missing provenance requires investigation and may block promotion.

---

## 9. Explicit Non-Goals

This document does not:

- fix exact model-size-specific hyperparameters,
- declare a single permanent token ratio recipe,
- authorize uncontrolled benchmark mixing,
- let OCR or parser outputs bypass normalization.
