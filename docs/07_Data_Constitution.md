# Data Constitution

**Document Type:** Canonical Binding Policy  
**Scope:** Training data pools, benchmark tiering, contamination control, multimodal parsing policy, and provenance requirements for IRIS-math v2  
**Gate policy:** Data-policy changes require regression review under `docs/06_Regression_and_Phase_Gates.md`  
**Related active docs:** `docs/14_Multimodal_Document_Pipeline.md`, `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`, `docs/17_Scaling_Promotion_and_Readiness.md`

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
This document therefore defines the executable policy objects and minimum compliance fields that profiles and runs must instantiate.

Benchmark family-specific governance is maintained in `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.
Document-pipeline mechanics are elaborated in `docs/14_Multimodal_Document_Pipeline.md`.

---

## 1.1 Executable Policy Surface

The following policy objects make this constitution operational.
The constitution defines their shape and required fields.
Profiles, manifests, and registries instantiate concrete values.

| Object | Scope | Purpose |
| --- | --- | --- |
| `data_realization_policy/v1` | profile / phase / run family | Declares pool allocations, role usage, admission gates, Tier 1 cap, weak-supervision cap, and references to benchmark and decontamination policy objects |
| `benchmark_family_policy/v1` | benchmark family | Declares train-visible unit classes, homology axes, source-lineage firewalls, strict held-out source, tuning firewalls, derivative-family refs, and forbidden uses |
| `decontam_policy/v1` | data program | Declares fingerprint layers, normalization rules, near-duplicate thresholds, homologous split rules, and leakage-audit method |
| `provenance_manifest/v1` | parser / formalizer / verifier / extractor surface | Declares immutable backend identity, build/config fingerprint, license state, and replay-relevant provenance |

Rules:

1. These objects must be versioned and immutable once referenced by a committed run.
2. Exact ratios and thresholds live in these instantiated objects, not in the constitution prose.
3. A run is not governance-complete unless its referenced policy objects can be resolved.

---

## 1.2 Bootstrap Execution Chain

The executable data path is:

`capture -> extract -> canonicalize -> fingerprint -> quality_gate -> admit/quarantine -> pack -> journal`

Bootstrap `P1-P3` realizations must make every stage explicit.

| Stage | Primary object | Minimum required outputs | Allowed downgrade / reject action | Required landing surface |
| --- | --- | --- | --- | --- |
| `capture` | source snapshot or raw export | `source_id`, snapshot identity, source hash, license / attribution state, source-family tag | reject if source identity or attribution is missing | source registry or immutable source snapshot |
| `extract` | parser / OCR / formula / semantic-unit output | parser manifests, extractor confidence, unresolved-region flags, source-local anchors | downgrade to `auxiliary` or `eval_only` when structure survives but quality thresholds miss | parse artifact staging plus parser provenance registry |
| `canonicalize` | `math_document_record/v1` or equivalent pool record | canonical anchors, reading order, normalized clean text / formulas, semantic units, quality flags | reject if canonical anchors or replay-critical quality flags are absent | record surface |
| `fingerprint` | `fingerprint_set` | `source_fingerprint`, `document_fingerprint`, semantic-unit / theorem / fragment / diagram layers as required | quarantine on near-duplicate or homologous firewall hit; drop on strict held-out leakage | contamination audit store plus record surface |
| `quality_gate` | pool-local quality decision | Pool B cleaning fields, Pool D admission scores, provenance completeness, benchmark tier tags where relevant | downgrade to fragment-only, `auxiliary`, or `eval_only`; reject if minimum booleans or provenance fail | record surface |
| `admit/quarantine` | policy decision record | `pool_id`, `pool_role`, `benchmark_tier`, admission outcome, quarantine reason when applicable | quarantine or reject must remain explicit and auditable | run-manifest staging plus audit summary |
| `pack` | train-visible or eval-only packed record | `data_realization_policy_id`, `decontam_policy_id`, resolvable provenance refs, lineage to canonical record | reject if any train-visible contract field is missing | train-visible record surface or eval pack surface |
| `journal` | segment journal / checkpoint / run manifest | stable `dataset_slice_id`, active policy ids, parser/formalizer/verifier provenance ids, human-readable version summaries | segment is non-compliant if policy or provenance ids are unresolved | run surface |

Rules:

1. Downgrade is not silent acceptance; the downgraded role or fragment-only restriction must persist on the record.
2. Quarantine is distinct from drop. Quarantined material may not become train-visible until the blocking audit is cleared under a new policy or review.
3. Train-visible packing may occur only after canonicalization, fingerprinting, and quality gating have all completed.

### 1.2.1 Stage-to-Surface Landing Matrix

| Field family | Record-level persistence | Segment/run persistence |
| --- | --- | --- |
| parser / OCR / formula / semantic-unit provenance | required on `math_document_record/v1` or train-visible record via manifest ids or `not_applicable` | required in journal/checkpoint whenever the segment uses document-derived or multimodal data |
| formalizer provenance | required on every formalized or semi-formal derived record | required in journal/checkpoint whenever formalizer-derived supervision is active |
| verifier provenance | required on verifier-labeled train-visible records or eval artifacts | required in journal/checkpoint whenever verifier-generated labels or verifier-conditioned gating is active |
| contamination fingerprints | required on every train-visible record | required as audit summary and policy id on the run surface |
| benchmark family policy refs | required on benchmark-derived records | required on journal/checkpoint/run manifest whenever benchmark-derived data is present |

If a surface does not apply, the value must be stored as `not_applicable` rather than omitted silently.

---

## 1.3 Bootstrap Example Objects

These examples are normative for shape and field presence, not for vendor selection.

Example `provenance_manifest/v1`:

```json
{
  "manifest_id": "layout-parser-v1",
  "surface_kind": "parser",
  "backend_family": "layout_parser",
  "backend_version": "bootstrap-v1",
  "build_or_commit_hash": "layout-bootstrap-001",
  "config_fingerprint": "cfg-layout-bootstrap",
  "artifact_fingerprint": "artifact-layout-bootstrap",
  "parent_manifest_ids": [],
  "license_scope": "internal-eval-approved",
  "created_at": "2026-03-11T00:00:00Z"
}
```

Example `decontam_policy/v1`:

```json
{
  "decontam_policy_id": "global-decontam-v1",
  "normalization_profile_id": "nfkc-math-doc-v1",
  "layer_payload_specs": {
    "source_fingerprint": "raw source snapshot bytes",
    "document_fingerprint": "clean text + formula inventory + anchors",
    "theorem_or_problem_fingerprint": "alpha-renamed statement + formula multiset"
  },
  "layer_thresholds": {
    "document_fingerprint": {
      "near_duplicate_min_hash_jaccard": 0.9
    }
  },
  "homologous_split_rules": {
    "default": "tier2 and tier3 must remain fingerprint-disjoint from tier1 after canonicalization"
  },
  "audit_method": {
    "cadence": "per run manifest"
  }
}
```

Example `benchmark_family_policy/v1`:

```json
{
  "benchmark_family_id": "aimo-v1",
  "allowed_tiers": ["Tier 1", "Tier 2", "Tier 3"],
  "tier1_train_visible_units": ["structural_label", "process_fragment"],
  "tier1_label_allowlist": ["domain", "problem_archetype", "difficulty_band"],
  "tier1_fragment_allowlist": ["first_representation_step", "branch_choice_snippet"],
  "tier2_homologous_source_id": "aimo-homologous-heldout-v1",
  "homology_axes": ["problem_type", "proof_pattern", "difficulty_band"],
  "source_lineage_firewall": ["contest_series", "contest_year", "source_snapshot"],
  "cluster_exclusion_key": "contest_year_problem_cluster",
  "tier3_strict_holdout_source_id": "aimo-post-cutoff-private-like-v1",
  "decontam_policy_id": "global-decontam-v1",
  "tuning_visible_surfaces": ["benchmark.tier1.weight", "contamination.summary"],
  "tuning_observe_only_surfaces": ["benchmark.tier2.generalization_gap"],
  "tuning_blocked_surfaces": ["tier3_all", "tier2_item_outputs"],
  "forbidden_uses": ["full_official_solution_mixing", "tier2_hyperparameter_sweeps"]
}
```

Example `data_realization_policy/v1`:

```json
{
  "data_realization_policy_id": "p1-bootstrap-b-v1",
  "profile_id": "P1",
  "phase": "B",
  "pool_allocations": {
    "A": {"token_weight": 20.0, "record_weight": 25.0, "allowed_roles": ["core"]},
    "B": {"token_weight": 10.0, "record_weight": 10.0, "allowed_roles": ["auxiliary", "weak_supervision_only"]},
    "C": {"token_weight": 25.0, "record_weight": 25.0, "allowed_roles": ["core"]},
    "D": {"token_weight": 35.0, "record_weight": 20.0, "allowed_roles": ["core", "auxiliary", "eval_only"]},
    "E": {"token_weight": 10.0, "record_weight": 20.0, "allowed_roles": ["auxiliary"]}
  },
  "tier1_global_cap": {"token_cap": 5.0, "record_cap": 5.0},
  "weak_supervision_cap": {"token_cap": 15.0, "record_cap": 15.0},
  "benchmark_family_policy_refs": ["aimo-v1", "omni-math-v1", "miniF2F-v1", "frontiermath-original-v1"],
  "decontam_policy_id": "global-decontam-v1"
}
```

These examples may be strengthened in concrete manifests, but a compliant bootstrap object may not omit any field family shown above.

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

#### 2.2.1 Executable Cleaning Fields for Pool B

Every Pool B train-visible record must retain the following cleaning fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `answer_reasoning_separated` | bool | Confirms final answer and intermediate reasoning were split |
| `trace_reconstructable` | bool | Confirms the remaining trace is usable as weak procedural evidence |
| `jump_flags` | list or count | Marks unsupported jumps or omitted justifications |
| `template_overlap_score` | float or band | Captures boilerplate / farm-like template overlap |
| `hidden_lemma_risk` | float or band | Marks likely unstated theorem or lemma dependence |
| `contradiction_flag` | bool | Marks contradiction with stated assumptions, source statement, or later-corrected steps |
| `duplicate_cluster_size` | int | Counts near-duplicate siblings after solution-farm clustering |
| `provenance_ok` | bool | Confirms source identity and attribution are intact |

Admission rule:

- A Pool B record is non-compliant for train-visible use if `answer_reasoning_separated = false`, `trace_reconstructable = false`, or `provenance_ok = false`.
- Thresholds for `jump_flags`, `template_overlap_score`, and `hidden_lemma_risk` must be declared in the active `data_realization_policy/v1`.

#### 2.2.2 Bootstrap Cleaning and Downgrade Table for Pool B

Bootstrap `data_realization_policy/v1` objects for `P1-P3` must treat `template_overlap_score` and `hidden_lemma_risk` as normalized `[0,1]` values.
`jump_flags` must be stored both as a count and as anchorable locations back to the source trace.

| Outcome | Minimum conditions | Allowed downstream use |
| --- | --- | --- |
| `accept_train_visible` | `answer_reasoning_separated = true`; `trace_reconstructable = true`; `provenance_ok = true`; `contradiction_flag = false`; `jump_flags <= 2`; `template_overlap_score < 0.35`; `hidden_lemma_risk < 0.50`; `duplicate_cluster_size <= 3` | weak procedural supervision, first-step structure, verifier contrast, short process fragments |
| `downgrade_fragment_only` | same boolean requirements as accept; `jump_flags = 3..4` or `template_overlap_score = 0.35..0.60` or `hidden_lemma_risk = 0.50..0.75` or `duplicate_cluster_size = 4..5` | fragment-only use: representation labels, branch-choice snippets, verifier contrast pairs, or short proof-shape fragments |
| `reject` | any of: `answer_reasoning_separated = false`; `trace_reconstructable = false`; `provenance_ok = false`; `contradiction_flag = true`; `jump_flags >= 5`; `template_overlap_score > 0.60`; `hidden_lemma_risk > 0.75`; `duplicate_cluster_size > 5` | no train-visible use |

Rules:

1. `downgrade_fragment_only` records may not contribute full-trace supervision.
2. Pool B records may not be relabeled as proof-valid gold without verifier-backed or formal-backed upgrade evidence.
3. Threshold relaxation requires a new `data_realization_policy_id` and regression review.

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
- DOCX lecture notes or manuscripts,
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

#### 2.4.1 Source Admission Rules for Pool D

Pool D admission must distinguish `core`, `auxiliary`, and `eval_only` intake.
The active `data_realization_policy/v1` must declare the exact thresholds and fallback rules for each source family.

| Source Family | Minimum admission requirements | Default fallback if incomplete |
| --- | --- | --- |
| `PDF` | stable page anchors, reading order, formula block extraction, parser provenance | downgrade to `auxiliary` if anchor traceability survives but structure is incomplete |
| `DOCX` | section hierarchy, equation object extraction or formula fallback, figure/table linkage, parser provenance | downgrade to `auxiliary` if hierarchy survives but embedded math coverage is partial |
| `image` | OCR/layout confidence, bbox coverage, text/formula region separation, parser provenance | downgrade to `eval_only` if anchors survive but text structure is weak |
| `scanned_note` | OCR/layout confidence, page/canvas anchors, unresolved-region flags, parser provenance | downgrade to `eval_only` unless handwriting/layout uncertainty is explicitly retained |
| `diagram` | region anchors, label extraction, caption or theorem/problem linkage, parser provenance | keep as `auxiliary` candidate structure only until context linkage exists |

Reject rule:

- Pool D records are non-compliant for train-visible use if they lack canonical anchors, parser provenance, or document-quality flags needed for replay and audit.

#### 2.4.2 Bootstrap Admission Matrix for Pool D

Bootstrap `data_realization_policy/v1` objects for `P1-P3` must use the following default admission thresholds.
Thresholds are minimums; stronger thresholds are allowed.

| Source Family | `core` admit | `auxiliary` admit | `eval_only` fallback | `reject` when |
| --- | --- | --- | --- | --- |
| `PDF` | `anchor_coverage_score >= 0.98`; `reading_order_confidence >= 0.95`; `math_region_coverage >= 0.90`; `parse_confidence >= 0.90`; parser provenance complete | `anchor_coverage_score >= 0.90`; `reading_order_confidence >= 0.85`; `math_region_coverage >= 0.75`; `parse_confidence >= 0.80` | not used by default | below auxiliary thresholds or provenance incomplete |
| `DOCX` | `section_hierarchy_completeness >= 0.95`; `equation_extraction_coverage >= 0.90`; `figure_table_linkage >= 0.90`; `parse_confidence >= 0.90`; parser provenance complete | `section_hierarchy_completeness >= 0.85`; `equation_extraction_coverage >= 0.70`; `figure_table_linkage >= 0.70`; `parse_confidence >= 0.80` | not used by default | below auxiliary thresholds or provenance incomplete |
| `image` | `ocr_text_confidence >= 0.97`; `bbox_coverage_score >= 0.95`; `text_formula_separation_score >= 0.90`; `unresolved_region_ratio <= 0.03`; parser provenance complete | `ocr_text_confidence >= 0.92`; `bbox_coverage_score >= 0.90`; `text_formula_separation_score >= 0.80`; `unresolved_region_ratio <= 0.08` | `ocr_text_confidence >= 0.85`; `bbox_coverage_score >= 0.80`; anchors intact; `unresolved_region_ratio <= 0.15` | below eval thresholds, anchors missing, or provenance incomplete |
| `scanned_note` | not allowed in bootstrap `P1-P3` core intake | `ocr_text_confidence >= 0.90`; `bbox_coverage_score >= 0.85`; `unresolved_region_ratio <= 0.10`; `handwriting_uncertainty_retained = true`; parser provenance complete | `ocr_text_confidence >= 0.80`; `bbox_coverage_score >= 0.70`; anchors intact; `unresolved_region_ratio <= 0.20`; `handwriting_uncertainty_retained = true` | uncertainty flags missing, anchors missing, or below eval thresholds |
| `diagram` | not allowed as standalone core intake in bootstrap `P1-P3` | `region_anchor_coverage >= 0.90`; `label_extraction_score >= 0.85`; `context_linkage_score >= 0.85`; parser provenance complete | `region_anchor_coverage >= 0.75`; `label_extraction_score >= 0.70`; caption or theorem/problem linkage preserved | context linkage absent, anchors missing, or provenance incomplete |

Rules:

1. Standalone `diagram` sources remain `auxiliary` until they are bound to theorem/problem context through the canonical record.
2. `scanned_note` sources may be train-visible only when uncertainty is explicitly preserved; silent denoising is non-compliant.
3. Source-family-specific overrides must be disclosed in the active `data_realization_policy/v1`.

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

Benchmark family registry and family-specific allowed uses are governed by `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.

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

### 3.1.1 Tier 1 Unit Classes and Default Admissibility

Every `benchmark_family_policy/v1` must classify Tier 1 material into the unit classes below.
Family policies may be stricter than this table, but may not be looser without a new policy id and regression review.

| Unit class | Typical examples | Default admissibility | Additional requirement |
| --- | --- | --- | --- |
| `structural_label` | problem family, theorem family id, proof-pattern id, output form, difficulty band | allowed when explicitly declared | label vocabulary and derivation rule must be declared in the family policy |
| `process_fragment` | first representation step, branch-choice snippet, invariant proposal, verifier contrast pair | allowed when explicitly declared | fragment must not reconstruct the full official solution on its own |
| `theorem_or_problem_statement` | benchmark statement text, formal theorem statement | blocked by default | family policy must explicitly allow it and Tier 2 must split by theorem/problem family identity |
| `proof_shape_fragment` | lemma skeleton, tactic-family outline, non-terminal proof plan | conditional | fragment must be canonicalized to omit decisive terminal derivations and Tier 2 must split by `proof_pattern` |
| `full_problem_solution_pair` | full item plus end-to-end official solution or proof | blocked for the registered families in `docs/15_Benchmark_Registry_and_Tiering_Playbook.md` | future exceptions require explicit family registration and regression review |

Rules:

1. `difficulty_band` is a `structural_label`; it is compliant only when produced from a declared binning policy rather than hidden leaderboard order.
2. If `theorem_or_problem_statement` is Tier 1-visible, the record must retain theorem/problem fingerprints and source-lineage metadata.
3. If `proof_shape_fragment` is Tier 1-visible, the record must retain proof-pattern fingerprints and source-lineage metadata.

### 3.2 Tier 2: Train-Hidden Homologous Eval

Tier 2 exists to answer:

- did the model generalize beyond seen benchmark items,
- did it memorize template families,
- did train-visible benchmark exposure distort local generalization.

### 3.2.1 Homologous Construction Rule

Tier 2 is not "same benchmark, different split" by default.
Every Tier 1-eligible `benchmark_family_policy/v1` must declare how homologous evaluation is built.

Required fields for Tier 1-eligible families:

- `homology_axes`: ordered axes chosen from `problem_type`, `theorem_family`, `proof_pattern`, `difficulty_band`, `representation_style`, and `modality_variant`
- `source_lineage_firewall`: source grouping that may not cross Tier 1 / Tier 2 / Tier 3
- `cluster_exclusion_key`: fingerprint or family key used to keep homologous clusters disjoint

Rules:

1. Tier 2 must match Tier 1 distribution on at least two declared homology axes.
2. At least one homology axis must be structural (`problem_type`, `theorem_family`, or `proof_pattern`), not only `difficulty_band`.
3. `source_lineage_firewall` is a separation key, not a substitute for a homology axis.
4. If theorem/problem statements are Tier 1-visible, `theorem_family` or a theorem/problem-family-equivalent key must appear in `homology_axes`.
5. If proof-shape fragments are Tier 1-visible, `proof_pattern` must appear in `homology_axes`.
6. Tier 2 and Tier 3 must remain fingerprint-disjoint from Tier 1 after canonicalization, reformulation, and translation checks.

### 3.3 Tier 3: Strict Held-Out Frontier Eval

Tier 3 must remain unseen during training and curriculum tuning.

Use this tier for:

- frontier math evaluation,
- adversarial reformulations,
- cross-modal variants,
- document-grounded theorem tracing,
- proof repair evaluation.

### 3.3.1 Default Tuning Firewalls for Benchmark Evaluation

Once a benchmark family has any Tier 1 train-visible exposure, the following defaults apply unless the family policy in `docs/15_Benchmark_Registry_and_Tiering_Playbook.md` is stricter.

`observe_only` surfaces allowed at fixed declared cadence:

- family-level Tier 2 aggregate metrics,
- family-level `benchmark.tier2.generalization_gap`,
- contamination, provenance, and leakage summaries,
- declared pass/fail gate summaries required by `docs/06_Regression_and_Phase_Gates.md` or `docs/17_Scaling_Promotion_and_Readiness.md`.

`tuning_blocked` surfaces:

- all Tier 3 metrics, items, explanations, and slice breakdowns during curriculum shaping, checkpoint selection, or hyperparameter selection,
- all item-level Tier 2 outputs,
- Tier 2 breakdowns keyed directly to the family's declared homology axes,
- repeated adaptive prompt, decoding, or hyperparameter sweeps against the same Tier 2 split,
- importing Tier 2 or Tier 3 misses back into train-visible curation without a newly declared benchmark family policy or derivative-family policy.

---

## 4. Contamination and Decontamination Rules

Every training program must declare a contamination-control plan.

Minimum requirements:

1. source-level hashing and provenance retention,
2. document-level deduplication,
3. semantic-unit-level fingerprints for theorem / proof / definition / exercise objects where applicable,
4. problem- and theorem-level normalized fingerprints where applicable,
5. solution- or proof-fragment fingerprints where applicable,
6. diagram-anchor fingerprints for multimodal or diagram-bearing sources where applicable,
7. homologous split policy for benchmark-adjacent corpora,
8. explicit disclosure of which benchmark families are Tier 1 train-visible,
9. strict separation of Tier 2 / Tier 3 from training,
10. reformulation and near-duplicate checks for document, OCR, and diagram variants.

Parallel fingerprint layers required in every `decontam_policy/v1`:

| Fingerprint Layer | Required Use |
| --- | --- |
| `source_fingerprint` | raw-source snapshot identity and coarse dedup |
| `document_fingerprint` | full-document or page-level dedup |
| `semantic_unit_fingerprint` | theorem / proof / definition / exercise family separation |
| `theorem_or_problem_fingerprint` | benchmark-adjacent and solver-facing dedup |
| `solution_or_proof_fragment_fingerprint` | process-fragment and weak-trace leakage checks |
| `diagram_anchor_fingerprint` | multimodal, OCR, and diagram variant leakage checks |

Rules:

1. No single fingerprint layer is sufficient on its own.
2. The active `decontam_policy/v1` must declare which layers are required per pool and per benchmark family.
3. If a layer is not applicable, the policy must record `not_applicable` explicitly rather than omitting the layer silently.

### 4.1 Canonicalization Required Before Fingerprinting

Every `decontam_policy/v1` must declare a canonicalization profile applied before any near-duplicate comparison.

Minimum normalization steps:

1. Unicode normalization to `NFKC` plus whitespace collapse.
2. Header/footer/page-number stripping or explicit side-channel retention before document-level hashing.
3. Formula normalization with standardized delimiters and macro canonicalization; both raw-formula and alpha-renamed variants must be retained where theorem/problem identity matters.
4. OCR uncertainty retention: low-confidence spans may be masked or tagged, but not silently dropped.
5. Diagram normalization into label sets, relation candidates, and anchor-topology summaries before diagram-level comparison.

### 4.2 Bootstrap Fingerprint Payloads and Duplicate Thresholds

Bootstrap `decontam_policy/v1` objects for `P1-P3` must declare at least the following payloads and thresholds.

| Fingerprint Layer | Minimum canonical payload | Exact-duplicate rule | Near-duplicate quarantine rule |
| --- | --- | --- | --- |
| `source_fingerprint` | raw source snapshot bytes or exported source package | identical `sha256` | not used; source layer is exact-match only |
| `document_fingerprint` | canonical clean text, formula inventory, page anchors, and document metadata | identical `sha256` over canonical document payload | quarantine when `5`-gram MinHash Jaccard `>= 0.90` or page-anchor-plus-formula overlap `>= 0.90` |
| `semantic_unit_fingerprint` | unit type plus normalized text/formula payload for `definition` / `theorem` / `lemma` / `proof` / `exercise` / `example` | identical unit hash | quarantine when normalized-unit MinHash Jaccard `>= 0.88` |
| `theorem_or_problem_fingerprint` | alpha-renamed normalized statement text plus normalized formula multiset | identical statement hash | quarantine when token Jaccard `>= 0.85` and formula multiset overlap `>= 0.90` |
| `solution_or_proof_fragment_fingerprint` | ordered claim fragments, step-boundary markers, and normalized local formulas | identical fragment hash chain | quarantine when fragment MinHash Jaccard `>= 0.80` and claim-dependency signature matches |
| `diagram_anchor_fingerprint` | label set, relation-candidate set, and bbox-topology summary | identical diagram signature hash | quarantine when label-set overlap `>= 0.85` and topology overlap `>= 0.80` |

### 4.3 Required Fingerprint Layers by Pool

Bootstrap `decontam_policy/v1` objects must require the following layer coverage.

| Pool | `source_fingerprint` | `document_fingerprint` | `semantic_unit_fingerprint` | `theorem_or_problem_fingerprint` | `solution_or_proof_fragment_fingerprint` | `diagram_anchor_fingerprint` |
| --- | --- | --- | --- | --- | --- | --- |
| `A` | required | required | required | conditional when exercise/problem-bearing | conditional for worked-example slices | conditional for diagram-bearing records |
| `B` | required | required | conditional | required | required | conditional for geometry/diagram-bearing traces |
| `C` | required | required | required | required | required for proof-bearing records | conditional |
| `D` | required | required | required | conditional when theorem/problem/exercise units exist | conditional when proof/solution units exist | required for `image`, `scanned_note`, and `diagram` sources |
| `E` | required | conditional when derived from documents | conditional | conditional | required when synthetic traces are derived from real parent traces | conditional |

### 4.4 Minimum Fields of `decontam_policy/v1`

Every executable decontamination policy must declare at least:

| Field | Requirement |
| --- | --- |
| `decontam_policy_id` | immutable policy identity |
| `normalization_profile_id` | canonicalization profile used before fingerprinting |
| `layer_payload_specs` | declared payload recipe per fingerprint layer |
| `layer_thresholds` | exact and near-duplicate thresholds per layer |
| `pool_layer_requirements` | required / conditional / not-applicable layer matrix by pool |
| `benchmark_family_overrides` | family-specific overrides where benchmark governance is stricter |
| `homologous_split_rules` | held-out split policy for benchmark-adjacent corpora |
| `quarantine_and_drop_rules` | actions taken when duplicates or leakage are detected |
| `audit_method` | leakage-audit method and review cadence |

If contamination cannot be measured, the run is not governance-complete.

---

## 5. Multimodal Parse Canonical Artifact

Document-derived and multimodal sources must be normalized into a canonical parse artifact before downstream packing.

Canonical artifact name:

`math_document_record/v1`

PDF and DOCX are first-class document-native sources and must normalize through this same canonical artifact path.

Minimum fields:

- `record_id`
- `source_id`
- `source_format`
- `source_uri_or_snapshot`
- `content_sha256`
- `pages`
- `reading_order`
- `anchor_index`
- `clean_text_blocks`
- `formula_blocks`
- `table_blocks`
- `diagram_regions`
- `section_structure`
- `math_semantic_units` (`definition`, `theorem`, `lemma`, `proof`, `example`, `exercise`, `remark`, `figure`)
- `cross_reference_edges`
- `parse_confidence`
- `semantic_unit_confidence`
- `bbox_coverage_score`
- `record_quality_flags`
- `parser_provenance_id`
- `parser_provenance_refs` (`layout_parser_manifest_id`, `ocr_manifest_id`, `formula_parser_manifest_id`, `semantic_unit_typer_manifest_id`)
- `ocr_layout_extractor_version`
- `formula_parser_version`
- `semantic_unit_typer_version`
- `parse_config_fingerprint`
- `unresolved_region_ratio`

This artifact is not State IR.
It is the canonical ingestion object from which State IR-aligned supervision or inputs are later derived.

### 5.1 Parse-Stage Provenance Persistence

Every `math_document_record/v1` must retain parser provenance at both pipeline and component granularity.

| Field | Requirement |
| --- | --- |
| `parser_provenance_id` | canonical parse-pipeline manifest id for the record |
| `parser_provenance_refs.layout_parser_manifest_id` | required when layout parsing is active |
| `parser_provenance_refs.ocr_manifest_id` | required when OCR is active, including OCR-backed PDF extraction |
| `parser_provenance_refs.formula_parser_manifest_id` | required when formula blocks or expression structure are extracted |
| `parser_provenance_refs.semantic_unit_typer_manifest_id` | required when theorem/proof/lemma/definition typing is mounted |
| `parse_config_fingerprint` | hash of behavior-affecting parse config |
| `ocr_layout_extractor_version` / `formula_parser_version` / `semantic_unit_typer_version` | convenience summaries for reporting; not authoritative without manifest ids |

Rules:

1. Component manifests must resolve to immutable `provenance_manifest/v1` objects.
2. Version strings without manifest ids are insufficient for train-visible admission.
3. If a surface is not mounted, the record must store `not_applicable` rather than omitting the key silently.

---

## 6. Provenance Requirements

Every pool realization must retain auditable provenance.
Parser-, formalizer-, verifier-, and extractor-related provenance must resolve to immutable `provenance_manifest/v1` objects.

Required dimensions include:

- dataset or corpus identity,
- dataset revision or snapshot id,
- sampling policy id,
- parser provenance id and component parser provenance refs when used,
- OCR/layout extractor version summary when used,
- formalizer provenance id and formalizer version summary when used,
- verifier provenance id and verifier build summary when used,
- license / attribution coverage.

Minimum fields of `provenance_manifest/v1`:

- `manifest_id`
- `surface_kind`
- `backend_family`
- `backend_version`
- `build_or_commit_hash`
- `config_fingerprint`
- `artifact_fingerprint`
- `parent_manifest_ids`
- `license_scope`
- `created_at`

Document-derived, formalized, or verifier-labeled data without provenance is non-compliant.

### 6.1 Provenance Landing Surfaces

Provenance is not governance-complete unless it lands on all three surfaces below.

| Landing Surface | Required persistence |
| --- | --- |
| provenance registry | immutable `provenance_manifest/v1` objects keyed by manifest id |
| record surface | `math_document_record/v1`, formalized records, and train-visible records must retain manifest ids in `provenance_refs` |
| run surface | segment journal, checkpoints, and run manifests must retain the active manifest ids plus human-readable version summaries |

### 6.2 Formalizer Provenance Contract

When Pool C data, verifier labels, or synthetic records are produced by natural-to-formal or semi-formal conversion, the derived record must retain at least:

- `formalizer_provenance_id`
- `formalizer_version`
- `formalization_mode`
- `formalizer_input_record_id`
- `formalizer_output_artifact_id`
- `formalization_confidence`
- `formal_check_status` when a checker bridge is used

Formalized records without input lineage or formalizer provenance are non-compliant.

---

## 7. Mixture Realization Rules

This document intentionally does **not** freeze a universal `90 / 10 / 0` recipe.

Instead:

- each active training profile must publish a `data_realization_policy/v1`,
- each active training run must reference that policy via `data_realization_policy_id`,
- each run must declare the realized benchmark tier usage,
- each run must declare whether a pool is used as `core`, `auxiliary`, `weak_supervision_only`, or `eval_only`.

Non-negotiable rules:

1. Pool roles must be explicit.
2. Tier 2 and Tier 3 remain evaluation-only.
3. Tier 1 train-visible benchmark usage must be disclosed and contamination-audited.
4. Benchmark data must never dominate the full supervised mixture.
5. Synthetic data must reinforce mechanisms, not replace real mathematical distributions.
6. Multimodal document data must enter through the canonical parse artifact path.

### 7.1 Required Fields of `data_realization_policy/v1`

Every executable realization policy must declare at least:

| Field | Requirement |
| --- | --- |
| `data_realization_policy_id` | immutable policy identity |
| `profile_id` | bound profile family such as `P1`, `P2`, `P3`, or `P4` |
| `phase` | active `A-E` phase at which the policy is valid |
| `pool_allocations` | for each pool `A-E`: `token_weight`, `record_weight`, and allowed roles |
| `source_family_allowlists` | allowed source families per pool |
| `quality_gate_thresholds` | threshold bands for Pool B cleaning and Pool D admission |
| `tier1_global_cap` | maximum train-visible benchmark share expressed as both token and record caps |
| `weak_supervision_cap` | maximum aggregate `weak_supervision_only` share expressed as both token and record caps |
| `benchmark_family_policy_refs` | referenced `benchmark_family_policy/v1` objects |
| `decontam_policy_id` | referenced `decontam_policy/v1` object |

Rules:

1. `token_weight` and `record_weight` must both be declared.
2. Pool D-heavy realizations may not rely on token weights alone because long documents distort mixture accounting.
3. Threshold changes require a new policy id or explicit version bump.

### 7.2 Bootstrap Realization Matrix for `P1-P3`

To prevent first-generation `data_realization_policy/v1` objects from drifting, the following bootstrap realization matrix is normative for `P1-P3`.
These defaults may be changed only by publishing a new policy id and passing regression review.
Each pool cell is `token_weight / record_weight`.

| Profile | Pool A | Pool B | Pool C | Pool D | Pool E | Tier 1 Cap (`token / record`) | Weak-Supervision Cap (`token / record`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `P1` | `20 / 25` | `10 / 10` | `25 / 25` | `35 / 20` | `10 / 20` | `<= 5 / <= 5` | `<= 15 / <= 15` |
| `P2` | `15 / 20` | `15 / 15` | `25 / 25` | `30 / 20` | `15 / 20` | `<= 8 / <= 8` | `<= 20 / <= 18` |
| `P3` | `10 / 15` | `15 / 15` | `30 / 30` | `30 / 20` | `15 / 20` | `<= 8 / <= 8` | `<= 20 / <= 18` |

Rules:

1. Pool B remains `weak_supervision_only` by default in `P1-P3`.
2. Tier 1 benchmark material counts against the Tier 1 cap even when it enters as fragments or structural labels.
3. Weak-supervision cap is aggregate across Pool B and any other records explicitly labeled `weak_supervision_only`.
4. `P4` remains readiness-reviewed and is not precommitted here.

### 7.3 Bootstrap Source-Family and Role Defaults

Bootstrap `source_family_allowlists` for `P1-P3` must start from the following defaults.

| Profile | Pool D `core` allowlist | Pool D `auxiliary` allowlist | Pool D `eval_only` bootstrap default |
| --- | --- | --- | --- |
| `P1` | `PDF`, `DOCX` | `image`, `diagram` | `scanned_note` |
| `P2` | `PDF`, `DOCX`, `image` | `scanned_note`, `diagram` | none by default; low-confidence fallbacks stay `eval_only` |
| `P3` | `PDF`, `DOCX`, `image` | `scanned_note`, `diagram` | none by default; low-confidence fallbacks stay `eval_only` |

Rules:

1. Standalone `diagram` sources remain `auxiliary` across bootstrap `P1-P3`.
2. `scanned_note` is never bootstrap `core` in `P1-P3`; if admitted train-visible, it is `auxiliary` and must satisfy Section `2.4.2`.
3. Any profile-specific expansion beyond these allowlists requires an updated `data_realization_policy_id`.

### 7.4 Train-Visible Record Admission Contract

Every train-visible record must retain at least:

- `record_id`
- `pool_id`
- `pool_role`
- `data_realization_policy_id`
- `decontam_policy_id`
- `fingerprint_set`
- `source_family`
- `provenance_refs`
- `quality_flags`
- `source_record_lineage`
- `benchmark_family_id` and `benchmark_tier` when benchmark-derived
- `math_document_record_id` when document-derived
- `formalizer_provenance_id` when formalized

Required `fingerprint_set` keys:

- `source_fingerprint`
- `document_fingerprint`
- `semantic_unit_fingerprint`
- `theorem_or_problem_fingerprint`
- `solution_or_proof_fragment_fingerprint`
- `diagram_anchor_fingerprint`

Any train-visible record missing `pool_id`, `pool_role`, `data_realization_policy_id`, `decontam_policy_id`, `fingerprint_set`, or resolvable provenance is non-compliant and may not enter a run manifest.

---

## 8. Monitoring and Gate Inputs

Training and data pipelines must track at least:

- realized pool token ratios,
- realized pool record ratios,
- realized benchmark-tier ratios,
- parser provenance coverage,
- formalizer provenance coverage,
- verifier provenance coverage,
- contamination audit status,
- document parse quality,
- Pool B reject and downgrade rates,
- Pool D core-admit, auxiliary-fallback, and reject rates,
- weak-supervision share,
- missing-fingerprint rate,
- strict held-out leakage score.

Significant drift or missing provenance requires investigation and may block promotion.

---

## 9. Explicit Non-Goals

This document does not:

- fix exact model-size-specific hyperparameters,
- declare a single permanent token ratio recipe,
- authorize uncontrolled benchmark mixing,
- let OCR or parser outputs bypass normalization.

---

## 10. Weak Procedural Trace Quality Appendix

Pool B remains `weak_supervision_only` by default.
Weak procedural traces must therefore be cleaned and labeled as weak evidence rather than treated as gold proofs.

### 10.1 Acceptable Imperfections

The following are acceptable if the trace remains reconstructable:

- omitted algebraic micro-steps,
- pedagogical narration or stylistic boilerplate,
- informal subgoal narration,
- partial strategy sketches that do not pretend to be full proofs.

### 10.2 Required Cleaning

Weak procedural traces should be normalized to:

- separate final answer from intermediate reasoning,
- flag unresolved jumps or unsupported claims,
- strip duplicated template boilerplate,
- preserve uncertainty markers,
- retain provenance to the original source.

### 10.3 Reject Conditions

Reject traces that show any of the following:

- contradiction with the stated assumptions or source,
- answer-only supervision with no usable intermediate structure,
- hidden lemma dependence presented as if it were justified,
- severe template spam or near-duplicate solution farms,
- missing provenance or unresolved contamination risk.

Weak procedural traces may not be upgraded to proof-valid gold labels without stronger verifier or formal support.
