# Data Mixture and Ingestion

**Document Type:** Stability-Critical Spec (Canonical Binding)  
**Scope:** Training data mixture, ingestion constraints, QA gates, monitoring, and change control  
**Gate policy:** Mixture/ingestion changes require regression gating per `docs/06_Regression_and_Phase_Gates.md`  
**Source lineage:** Consolidated from a legacy data-mixture spec (removed on 2026-02-27).  
**Source consolidation note (2026-03-03):** Integrated the Pure LM dataset breakdown from `Pure LM (90% of total tokens).md` into Section 3 with no change to top-level mixture authority.

---

## 1. Scope

This document defines:

- The global training data mixture
- The internal composition of the Pure LM corpus
- The structure of IR-aligned synthetic data
- The role of benchmark datasets
- Data quality and ingestion constraints
- Change control requirements

This specification is a stability-critical training configuration.
Modifications require phase-gated review and regression validation.

---

## 2. Global Data Mixture

### 2.1 Top-Level Composition

| Category | Ratio | Description |
| --- | --- | --- |
| Pure LM (Primary Corpus) | 90% | Standard language-model pretraining corpus with controlled sub-mixture |
| IR-aligned Synthetic | 10% | Mechanism-aligned capability reinforcement |
| Benchmark Data | 0% | Regression probe only (not used for training) |

### 2.2 Non-Negotiable Boundary

Benchmark datasets (for example ARC-family) must not be included in training data and must not influence mixture composition.

---

## 3. Pure LM (90%)

### 3.1 Composition Overview

Pure LM remains fixed at 90% of total training tokens and is partitioned as:

| Pure LM Segment | Ratio of Total Training Tokens | Ratio Within Pure LM |
| --- | --- | --- |
| General and specialized text/code/math corpus | 80% | 88.89% |
| Document-extracted long-form corpus | 10% | 11.11% |

### 3.2 Internal Sub-Mixture Breakdown (Baseline)

All percentages below are ratios of total training tokens and must sum to 90%.

| Dataset | Ratio of Total Training Tokens | Ratio Within Pure LM | Segment | Primary Role |
| --- | --- | --- | --- | --- |
| `HuggingFaceFW/fineweb-edu` | 60% | 66.67% | General/specialized | General language capability backbone |
| `allenai/peS2o` | 10% | 11.11% | Document-extracted | Long-form, structured academic text |
| `bigcode/the-stack` | 8% | 8.89% | General/specialized | Source code dominant corpus |
| `open-web-math/open-web-math` | 4% | 4.44% | General/specialized | High math-symbol density text |
| `EleutherAI/proof-pile-2` (`algebraic-stack`) | 2% | 2.22% | General/specialized | Formal/math-code and CAS style data |
| `phanerozoic/Lean4-Mathlib` | 2% | 2.22% | General/specialized | Dependent-type and tactic-script text |
| `togethercomputer/RedPajama-Data-1T` (`arxiv`) | 2% | 2.22% | General/specialized | Type-theory and formal-derivation prose |
| `crumb/openstax-text` | 1% | 1.11% | General/specialized | Textbook procedural exposition |
| `togethercomputer/RedPajama-Data-1T` (`stackexchange`) | 1% | 1.11% | General/specialized | Practical rule-to-procedure explanations |

### 3.3 Corpus and Ingestion Constraints (Mandatory)

Allowed document extraction source formats (for document-derived ingestion) are:

- PDF
- HTML
- Word
- PPT

Ingestion rules:

- Only clean text in UTF-8 (`clean_text`) may enter the tokenizer.
- Metadata (source, extractor version, hash, provenance, dataset/subset identifiers) must remain external to token sequences.
- State IR schema must remain unchanged; no addition or expansion of State IR token types is permitted.
- Dataset identifier, subset/config, extracted fields, and sampling policy must be pinned and logged for every run.
- Licensing and attribution obligations for each dataset must be satisfied and auditable.
- `bigcode/the-stack-v2-dedup` must not be used as a direct content source for this mixture baseline.

### 3.4 Dataset-Specific Extraction and Filter Constraints

The following constraints are mandatory for the baseline defined in Section 3.2:

- `HuggingFaceFW/fineweb-edu`:
  - Extract `text` for training content.
  - Preserve `id`, `url`, and `dump` for provenance.
  - Optional resampling may use `token_count`, `score`, or `int_score`; if used, the policy must be pinned and logged.
- `allenai/peS2o`:
  - Extract `text`; preserve `id`, `source`, `created`, and `added`.
  - Restrict to `source="s2orc"` for full-text paper coverage.
- `bigcode/the-stack`:
  - Extract `content`; use `lang`, `ext`, `avg_line_length`, and `alphanum_fraction` for filtering.
  - Remove comments, docstrings, README-like files, and Markdown-heavy files from primary code content.
  - Use an explicit language allowlist (for example C/C++/Rust/Python/OCaml/Haskell/Java) and avoid high HTML/Markdown/TeX contamination.
- `open-web-math/open-web-math`:
  - Extract `text` and use `metadata`/`metadata.extraction_info.*` to retain high math-density content.
  - Exclude low-math conversational/forum-like narrative content.
- `EleutherAI/proof-pile-2` (`algebraic-stack`):
  - Extract `text`; preserve `meta`.
  - Prioritize high symbol-density/formal subsets (for example Python/Isabelle/Lean/Coq/Julia/TeX slices).
- `phanerozoic/Lean4-Mathlib`:
  - Use `fact` as primary training text and do not use `docstring` as primary content.
  - Restrict selected entries to formal definition/theorem-like `type` values.
  - Drop excessively short `fact` entries to reduce fragmentation.
- `togethercomputer/RedPajama-Data-1T` (`arxiv`):
  - Enforce `red_pajama_subset="arxiv"`.
  - Extract `text` and `meta`, and apply pattern filters aligned to formal logic/type-system material (for example `\\Gamma`, `\\vdash`, `\\lambda`, `\\Pi`, `\\Sigma`, and type-theory keywords).
- `crumb/openstax-text`:
  - Extract `text`.
  - Apply paragraph-level procedural filters (`Algorithm:`, `Procedure:`, `Step 1`, `Input/Output`, structured step lists, pseudo-code markers).
  - Preserve required CC BY attribution metadata.
- `togethercomputer/RedPajama-Data-1T` (`stackexchange`):
  - Enforce `red_pajama_subset="stackexchange"`.
  - Extract `text` and `meta`.
  - Restrict by site/domain allowlist (for example `cs.stackexchange.com`, `math.stackexchange.com`, `stackoverflow.com`) and preserve source metadata.

#### 3.4.1 Dataset References and Informative Notes (Non-normative)

This subsection is **informative** and does not add constraints beyond Sections 3.3 and 3.4.
It exists to provide dataset landing pages and operational context (motivation, common pitfalls, and provenance/licensing reminders).

- `HuggingFaceFW/fineweb-edu`:
  - Hugging Face: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  - Rationale (informative): education-filtered general text that stabilizes base language capability while keeping broad coverage.
  - Operational note (informative): for pilot-scale ingestion, FineWeb sample configs (`sample-10BT`, `sample-100BT`, `sample-350BT`) can be used before full runs.
- `allenai/peS2o`:
  - Hugging Face: [allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o)
  - Rationale (informative): long-form academic-style prose with sectioning and mixed symbol/text.
  - Operational note (informative): `text` often contains paragraph breaks (`\n\n`); keep paragraph boundaries stable to reduce fragmentation artifacts.
- `bigcode/the-stack`:
  - Hugging Face: [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack)
  - Rationale (informative): high-density source code that pushes non-natural-language token patterns and syntax regularities.
  - Operational note (informative): The Stack is a multi-license corpus; keep provenance and licensing auditable per Section 3.3.
- `bigcode/the-stack-v2-dedup` (not used as a direct content source in this baseline):
  - Hugging Face: [bigcode/the-stack-v2-dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)
  - Operational note (informative): this dataset primarily provides file identifiers; bulk content retrieval is typically via the Software Heritage archive/object storage path and may require additional access steps/agreements.
- `open-web-math/open-web-math`:
  - Hugging Face: [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)
  - Rationale (informative): math-heavy text with LaTeX/MathJax-style markup that increases symbol density and derivation-like patterns.
  - Operational note (informative): lean on `metadata.extraction_info.*` (math-signal fields) to avoid drifting toward narrative/forum content.
- `EleutherAI/proof-pile-2` (`algebraic-stack`):
  - Hugging Face: [EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
  - Rationale (informative): formal math, CAS-like text, and math-code mixtures that complement open-web-math with more rule/grammar regularity.
  - Operational note (informative): within `algebraic-stack`, the higher signal-to-noise slices are often formal languages and math-code (for example Lean/Coq/Isabelle, Python/Julia, TeX).
- `phanerozoic/Lean4-Mathlib`:
  - Hugging Face: [phanerozoic/Lean4-Mathlib](https://huggingface.co/datasets/phanerozoic/Lean4-Mathlib)
  - Rationale (informative): dependent-type and tactic-script text with strong syntactic constraints and proof-structured patterns.
  - Operational note (informative): `docstring` is natural-language documentation; `fact` is typically the formal payload.
- `togethercomputer/RedPajama-Data-1T` (`arxiv`):
  - Hugging Face: [togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
  - Rationale (informative): type-theory and formal-derivation prose mixed with symbols; complements Lean/proof corpora with explanatory transitions.
  - Operational note (informative): the arXiv slice is sourced from LaTeX; common preprocessing removes preamble/comments/macros/bibliographies, which tends to concentrate on derivation content rather than document scaffolding.
- `crumb/openstax-text`:
  - Hugging Face: [crumb/openstax-text](https://huggingface.co/datasets/crumb/openstax-text)
  - Rationale (informative): textbook-style procedural exposition ("rule -> procedure -> outcome") useful for step-structured language.
  - Operational note (informative): OpenStax content is CC BY; keep attribution metadata intact and auditable.
- `togethercomputer/RedPajama-Data-1T` (`stackexchange`):
  - Hugging Face: [togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
  - Rationale (informative): practical, explanation-first content that often includes algorithms and stepwise reasoning.
  - Operational note (informative): this slice is typically HTML-stripped, organized into Q/A pairs, and sorted by a quality/score signal; site allowlisting helps avoid low-signal conversational spillover.

### 3.5 Document-Extracted Text Policy

- Baseline allocation is 10% of total training tokens (11.11% of Pure LM) under Section 3.2.
- Upper bound is 15% of Pure LM tokens (13.5% of total training tokens), subject to stability review and regression gating.
- Inclusion requires passing the Data QA Gate (Section 3.6).
- Extractor version updates are treated as distributional shifts and require regression validation before promotion.

### 3.6 Data QA Gate (Mandatory)

Document-derived text must satisfy all of the following:

1. Control/non-printable character ratio <= 2%
2. Repetition rate <= 20% (template/header/footer contamination filter)
3. No severe fragmentation (for example average line length < 20 characters with excessive line breaks)
4. Language distribution consistent with expected corpus distribution
5. Extractor version pinned and logged

Documents failing any criterion must be excluded from the primary corpus.
Dataset-specific filters in Section 3.4 are additional constraints and do not replace this gate.

### 3.7 Scaling and Ramp Notes

- For pilot-scale validation, FineWeb sample configs (`sample-10BT`, `sample-100BT`, `sample-350BT`) may be used before full-scale ingestion runs.
- Any scale-up, resampling, or field/filter adjustment must preserve Section 2 top-level ratios and pass required regression gates.

---

## 4. IR-Aligned Synthetic (10%)

### 4.1 Objective

Synthetic data is used to reinforce core architectural mechanisms:

- Credit assignment
- Learned routing/gating
- Failure recovery
- Stable state evolution

Synthetic data must not become the dominant optimization objective.

### 4.2 Composition

| Category | Ratio | Target Mechanism |
| --- | --- | --- |
| Multi-step Credit Tasks | 3% | Delayed reward/credit routing |
| Routing/Gating Tasks | 3% | Learned control flow |
| Failure Recovery Tasks | 2% | Error detection and correction |
| Structured World Modeling | 2% | Stable state-update dynamics |

### 4.3 Synthetic Constraints

- Must not introduce new State IR token categories.
- Must not enforce fixed reasoning templates.
- Must remain <= 10% of total training tokens.
- ARC-family datasets are strictly regression probes, not synthetic sources.

---

## 5. Benchmark Policy

Benchmark datasets:

- Are not included in training
- Serve exclusively as regression probes
- Must not influence mixture ratios
- Must not shape synthetic generation strategies

---

## 6. Monitoring Requirements

The training pipeline must track:

1. Loss and perplexity sliced by data source
2. Character distribution drift
3. Language distribution drift
4. Document-extracted proportion over time
5. Synthetic task performance distribution
6. Pure LM sub-mixture realization versus Section 3.2 targets
7. Dataset-level provenance and license-compliance coverage

Significant drift requires investigation and potential gating.

---

## 7. Change Control

The following changes require regression gating:

- Adjustment of top-level mixture ratios
- Any change to the Section 3.2 Pure LM sub-mixture breakdown
- Document corpus proportion changes
- Dataset substitution, subset/config changes, or extraction-field changes
- Extractor version updates
- Dataset filter-rule changes (including sampling/reweighting policy)
- Synthetic composition changes
- Addition of new synthetic categories
- Inclusion of benchmark data in training

---

## 8. Final Mixture Summary

```text
Pure LM (90% total):
- 60% HuggingFaceFW/fineweb-edu
- 10% allenai/peS2o
- 8% bigcode/the-stack
- 4% open-web-math/open-web-math
- 2% EleutherAI/proof-pile-2 (algebraic-stack)
- 2% phanerozoic/Lean4-Mathlib
- 2% RedPajama-Data-1T (arxiv)
- 1% crumb/openstax-text
- 1% RedPajama-Data-1T (stackexchange)

IR-aligned Synthetic (10% total): 3/3/2/2
Benchmark datasets: 0% in training (regression only)
State IR schema drift: forbidden
Synthetic share > 10%: forbidden
```

This specification establishes a stable, pretraining-first data regime aligned with architectural invariants and long-run training stability.
