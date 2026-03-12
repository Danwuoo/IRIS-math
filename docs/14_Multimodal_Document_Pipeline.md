# Multimodal Document Pipeline

**Document Type:** Active Companion Authority  
**Scope:** Document-native ingestion, normalization, parser boundaries, canonical anchors, and State IR projection for IRIS-math v2  
**Boundary:** This document elaborates the multimodal document path. It does not freeze a single global anchor serialization scheme or replace the canonical State IR contract.

---

## 0. Purpose and Positioning

IRIS-math v2 treats mathematical documents as first-class inputs.

This document defines the end-to-end contract from raw source material to State IR-aligned reasoning inputs:

- source families,
- ingestion objects,
- normalization stages,
- parser responsibilities,
- anchor requirements,
- failure mapping.

It complements:

- `docs/07_Data_Constitution.md` for canonical artifact and provenance policy,
- `docs/02_State_IR_Spec.md` for the seven-slot State IR target,
- `docs/03_Level_Contracts_L0-L6.md` for level responsibilities.

---

## 1. Supported Source Families

The active document-native source families are:

- `PDF`
- `DOCX`
- `image`
- `scanned_note`
- `diagram`

Rules:

1. `DOCX` is a first-class source, not a convenience conversion path.
2. Images and scanned notes must be normalized through the same canonical pipeline, not bypassed into ad hoc OCR text.
3. Standalone diagrams are allowed when they carry mathematical meaning or are attached to theorem/problem context.

---

## 2. Ingestion Object Family

The document pipeline uses three object layers.

| Object | Purpose | Ownership |
| --- | --- | --- |
| `math_document_source/v1` | Raw source identity, modality, capture metadata, and provenance envelope | ingestion layer |
| `math_document_record/v1` | Canonical normalized document artifact | normalization layer |
| `math_document_projection/v1` | State IR-oriented projection pack or supervision pack derived from the record | projection layer |

`math_document_record/v1` is the canonical artifact referenced by `docs/07_Data_Constitution.md`.

`math_document_projection/v1` may vary by task, but it must remain traceable back to the canonical record and its anchors.

---

## 3. Normalization Stages

### 3.1 Stage A: Source Capture

Capture:

- source format,
- snapshot identity,
- source hash,
- modality,
- parser eligibility,
- license / attribution state.

### 3.2 Stage B: Modality-Specific Extraction

Extraction responsibilities by source type:

- `PDF`: page segmentation, text extraction, formula region detection, layout ordering
- `DOCX`: paragraph/run structure, equation objects, section hierarchy, table extraction, embedded figure references
- `image` / `scanned_note`: OCR, layout detection, handwriting-aware segmentation when available, image region cropping
- `diagram`: object regions, labels, adjacency or incidence candidates, caption association when available

### 3.3 Stage C: Mathematical Structure Typing

Normalize mathematical structure into:

- text blocks,
- formula blocks,
- table blocks,
- diagram regions,
- section hierarchy,
- semantic units such as `definition`, `theorem`, `lemma`, `proof`, `example`, `exercise`, `remark`, `figure`.

### 3.4 Stage D: Anchor Canonicalization

Assign canonical anchors to all content that may later be referenced by:

- State IR slots,
- retrievers,
- verifiers,
- training traces,
- evaluation artifacts.

### 3.5 Stage E: State IR Projection

Project normalized content into task-specific State IR-aligned views without bypassing the seven-slot contract.

### 3.6 Semantic Unit Cut and Alignment Rules

Semantic units should be cut at the smallest stable mathematical boundary that remains useful for later reasoning:

- `definition`, `theorem`, `lemma`, `proposition`, `corollary`, `proof`, `example`, `exercise`, `remark`
- proof units may contain nested claim spans, but the parent `proof` boundary remains the canonical unit for fingerprinting and provenance
- if a formula block is embedded inside a text block, the canonical record must retain both the clean-text block and the aligned formula block rather than flattening one into the other

Formula/text alignment minimum:

- every formula block used downstream must retain the parent text-block id or anchor id when one exists
- every semantic unit that contains displayed formulas must retain the ordered formula-block refs that belong to that unit
- if alignment confidence is low, the record must retain the uncertainty rather than silently emitting a clean merged paragraph

---

## 4. Parser Responsibility Boundaries

### 4.1 Layout Parser

Responsible for:

- reading order,
- block segmentation,
- section hierarchy,
- table boundaries,
- figure / caption association.

Not responsible for:

- solving the problem,
- inventing semantic relations not supported by structure,
- assigning proof validity.

### 4.2 Formula Parser

Responsible for:

- detecting formula spans,
- normalizing equation or expression structure,
- preserving symbol spans and source anchors,
- exposing parse uncertainty.

Not responsible for:

- choosing theorem strategy,
- silently repairing unsupported mathematical meaning.

### 4.3 Diagram Parser

Responsible for:

- diagram region detection,
- label extraction,
- candidate geometric or graph-like relations,
- provenance and confidence emission.

Not responsible for:

- claiming theorem truth from visual heuristics alone,
- bypassing later structuring or verification.

---

## 5. Canonical Anchors

Every anchorable item must expose at least:

- `anchor_id`
- `source_id`
- `source_format`
- `page_or_canvas_id`
- `modality`
- `span_or_bbox`
- `structural_role`
- `parent_anchor_id` when nested
- `confidence`
- `parser_provenance_id`

Anchor rules:

1. Anchors must be stable enough for replay and audit within a fixed canonical record.
2. Anchors must support both text-like spans and region-like boxes.
3. This document does **not** freeze one universal global id packing scheme across all modalities.

### 5.1 Canonical Anchor Serialization Minimum

The canonical record should serialize anchor-bearing layout fields explicitly enough to survive replay:

- `page_or_canvas_id`
- `reading_order_index`
- `bbox` as `(x0, y0, x1, y1)` in source-local coordinates when a region exists
- `span_start` / `span_end` when a text span exists
- `parent_anchor_id` when an anchor is nested under a semantic unit, figure, or block

`reading_order` and `anchor_index` must therefore be replayable without reconstructing layout from raw parser traces.

---

## 6. Projection Into State IR

Normalized artifacts project into the seven State IR groups as follows.

| State IR Slot | Typical Document Contributions |
| --- | --- |
| `PF` | task type, target statement, output form, source anchors, domain clues |
| `SY` | symbols, scopes, definitions, typed entities from text/formulas |
| `CG` | equalities, inequalities, dependencies, incidences, recurrence links, diagram-derived candidate relations |
| `FR` | subgoals, proof obligations, candidate strategy families, branch context |
| `LM` | cited lemmas, retrieved local definitions, candidate theorem matches with applicability notes |
| `VS` | parse-linked validity warnings, gap signals, contradiction or counterexample evidence references |
| `CS` | reparse / continue / backtrack / switch behavior tied to document uncertainty or verifier outcomes |

Projection rules:

1. Projection must preserve anchor traceability.
2. Projection may summarize parser outputs, but raw unmanaged parser traces must not become semantic state.
3. Diagram-derived relations remain candidate structure until later levels or verifiers strengthen them.

### 6.2 Minimum Projection Payload by Slot

Projection into a State IR slot is compliant only when the projection path can justify the slot's minimal semantic payload.

Minimum expectations:

- `PF`: emit task framing, target statement or target anchor neighborhood, and source anchors; if task type or output form is unresolved, keep that uncertainty explicit rather than silently fabricating a precise task contract
- `SY`: emit surface form, scope clue, binding state, and anchor-backed symbol provenance; unresolved symbols must remain explicit instead of being dropped
- `CG`: emit relation type, ordered arguments or anchor-backed placeholders, relation status, and source support; document- or diagram-derived relations default to `candidate` unless stronger support exists
- `FR`: seed only branch / subgoal / obligation structure that is actually supported by theorem, proof, exercise, or derivation units; sparse `FR` is preferable to invented proof search state
- `LM`: seed only cited local lemmas, local definitions, or retrieval candidates that carry provenance and at least an `unchecked` applicability audit
- `VS`: seed only provenance-bearing parser or verifier warnings as local-validity or gap-style evidence; parse confidence alone is not proof-validity evidence
- `CS`: seed only document-conditioned control intent such as `continue`, `reparse`, or escalation due to unresolved regions; parser heuristics must not masquerade as full solver policy

### 6.3 Projection Freeze Rules

Each task-specific projection path must declare which slots it is authoritative enough to seed.

Rules:

1. Slots that are not authoritatively seeded must remain empty, sparse, or explicitly draft-like rather than being hallucinated into completeness.
2. Cross-page or cross-anchor relationships may not be collapsed into a clean single-block summary if that would destroy anchor provenance.
3. Candidate and unresolved status must be preserved through projection whenever parser confidence or semantic support is incomplete.
4. Projection-time convenience transforms must not erase the distinction between source-grounded structure and later verifier-upgraded structure.

### 6.1 Diagram Candidate Promotion Rule

Diagram relations enter the canonical record first as candidate relations.

They may be promoted into `CG` only when at least one of the following holds:

- the relation is supported by explicit labels or captions tied to the same anchor neighborhood,
- the relation is cross-supported by nearby clean text or formula blocks,
- a later verifier or structured consistency check upgrades the candidate with explicit provenance.

Otherwise the relation remains a candidate and any downstream misuse should be attributable as `F_REP`.

---

## 7. Failure Mapping

Multimodal document failures must be mapped into the canonical taxonomy.

Primary mappings:

| Failure Mode | Primary Code | Typical Locus |
| --- | --- | --- |
| OCR/layout corruption, broken reading order, missing formula region, bad DOCX structure import | `F_REP` | `L0/L1` |
| Source anchors lost or mismatched during projection | `F_REP` | `L0/L1` |
| Diagram relation overreach or unsupported structural invention | `F_REP` | `L0/L1` |
| Verifier accepting unsupported document-grounded claims | `F_EVAL` | `L6` |
| Verifier rejecting valid document-grounded claims due to weak evidence integration | `F_EVAL` | `L6` |

Secondary effects may propagate to `F_PROC`, `F_SEARCH`, or `F_MEM`, but document-pipeline breakage must not be hidden under downstream labels when `L0/L1` caused it.

---

## 8. Explicit Non-Goals

This document does not:

- define the full global anchor id serialization scheme,
- authorize raw OCR or parser outputs to bypass State IR,
- declare a lemma-memory constitution,
- freeze one parser backend family,
- convert document parsing into a hard-coded solver core.

---

## 9. Related Documents

- `docs/07_Data_Constitution.md`
- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/16_Verifier_and_Formalization_Stack.md`
