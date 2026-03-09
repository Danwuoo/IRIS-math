# Document Math Parse Canonical Format

**Document Type:** Change Proposal (Requires Approval)  
**Purpose:** Proposed canonical sidecar and schema discipline for document-math
parsing in the IRIS-math transition  
**Non-Override Clause:** This proposal does not approve new model-facing parse
interfaces until explicitly approved.

---

## 1. Scope

This proposal covers:

- Canonical sidecar requirements for document and math parsing
- Separation between raw parser outputs, normalized text, and model-facing data
- Versioning and migration expectations for parse schemas
- Constraints that prevent parser-specific bypasses into the trunk

---

## 2. Proposed Direction

IRIS-math should treat document and math parse products as canonical sidecars,
not as ad-hoc latent channels.

The canonical format should eventually support:

- Text spans and paragraph structure
- Formula spans and math markup references
- Layout regions and reading order
- Figures, tables, captions, and references
- Theorem, lemma, proof, and derivation-style structure
- Provenance, hashes, extractor versions, and parser versions

---

## 3. Approved Surface

The following work is approved now:

- Designing the sidecar schema and versioning plan
- Building parser adapters that emit sidecars outside the model-facing path
- Comparing parser outputs and documenting field candidates
- Auditing existing ingestion paths for hidden parser bypasses
- Writing migration notes for any future model-facing integration

---

## 4. Blocked Surface

The following work is blocked until approval:

- Feeding raw parser outputs or parser-specific tensors directly into the trunk
- Silently adding new State IR token categories or changing canonical ordering
- Treating parser-specific features as approved model interfaces without a
  canonical sidecar
- Coupling parse schema changes to benchmark-specific logic without explicit
  approval

---

## 5. Required Migration Artifacts

This proposal expects the following artifacts before approval:

- A versioned sidecar schema draft with field definitions
- Provenance requirements for extractor and parser versions
- Mapping notes to the data constitution proposal in
  `docs/14_IRIS_Math_Data_Constitution_v2.md`
- Any benchmark or held-out implications documented in
  `docs/15_Benchmark_Training_and_Eval_Tiering.md`
- Approval record identifying the approved model-facing surface, if any

---

## 6. Open Questions

Label unresolved items as **不確定**, especially for:

- Which fields are mandatory in v1 of the sidecar
- Whether theorem/proof structures need separate normalized views
- How much parser disagreement metadata should be retained
- Promotion rules from proposal to approved transition spec
