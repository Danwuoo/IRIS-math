# IRIS-math Data Constitution v2

**Document Type:** Change Proposal (Requires Approval)  
**Purpose:** Proposed data constitution for the IRIS-math transition  
**Non-Override Clause:** This proposal does not approve conflicting data,
training, or architecture changes until it is explicitly approved.

---

## 1. Scope

This proposal covers:

- Math-native and document-native data direction
- Provenance and sidecar requirements
- Contamination boundaries
- Migration away from the historical baseline data assumptions where necessary

---

## 2. Proposed Direction

IRIS-math data policy should move toward:

- Document-native mathematical corpora with auditable provenance
- Native handling of document structure, formulas, theorem/proof text, and
  derived sidecars
- Explicit separation between raw extracted content, normalized text, and
  parser-produced sidecars
- Benchmark-aware data governance with contamination accounting
- Scaling-friendly mixtures that can support the `3B -> 120B` path

---

## 3. Approved Surface

The following work is approved now:

- Dataset inventory and provenance auditing
- Contamination-risk analysis and documentation
- Proposal work for new mixtures, sidecars, and ingestion policies
- Non-conflicting tooling that records metadata without changing active training
  semantics
- Migration notes comparing the proposed direction to the historical baseline

---

## 4. Blocked Surface

The following work is blocked until approval:

- Changing the active global data mixture
- Declaring benchmark data approved for training
- Introducing new parser-derived training channels that bypass the canonical
  format process
- Treating parser outputs as a direct replacement for approved model-facing
  interfaces
- Claiming that a proposed mixture is active without an approval record

---

## 5. Required Migration Artifacts

This proposal expects the following artifacts before approval:

- A benchmark tiering and decontamination plan from
  `docs/15_Benchmark_Training_and_Eval_Tiering.md`
- A canonical parser sidecar proposal from
  `docs/16_Document_Math_Parse_Canonical_Format.md`
- Dataset provenance and contamination reporting templates
- An approval record identifying the approved and blocked surfaces

---

## 6. Open Questions

Label unresolved items as **不確定**, especially for:

- Final top-level mixture ratios
- Benchmark-adjacent training allowances
- Parser-sidecar storage format and retention policy
- Promotion criteria from proposal to approved spec
