# IRIS-math v2 Charter

**Document Type:** Design Note (Non-normative, Active Direction)  
**Purpose:** Active project charter for the IRIS-math transition  
**Non-Override Clause:** This charter sets project direction and priorities. It
does not by itself approve conflicting architecture, data, parser, or
evaluation changes.

---

## 1. Project Direction

IRIS-math is the active direction for this repository.

Primary target:

- Document-native mathematical reasoning
- Math-native multimodal input handling
- Verifier-centered proof and search
- Benchmark-aware, but not benchmark-locked, development
- Staged scaling from `3B -> 7B -> 14B -> 30B -> 70B -> 120B`

This repo is no longer optimizing for the baseline IRIS objective alone.

---

## 2. Transition Rule

The repo is moving from a historical baseline contract stack toward an
IRIS-math-specific stack.

Rules:

1. Do not silently force the historical baseline when it conflicts with the
   active IRIS-math direction.
2. Do not treat AGENTS, this charter, or any design note as sufficient approval
   for a conflicting change.
3. For any conflicting surface, first consult the relevant transition proposal
   or approved transition spec.
4. If only a proposal exists, update the proposal and stop short of the
   conflicting implementation.

---

## 3. Product Priorities

IRIS-math work should prioritize:

- Robust document ingestion and provenance
- Canonical parse sidecars instead of ad-hoc parser paths
- Verifier-grounded search and proof workflows
- Benchmark tiering, contamination control, and held-out evaluation discipline
- Explicit hardware routing instead of single-profile assumptions

The repo should not prioritize:

- A baseline skeleton as the primary product target
- Benchmark score chasing without attribution or contamination control
- Single-H100-only planning as the default long-range assumption

---

## 4. Approved Surface

The following surfaces are approved now:

- Rewriting control-plane docs to reflect the IRIS-math direction
- Creating transition proposals and migration notes
- Performing compatibility analysis against the historical baseline
- Planning hardware profiles and scaling paths
- Building non-conflicting tooling or documentation that does not depend on an
  unapproved architecture, data, parser, or benchmark-policy change

---

## 5. Blocked Surface

The following surfaces remain blocked until the relevant transition spec is
approved:

- Architecture changes that conflict with the historical baseline contracts
- Data-mixture changes that conflict with the historical baseline data policy
- Benchmark training inclusion without an approved tiering and decontamination
  plan
- Parser or document-format changes that create new trunk-facing interfaces
  without an approved canonical format
- Claims that a proposed IRIS-math direction is already approved simply because
  it appears in AGENTS or a design note

---

## 6. Required Migration Artifacts

IRIS-math transition work is expected to flow through these artifacts:

- `docs/14_IRIS_Math_Data_Constitution_v2.md`
- `docs/15_Benchmark_Training_and_Eval_Tiering.md`
- `docs/16_Document_Math_Parse_Canonical_Format.md`
- Approval records or follow-on approved transition specs for each conflicting
  surface

---

## 7. Completion Rule

Every IRIS-math change should end in one of:

- `Done`
- `Blocked`
- `Cancelled`

No open-ended transition work should be left implicit.
