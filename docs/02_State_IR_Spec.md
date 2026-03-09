# State IR Spec

**Document Type:** Canonical Specification (Normative)  
**Effective date:** 2026-03-10  
**Version:** 2.0  
**Authority:** This document defines the active IRIS-math v2 State IR.

---

## 1. Purpose and Transition Scope

State IR is the canonical internal work-state for IRIS-math v2.

It exists to represent mathematical reasoning over:

- text,
- documents,
- formulas,
- diagrams,
- formal or semi-formal artifacts,
- verifier feedback.

The repository is in a documentation-first transition.
During this period, baseline implementation may still expose legacy scaffolding.
Such scaffolding is transitional and does not replace the v2 contract defined here.

---

## 2. Core Principles

1. **Single Canonical Work State**  
   All semantic reasoning state must map into this State IR.

2. **Math-Native Semantics**  
   State IR represents mathematical work structure, not generic conversation state.

3. **Verifier Visibility**  
   Verification and proof-validity signals are first-class parts of state.

4. **Control Visibility**  
   Strategy switching, backtracking, stopping, and budget state must be representable.

5. **Cross-Modal Grounding**  
   Text, document layout, formulas, and diagrams must normalize into one coherent state contract.

6. **Versioned Evolution**  
   Slot inventory, ordering, or mandatory semantics cannot drift silently.

---

## 3. Canonical Slot System

State IR v2 uses seven fixed-order slot groups.

| Slot | Symbol | Cardinality | Purpose |
| --- | --- | --- | --- |
| Problem Frame | `PF` | exactly 1 | Task type, output form, assumptions, domain, source anchors |
| Symbol Table | `SY` | `0..N_sy` | Variables, constants, functions, sets, types, scopes |
| Constraint Graph | `CG` | `0..N_cg` | Equalities, inequalities, dependencies, incidences, recurrences |
| Proof / Program Frontier | `FR` | `0..N_fr` | Hypotheses, subgoals, candidate strategies, branches, obligations |
| Memory / Lemma Interface | `LM` | `0..N_lm` | Retrieved lemmas, match conditions, applicability audit, mismatch notes |
| Verifier State | `VS` | `0..N_vs` | Local validity, gap risk, counterexample risk, consistency state |
| Control State | `CS` | exactly 1 | Continue/backtrack/reparse/switch/stop intent, budget, escalation |

No additional top-level slot groups are permitted.

---

## 4. Canonical Sequence Construction

The fixed slot order is:

```text
Z = [ PF ; SY_1...SY_n ; CG_1...CG_m ; FR_1...FR_k ; LM_1...LM_p ; VS_1...VS_q ; CS ]
```

Rules:

- `PF` and `CS` must always be present.
- `SY`, `CG`, `FR`, `LM`, and `VS` may be empty.
- Relative ordering within each slot group must remain stable within a reasoning cycle.
- Reordering inside a slot group is a state mutation and must be explicitly produced.

---

## 5. Slot Semantics

### 5.1 Problem Frame (`PF`)

`PF` represents:

- task type (`prove`, `compute`, `construct`, `decide`, `formalize`, `find_counterexample`, ...),
- required output form,
- assumptions currently in scope,
- mathematical domain,
- target object or target statement,
- source anchors back to document/formula/diagram regions when available.

### 5.2 Symbol Table (`SY`)

`SY` stores:

- symbols,
- domains / types,
- scope boundaries,
- bindings introduced by definitions or assumptions,
- unresolved references.

### 5.3 Constraint Graph (`CG`)

`CG` stores:

- equalities,
- inequalities,
- divisibility or modular relations,
- implication or dependency edges,
- geometric incidences,
- recurrence or transformation relations.

### 5.4 Proof / Program Frontier (`FR`)

`FR` stores the active working frontier:

- current hypotheses,
- subgoals,
- candidate strategy families,
- active branch,
- abandoned branch summaries,
- unresolved obligations.

### 5.5 Memory / Lemma Interface (`LM`)

`LM` stores:

- retrieved lemmas, patterns, or examples,
- match conditions,
- applicability confidence,
- mismatch notes,
- reasons a retrieved item is not yet safe to use.

Similarity alone is insufficient; applicability audit is required.

### 5.6 Verifier State (`VS`)

`VS` stores:

- local step-validity signals,
- hidden-assumption risk,
- proof-gap detection,
- counterexample risk,
- branch consistency summaries.

### 5.7 Control State (`CS`)

`CS` stores:

- next-action mode (`continue`, `backtrack`, `reparse`, `switch_strategy`, `stop`, ...),
- remaining budget,
- uncertainty / escalation state,
- recovery mode state when targeted repair is active.

---

## 6. Cross-Modal Anchoring and External Artifact Normalization

Document parsers, OCR systems, formalizers, verifiers, and external tools do not write raw outputs directly into the trunk.

They must first be normalized into:

- canonical parse artifacts governed by `docs/07_Data_Constitution.md`,
- State IR slot content governed by this document.

State IR may retain **anchors** to source regions or provenance ids, but not raw unmanaged tool traces.

---

## 7. Mutability Rules

1. New slot entries may be created only by learned or contract-governed modules.
2. Deletion or consolidation must be explicit.
3. Control and verifier state are part of the same work-state contract, not out-of-band metadata.
4. Transition adapters that project baseline code into v2 slots are temporary and must be labeled as such.

---

## 8. Cross-Level Contract

All levels `L0-L6`:

- must consume this State IR or a schema-faithful slice of it,
- must produce updated State IR or scalar signals derived from it,
- must not invent alternative semantic state channels,
- must not bypass the slot ordering defined above.

Program traces or symbolic artifacts may exist outside State IR as auxiliary objects, but any behavior-affecting summary must be represented back into `FR`, `LM`, `VS`, or `CS`.

---

## 9. Explicit Non-Goals

State IR is not:

- a raw OCR dump,
- a raw tool log,
- a benchmark-specific action schema,
- a human-readable proof script requirement,
- a generic assistant conversation buffer.

---

## 10. Versioning and Compliance

- This specification is versioned.
- Any change to the seven slot groups, their order, or their mandatory semantics requires a major revision.
- Silent fallback to baseline `T/G/O/R/X/M` semantics is not compliant with the active target, even if temporary adapters still exist in code.

