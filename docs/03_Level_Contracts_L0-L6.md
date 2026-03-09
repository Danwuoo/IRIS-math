# Level Contracts L0-L6

**Document Type:** Normative Contract  
**Effective date:** 2026-03-10  
**Authority:** These contracts define the active v2 responsibilities and interface expectations for external levels `L0-L6`.

---

## 1. Common Interface Contract

The external numbering remains `L0-L6` during the first-round v2 migration.
The responsibility semantics are math-specific.

### Inputs

- `state_in`: canonical State IR or a schema-faithful slice
- `context_in`: optional external context
- `control_in`: optional upstream control signals
- `resource_budget`: optional time / memory / step limits

### Outputs

- `state_out`: updated State IR
- `control_out`: learned downstream suggestions or neutral control
- `diagnostics`: level-observable signals for attribution, calibration, and regression

### Stub Behavior

When disabled:

- the interface must still exist,
- `state_out` must preserve schema compatibility,
- `control_out` must be neutral,
- `diagnostics` must emit a disabled marker plus minimal summaries.

### Observability Requirement

Each level must expose diagnostics sufficient for:

- failure attribution,
- calibration,
- regression comparison,
- stub-vs-mounted distinction.

---

## 2. Level Responsibilities

### 2.1 L0 — Representation

Responsibilities:

- parse text, formulas, layouts, and diagrams into math-relevant observations,
- preserve source anchors needed for later grounding,
- expose uncertainty about parsing or grounding.

Typical diagnostics:

- parse confidence,
- OCR/layout uncertainty,
- modality coverage,
- diagram grounding confidence.

Prohibited patterns:

- hard-coding task solutions into parsing logic,
- bypassing State IR with raw parser outputs.

### 2.2 L1 — Structuring

Responsibilities:

- construct or update the symbol table,
- build constraint structure,
- maintain assumption scope and local consistency,
- convert observations into structured mathematical state.

Typical diagnostics:

- symbol binding quality,
- scope conflict rate,
- constraint coverage / conflict signals.

Prohibited patterns:

- hiding representation errors by inventing unsupported structure,
- baking benchmark-specific schema assumptions into the contract.

### 2.3 L2 — Strategy Induction

Responsibilities:

- propose candidate strategy families,
- expand the proof / program frontier with learned intermediate structure,
- preserve diversity among plausible approaches.

Typical diagnostics:

- strategy diversity,
- frontier quality,
- candidate collapse indicators.

Prohibited patterns:

- turning L2 into a handwritten symbolic executor,
- forcing a single deterministic strategy path by default.

### 2.4 L3 — Search Control and Recovery Policy

Responsibilities:

- allocate budget across branches,
- select expansion / backtrack / switch behavior,
- perform targeted recovery and repair policy in the first-round v2 design,
- decide whether more effort is warranted.

Typical diagnostics:

- budget pressure,
- backtrack rate,
- recovery precision,
- termination margin.

Prohibited patterns:

- fixed search schedules as default semantics,
- blind global retry loops.

### 2.5 L4 — Memory / Retrieval Binding

Responsibilities:

- retrieve lemmas, patterns, examples, or prior state,
- audit applicability conditions,
- reject tempting but unsafe retrievals.

Typical diagnostics:

- retrieval similarity,
- applicability precision,
- mismatch rejection rate,
- write / reuse summaries.

Prohibited patterns:

- similarity-only retrieval without applicability audit,
- memory becoming a second semantic trunk.

### 2.6 L5 — Abstraction / Compression

Responsibilities:

- compress repeated local structure into reusable abstractions,
- propose invariants, macros, or lemma-like summaries,
- preserve important distinctions while abstracting.

Typical diagnostics:

- abstraction granularity,
- invariant reuse,
- downstream override rate.

Prohibited patterns:

- abstractions that erase essential proof conditions,
- benchmark-shaped abstraction templates.

### 2.7 L6 — Verification and Learning-Signal Routing

Responsibilities:

- verify local and global validity,
- estimate proof-gap and counterexample risk,
- emit calibrated confidence,
- output failure taxonomy and level credit routing,
- package learning-signal routing in the first-round v2 design.

Typical diagnostics:

- false accept / false reject rates,
- calibration error,
- disagreement,
- failure-credit distribution.

Prohibited patterns:

- treating the verifier as a fixed acceptance script,
- collapsing credit routing to a single hard blame label.

---

## 3. Compliance Checklist

1. Does every level keep its interface when disabled?
2. Are diagnostics sufficient for attribution and regression?
3. Does any level bypass State IR?
4. Has deterministic control replaced learned policy?
5. Has L3 absorbed recovery policy without deleting the external `L0-L6` numbering?
6. Has L6 retained verification plus credit routing without inventing new external levels?
