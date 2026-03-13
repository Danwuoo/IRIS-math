# Level Contracts L0-L6

**Document Type:** Normative Contract  
**Effective date:** 2026-03-10  
**Authority:** These contracts define the active v2 responsibilities and interface expectations for external levels `L0-L6`.

---

## 1. Common Interface Contract

The external numbering remains `L0-L6` during the first-round v2 migration.
The responsibility semantics are math-specific.

### Internal Module Mounting Rule

`L7 Recovery / Repair` and `L8 Learning Signal Router` are not active external levels in v2.
If these labels survive in code or design discussion, they may exist only as internal implementation names mounted under `L3` or `L6`.

Rules:

- internal modules may be separate code modules, parameter heads, or fused subgraphs,
- internal modules must not introduce new external level ids, new top-level failure codes, or independent State IR bypass paths,
- compliance, metrics, and regression gates still attach to the owning external level.

### Inputs

- `state_in`: canonical State IR or a schema-faithful slice
- `context_in`: optional external context
- `control_in`: optional upstream control signals
- `resource_budget`: optional time / memory / step limits

### Outputs

- `state_out`: updated State IR
- `control_out`: learned downstream suggestions or neutral control
- `diagnostics`: level-observable signals for attribution, calibration, and regression

### Mounted vs Stub Classification

A level counts as `mounted` only when all of the following hold:

- it consumes canonical State IR or a schema-faithful slice,
- it produces non-neutral behavior on the level's contract-bearing surfaces,
- it emits diagnostics that expose what it acted on and what it produced,
- its behavior is attributable in regression as that level rather than as generic passthrough plumbing.

A level counts as a `stub` when any of the following are true:

- it only preserves the interface shape,
- it emits neutral control only,
- it does not exercise the level's contract-bearing sub-responsibilities,
- it cannot support meaningful level-addressable attribution.

Temporary partial implementations may report `implementation_status = partial_mount` in diagnostics, but for compliance and gate purposes they still count as `stub` until the level's required mounted surfaces are all live.

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

Minimum diagnostic granularity:

- level summary: `implementation_status`, invocation outcome, and high-level target summary,
- emitted object refs: affected `state_ref`, `branch_id`, `subgoal_id`, `vs_id`, or equivalent ids when applicable,
- evidence / trigger refs: verifier or retrieval refs that materially influenced the decision when applicable,
- internal-head status for composite levels, especially `L3` and `L6`.

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

Mounted implementation rule:

`L3` must be semantically decomposable into these internal control heads:

- `branch_controller`: chooses branch, subgoal, and strategy-transition behavior,
- `budget_allocator`: assigns or reallocates search, verifier, and reparse budget,
- `repair_scheduler`: orders targeted repair actions against the credited failure loci.

These heads may be implemented as separate internal modules or as one fused learned block, but mounted `L3` diagnostics must expose their outputs separately enough for attribution.

Typical diagnostics:

- budget pressure,
- backtrack rate,
- recovery precision,
- termination margin.

Attribution-supporting diagnostics must also identify:

- selected branch / subgoal or stop target,
- budget change or saturation summary,
- recovery target level or locus,
- whether the action came from ordinary expansion, backtrack, or targeted repair scheduling.

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

Canonical landing rule:

- reusable abstractions, macros, and lemma-like summaries must land in `LM`,
- branch-scoped active invariants or compressed plans must land in `FR`,
- claim-bearing invariant relations must land in `CG`,
- no behavior-affecting abstraction may survive only as hidden side metadata.

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

Mounted implementation rule:

`L6` must be semantically decomposable into these internal diagnostic heads:

- `verifier_aggregator`: aggregates evidence classes into validity and risk summaries,
- `credit_router`: emits `failure.credit`, failure taxonomy support, and suspected failure loci,
- `calibration_head`: emits calibrated confidence, abstention margin, and false-accept / false-reject risk summaries.

These heads may be implemented as separate internal modules or as one fused learned block, but mounted `L6` diagnostics must expose their outputs separately enough for attribution.

Typical diagnostics:

- false accept / false reject rates,
- calibration error,
- disagreement,
- failure-credit distribution.

Attribution-supporting diagnostics must also identify:

- supporting verifier evidence classes and refs,
- whether credit routing is multi-level or near-collapsed,
- calibration state for the accepted or rejected outcome,
- which evidence changed the downstream recovery recommendation.

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
7. Are `L3` and `L6` internal heads mounted only as internal modules rather than new external ids?
8. Is any `partial_mount` still being counted as fully mounted in regression or readiness claims?
9. Are `L5` abstractions landing canonically in `LM`, `FR`, or `CG` rather than in side metadata?
