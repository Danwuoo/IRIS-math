# State IR Spec

**Document Type:** Canonical Specification (Normative)  
**Effective date:** 2026-02-27  
**Replaces (removed on 2026-02-27):**
- State IR Canonical Spec
- State IR Examples & Edge Cases

(See `docs/00_INDEX.md` for the historical path mapping.)

**Authority:** This document defines the closed State IR token type set and canonical ordering.

---

## State IR Canonical Specification

**Version 1.0 (Normative)**

### 1. Purpose and Scope

This document defines the **canonical State Intermediate Representation (State IR)** used across the entire system.

State IR is the *only* shared representational substrate consumed and produced by the Trunk and all Levels (L0–L6).

This specification exists to ensure:

- Cross-Level composability without implicit assumptions
- Architectural stability against uncontrolled token or modality expansion
- Clear semantic contracts independent of tensor shapes or implementation details

Any deviation from this specification constitutes an **architectural violation**, not an optimization.

---

### 2. Core Principles (Non-Negotiable)

1. **Single Canonical Representation**
    
    All internal reasoning state must be representable as State IR.
    
    No Level may introduce an alternative latent state space.
    
2. **Closed Token Type Set**
    
    The set of token types defined in this document is *closed*.
    
    New token types MAY NOT be added without a versioned revision of this spec.
    
3. **Uniform Trunk Processing**
    
    All State IR tokens are processed by the same Trunk without conditional routing at the representation level.
    
4. **Semantic, Not Procedural**
    
    State IR encodes *what the system believes the world/task/program state is*, not how it was produced.
    

---

### 3. Token Type System

State IR consists of a fixed set of token categories.

Each token is a vector in a shared latent space of dimension **d**.

#### 3.1 Token Categories (Exhaustive)

| Token Type | Symbol | Cardinality | Description |
| --- | --- | --- | --- |
| Task Token | T | 1 | Encodes the task-level intent, objective, and constraints |
| Global Token | G | 1 | Aggregated global context and control-relevant state |
| Object Tokens | O | Nₒ | Discrete entities or structured units in the world |
| Relation Tokens | R | Nᵣ | Explicit relations between objects |
| Event Tokens | X | Nₓ | State transitions, actions, or temporal changes |
| Macro Tokens | M | Nₘ | Abstracted patterns, programs, or compressed histories |

No other token categories are permitted.

---

### 4. Canonical Sequence Construction

#### 4.1 Concatenation Order (Mandatory)

All State IR instances MUST be constructed using the following fixed order:

```
Z = [ T ; G ; O₁…Oₙ ; R₁…Rₖ ; X₁…Xₘ ; M₁…Mₚ ]

```

Only the token categories defined in Section 3.1 may appear in `Z`. Program IR tokens are not State IR tokens and must never be concatenated into the canonical State IR sequence `Z`, even transiently.

Where:

- `T ∈ ℝ¹ˣᵈ`
- `G ∈ ℝ¹ˣᵈ`
- `O ∈ ℝᴺᵒˣᵈ`
- `R ∈ ℝᴺʳˣᵈ`
- `X ∈ ℝᴺˣˣᵈ`
- `M ∈ ℝᴺᵐˣᵈ`

Padding MAY be applied internally but MUST preserve relative ordering.

#### 4.2 Ordering Stability

- Relative ordering *within* each token group MUST be stable across a reasoning cycle.
- Reordering tokens is considered a **state mutation** and must be explicitly produced by a learned module.

---

### 5. Token Semantics

#### 5.1 Task Token (T)

The Task Token represents:

- Task objective
- Success criteria
- Global constraints
- Instructional priors

It MUST be present at all times and MUST NOT be removed or replaced.

#### 5.2 Global Token (G)

The Global Token serves as:

- Cross-token aggregation point
- Control and routing context
- Memory fusion anchor

Heads MAY preferentially read from or write to G, but MUST NOT assume exclusivity.

#### 5.3 Object Tokens (O)

Object Tokens represent:

- Discrete entities
- Structured components
- Symbolic units derived from perception or abstraction

They MUST be referentially stable within a reasoning cycle.

#### 5.4 Relation Tokens (R)

Relation Tokens encode:

- Binary or higher-order relations between objects
- Structural, spatial, or logical links

Relations MUST NOT be implicitly encoded solely via object embeddings.

#### 5.5 Event Tokens (X)

Event Tokens represent:

- State transitions
- Actions
- Temporal deltas between states

They are the only token type permitted to carry explicit temporal semantics.

#### 5.6 Macro Tokens (M)

Macro Tokens represent:

- Learned abstractions
- Program fragments
- Compressed multi-step patterns

They MUST be treated as first-class tokens, not annotations.

---

### 6. Mandatory Embeddings

Every token in State IR MUST include the following additive embeddings:

#### 6.1 Type Embedding (Required)

A learned embedding indicating token category:

- Task
- Global
- Object
- Relation
- Event
- Macro

Type embeddings are mandatory and MUST be consumed by the Trunk.

#### 6.2 Structural Embedding (Conditional)

Applied where applicable:

- Object geometry or topology
- Relation endpoints or arity
- Program node roles (for Macro tokens)

Absence of structure MUST be encoded explicitly, not omitted.

#### 6.3 Temporal Embedding (Restricted)

- Applied ONLY to Event Tokens and (optionally) Macro Tokens
- MUST NOT be applied to Object or Relation Tokens

---

### 7. State IR Mutability Rules

1. **Creation**
    
    New tokens MAY be created only by learned modules (e.g., L0, L2, L5).
    
2. **Deletion**
    
    Tokens MAY be dropped only via explicit learned gating or consolidation.
    
3. **Modification**
    
    Token content may change freely through Trunk processing and adapters.
    
4. **Persistence**
    
    Tokens persist across reasoning cycles unless explicitly removed.
    

---

### 8. Cross-Level Contract

All Levels (L0–L6):

- MUST consume State IR as defined here
- MUST produce State IR or scalar outputs derived from it
- MUST NOT bypass State IR to exchange hidden states

No Level may assume privileged access to pre- or post-Trunk representations.

---

### 9. Explicit Non-Goals

State IR is **not**:

- A human-readable program representation
- A symbolic execution trace
- A DSL or instruction sequence
- A tool invocation log

Any of the above must be *encoded into* State IR, not replace it.

---

### 10. Versioning and Extension Policy

- This specification is versioned.
- Any change to token types, ordering, or mandatory embeddings requires a new major version.
- Silent extensions or “temporary” token hacks are forbidden.

---

### 11. Compliance Requirement

Any model, agent, or training procedure claiming compatibility with this architecture MUST:

- Accept this State IR verbatim
- Reject undefined token types
- Preserve ordering and semantics as specified

Non-compliance is considered a **system-level defect**, not a performance trade-off.

---

**End of Document**

---

## **State IR Examples & Edge Cases**

**(Normative Companion to “State IR Canonical Spec”)**

This document provides concrete examples, pathological cases, and explicit boundary conditions for the **State IR** used across all Levels and the Single Trunk.
Its purpose is to prevent silent drift, implicit token invention, or heuristic shortcuts by agents or future contributors.

This document is **normative**: if an example here contradicts an implementation, the implementation is incorrect.

This document is written to be read *after* **State IR Canonical Spec** and *before* any Level-specific contract.

---

### 1. Scope and Non-Goals

#### 1.1 Scope

This document covers:

* Canonical **example instantiations** of State IR
* **Edge cases** that commonly induce architectural violations
* Explicitly **forbidden interpretations**
* Clarifications on **absence, emptiness, truncation, and overflow**

#### 1.2 Non-Goals

This document does **not**:

* Define training procedures
* Specify loss functions
* Introduce new token types
* Redefine State IR semantics

---

### 2. Canonical Reminder: Fixed Token Inventory

All examples in this document use the **fixed token inventory**:

```
Z = [ T ; G ; O₁…Oₙ ; R₁…Rₘ ; X₁…Xₖ ; M₁…Mₚ ]
```

Where:

* `T` : Task token (exactly 1)
* `G` : Global token (exactly 1)
* `O` : Object tokens (0 ≤ n ≤ N_o)
* `R` : Relation tokens (0 ≤ m ≤ N_r)
* `X` : Event tokens (0 ≤ k ≤ N_x)
* `M` : Macro tokens (0 ≤ p ≤ N_m)

**Invariant**:
Token *types* are fixed. Token *counts* may vary, but ordering and presence slots are stable.

No example in this document introduces any additional token category.

---

### 3. Example 1: Minimal Static Task (Degenerate Case)

#### 3.1 Scenario

A trivial task with:

* No objects
* No relations
* No events
* No macros

Example: “Output a constant symbol regardless of input.”

#### 3.2 State IR Instantiation

```
Z = [
  T,
  G,
  ∅ O,
  ∅ R,
  ∅ X,
  ∅ M
]
```

#### 3.3 Key Properties

* Object, Relation, Event, and Macro sections are **empty**, not removed.
* Trunk input length is still ≥ 2.
* Type embeddings are still applied to `T` and `G`.

#### 3.4 Explicit Edge Rule

**Absence is represented by empty sections, never by deleting token classes.**

If an implementation removes the Object segment entirely, it is invalid.

---

### 4. Example 2: Simple Object-Only Perception Task

#### 4.1 Scenario

Perceptual input yields objects but no relations or events.

Example: detecting isolated shapes without spatial relationships.

#### 4.2 State IR Instantiation

```
Z = [
  T,
  G,
  O₁, O₂, O₃,
  ∅ R,
  ∅ X,
  ∅ M
]
```

#### 4.3 Clarifications

* Object tokens **must not** encode relations implicitly.
* “Nearest neighbor”, “alignment”, or “containment” must **not** be smuggled into object embeddings.

#### 4.4 Forbidden Shortcut

Encoding relational structure inside object embeddings to avoid `R` tokens is forbidden.

---

### 5. Example 3: Object + Relation Graph (No Events)

#### 5.1 Scenario

A static structured scene.

Example: a grid or graph with stable relationships.

#### 5.2 State IR Instantiation

```
Z = [
  T,
  G,
  O₁…Oₙ,
  R₁…Rₘ,
  ∅ X,
  ∅ M
]
```

#### 5.3 Relation Token Semantics

Each `Rᵢ` represents:

* A learned relation embedding
* Soft references to participating objects (via learned attention or indices)
* No hard-coded symbolic edges

#### 5.4 Edge Case: Dense Relations

If `m > N_r`:

* Relations are **truncated**, not merged.
* Truncation policy must be learned or deterministic, but **documented**.
* Overflow must **not** create additional token types.

---

### 6. Example 4: Eventful Transition (Single Step)

#### 6.1 Scenario

A before/after change in state.

Example: applying a transformation once.

#### 6.2 State IR Instantiation

```
Z = [
  T,
  G,
  O₁…Oₙ,
  R₁…Rₘ,
  X₁,
  ∅ M
]
```

#### 6.3 Event Token Semantics

An Event token:

* Represents *change*, not state
* May encode:

  * Δobject attributes
  * Δrelations
  * Action-conditioned effects

#### 6.4 Explicit Rule

Events do **not** replace objects or relations.
They coexist and contextualize them.

---

### 7. Example 5: Multi-Step Rollout with Events

#### 7.1 Scenario

Sequential reasoning or rollout over time.

#### 7.2 State IR Instantiation

```
Z = [
  T,
  G,
  O₁…Oₙ,
  R₁…Rₘ,
  X₁, X₂, X₃,
  ∅ M
]
```

#### 7.3 Time Encoding

* Only `X` tokens receive **time embeddings**
* Objects and relations remain time-invariant unless updated via events

#### 7.4 Edge Case: Event Overflow

If `k > N_x`:

* Older events are dropped or compressed
* Objects must **not** absorb temporal history to compensate

---

### 8. Example 6: Macro Abstraction Present

#### 8.1 Scenario

Repeated patterns summarized into abstractions.

#### 8.2 State IR Instantiation

```
Z = [
  T,
  G,
  O₁…Oₙ,
  R₁…Rₘ,
  X₁…Xₖ,
  M₁, M₂
]
```

#### 8.3 Macro Semantics

Macro tokens:

* Summarize repeated object/event structures
* Are **not executable programs**
* Are **not control tokens**

#### 8.4 Forbidden Interpretation

Macros must not act as:

* Hard subroutines
* If/else controllers
* DSL instructions

They are representational, not imperative.

---

### 9. Example 7: Program Execution Does NOT Modify Token Inventory

#### 9.1 Scenario

A program proposal and execution step at Level 2.

#### 9.2 Correct Behavior

* Program IR exists **outside** State IR
* Execution produces:

  * Updated embeddings of existing tokens, or
  * New *values* routed via fusion

#### 9.3 Forbidden Behavior

* Injecting “Program tokens” into `Z`
* Temporarily expanding State IR schema

State IR is **structurally immutable**.

---

### 10. Null, Padding, and Masking Semantics

#### 10.1 Padding

* Padding is positional and masked
* Padding tokens still have a **type**
* Padding must not be overloaded as “null logic”

#### 10.2 Null Meaning

There is **no null token** in State IR.

Absence is expressed by:

* Zero count in that token segment
* Not by sentinel embeddings

---

### 11. Cross-Level Consistency Constraints

#### 11.1 All Levels See the Same Z

* Level 0–6 operate on the **same schema**
* No Level may reinterpret token meaning

#### 11.2 Learned Routing Does Not Alter Schema

Gating or routing:

* May scale contributions
* May skip computation
* Must not remove token types

---

### 12. Common Failure Modes (Explicitly Disallowed)

| Failure Mode                         | Why It Is Invalid             |
| ------------------------------------ | ----------------------------- |
| Folding relations into objects       | Breaks IR factorization       |
| Using macros as programs             | Violates abstraction boundary |
| Encoding control flow in token order | Router ≠ if-else              |
| Adding temporary tokens              | Schema drift                  |
| Removing unused token types          | Breaks trunk contract         |

---

### 13. Compliance Checklist

An implementation is **State IR compliant** iff:

* Token inventory exactly matches the canonical set
* Empty segments remain present as empty
* No Level introduces or deletes token types
* All “logic” beyond representation lives in weights

---

### 14. Relationship to Other Documents

* Depends on: **State IR Canonical Spec**
* Constrains: **All Level Contracts**
* Enforced by: **Core Invariants & Non-Negotiables**
* Assumed by: **Single Trunk Contract**

---

### 15. Final Normative Statement

> **State IR is a representational contract, not a convenience layer.**
> Any optimization that alters its structure is a system violation, even if task performance improves.
