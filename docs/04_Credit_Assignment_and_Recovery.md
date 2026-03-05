# Credit Assignment and Failure Recovery

**Document Type:** Canonical Specification (Normative)  
**Effective date:** 2026-02-27  
**Replaces (removed on 2026-02-27):**
- Credit Assignment & Failure Recovery Model

(See `docs/00_INDEX.md` for the historical path mapping.)

**Authority:** Defines canonical failure taxonomy, credit routing semantics, and recovery responsibility boundaries.

---

## Credit Assignment & Failure Recovery Model

*(Canonical Specification)*

This document defines how **credit, blame, and recovery responsibility** are assigned across Levels (L0–L6) in the system.
It explicitly constrains **where learning pressure, adaptation, and retry logic must reside**, and forbids collapsing the hierarchy into end-to-end loss or opaque controller heuristics.

This document is normative and non-optional.



---

### 1. Purpose and Scope

The purpose of this document is to ensure that:

1. **Failures are attributed to specific Levels with semantic meaning**, not merely to parameters.
2. **Recovery actions modify the correct Level’s behavior**, rather than masking errors upstream or downstream.
3. **Training signals respect the architectural decomposition**, preserving modularity and long-term scalability.
4. **Control, retry, and adaptation decisions remain learnable**, not hard-coded.

This document governs:

* Online inference-time recovery behavior
* Training-time credit assignment
* Inter-Level responsibility boundaries

It does **not** specify optimization algorithms, loss functions, or implementation details.

---

### 2. Core Principles (Non-Negotiable)

#### 2.1 Semantic Credit, Not Gradient Convenience

Credit assignment is defined in terms of **semantic failure modes**, not gradient flow convenience.

> A Level is responsible if the *meaning* of its output caused downstream failure,
> even if upstream or downstream gradients could technically fix it.

---

#### 2.2 No End-to-End Collapse

The system **must not** rely on a single end-to-end loss that freely adjusts all Levels simultaneously.

* Joint optimization is allowed only if **Level-specific attribution signals** are preserved.
* Any training regime that removes Level identity is invalid.

---

#### 2.3 Failure Recovery Is a First-Class Output

Failure recovery decisions are **model outputs**, not engineering fallbacks.

* “Retry,” “expand search,” “re-perceive,” etc. are **actions chosen by learned heads**.
* Hard-coded retry policies are considered technical debt and must be explicitly marked as such; they may only act as guardrails, not routine policy.

---

### 3. Failure Taxonomy (Canonical)

All failures must be mapped into one or more of the following categories.

#### 3.1 Representation Failure (L0 / L1)

**Symptoms**

* Missing, fragmented, or spurious objects
* Incorrect relations or events
* Inconsistent state across rollouts

**Semantic Cause**

* The State IR does not faithfully represent the task environment.

**Primary Responsible Levels**

* Level 0 (Objectization / Relation / Event induction)
* Level 1 (Dynamics / State transition modeling)

---

#### 3.2 Procedural Failure (L2)

**Symptoms**

* Programs that are syntactically valid but semantically wrong
* Correct primitives composed in incorrect order
* Search converges on locally coherent but globally incorrect procedures

**Semantic Cause**

* Incorrect program induction, execution, or scoring.

**Primary Responsible Level**

* Level 2 (Program Proposal, Executor, Scorer)

---

#### 3.3 Search / Resource Allocation Failure (L3)

**Symptoms**

* Premature termination
* Insufficient beam width or rollout depth
* Excessive computation on low-value branches

**Semantic Cause**

* Poor allocation of computational budget or exploration depth.

**Primary Responsible Level**

* Level 3 (Budget Controller, Node Expansion, Termination)

---

#### 3.4 Knowledge / Memory Failure (L4)

**Symptoms**

* Relevant prior programs or concepts not retrieved
* Incorrect or outdated memory reused
* Overwriting useful abstractions

**Semantic Cause**

* Retrieval, consolidation, or write-gating errors.

**Primary Responsible Level**

* Level 4 (Memory Read / Write / Consolidation)

---

#### 3.5 Abstraction Failure (L5)

**Symptoms**

* Overly concrete reasoning when abstraction is needed
* Over-general macros that erase critical distinctions

**Semantic Cause**

* Incorrect abstraction granularity or macro construction.

**Primary Responsible Level**

* Level 5 (Abstraction / Macro Management)

---

#### 3.6 Evaluation / Diagnosis Failure (L6)

**Symptoms**

* Accepting invalid solutions
* Rejecting valid solutions
* Miscalibrated confidence leading to wrong termination decisions

**Semantic Cause**

* Faulty verification or meta-evaluation.

**Primary Responsible Level**

* Level 6 (Verifier, Confidence, Credit Router)

---

### 4. Credit Assignment Flow (Canonical Direction)

Credit assignment flows **top-down**, not bottom-up.

#### 4.1 Directionality Rule

1. **Level 6 diagnoses outcome validity**
2. **Level 3 decides whether more effort is warranted**
3. **Level 6 routes blame toward candidate Levels**
4. **Target Levels adjust behavior or are re-invoked**

Lower Levels **do not self-diagnose global failure**.

---

#### 4.2 L6 Credit Router (Mandatory)

Level 6 must output a **credit routing distribution**, not a hard decision:

```
C = { c0, c1, c2, c3, c4, c5 }
```

Where:

* `ck ∈ [0,1]`
* `Σ ck = 1`
* Each `ck` represents the probability that Level `k` is responsible for failure

This distribution is consumed by:

* Level 3 (for recovery strategy selection)
* Training pipelines (for loss routing)

Level 6 must not emit direct computation-budget parameters; recovery scheduling and parameterization are Level 3 policy responsibilities.

---

### 5. Failure Recovery Semantics (Inference-Time)

#### 5.1 Recovery Is Targeted, Not Global

When failure is detected:

* The system **must not** blindly rerun the entire pipeline.
* Recovery actions must correspond to the credited Level(s).

---

#### 5.2 Canonical Recovery Actions by Level

| Credited Level | Permitted Recovery Actions                                          |
| -------------- | ------------------------------------------------------------------- |
| L0             | Re-objectize, adjust segmentation thresholds, re-induce relations   |
| L1             | Increase rollout depth, alter dynamics uncertainty handling         |
| L2             | Expand program beam, resample proposals, alter executor temperature |
| L3             | Increase budget, delay termination, rebalance exploration           |
| L4             | Increase retrieval k, bypass consolidation, force fresh write       |
| L5             | Change abstraction granularity, suppress macro usage                |
| L6             | Re-verify with stricter criteria, recalibrate confidence            |

This table defines permissible actions, not decision authority: Level 6 diagnoses and routes credit, while Level 3 selects and parameterizes recovery actions.

Recovery actions must be **parameterized by learned heads**, even if bounded by hard limits.

---

#### 5.3 Multi-Level Failures

If credit is distributed across multiple Levels:

* Recovery may be staged (e.g., L2 first, then L0)
* Or parallel (e.g., re-perception + larger program beam)

The ordering itself should be learnable via Level 3 policies.

---

### 6. Training-Time Credit Assignment

#### 6.1 Loss Routing Constraint

Training losses must be **Level-addressable**.

* Each loss term must declare which Level(s) it supervises.
* Shared losses must expose attribution weights consistent with L6 credit routing.

---

#### 6.2 Forbidden Training Patterns

The following are explicitly forbidden:

* Single scalar loss backpropagated uniformly to all Levels
* Reinforcement learning signals that bypass Level identity
* Controller-only training that treats other Levels as black boxes

---

#### 6.3 Acceptable Patterns

Allowed patterns include:

* Multi-head losses with Level-specific targets
* Hierarchical RL where rewards are decomposed per Level
* Self-consistency losses gated by L6 diagnostic outputs

---

### 7. Responsibility Boundaries (Hard Rules)

#### 7.1 Lower Levels Cannot “Fix” Higher-Level Mistakes

Examples:

* L0 must not learn to encode task solutions to compensate for bad L2 programs.
* L1 must not hallucinate dynamics to rescue poor search decisions.

---

#### 7.2 Higher Levels Cannot Mask Lower-Level Errors

Examples:

* L3 must not endlessly increase search depth to hide perception failures.
* L6 must not lower verification thresholds to accept flawed solutions.

---

### 8. Observable Signals Required (Minimum)

Each Level must expose **diagnostic signals** consumable by L6:

* Confidence or uncertainty estimates
* Entropy or dispersion measures
* Self-consistency or disagreement indicators

Opaque Levels that provide no introspectable signals are non-compliant.

---

### 9. Non-Goals (Explicit)

This model explicitly does **not** aim to:

* Produce human-readable explanations of credit assignment
* Guarantee perfect blame isolation
* Eliminate all heuristic bounds in early MVP stages

However, any heuristic must be:

* Explicitly labeled
* Isolated
* Designed to be replaced by learned behavior

---

### 10. Summary (Normative)

* Credit assignment is **semantic, hierarchical, and explicit**
* Failure recovery is **targeted, learnable, and Level-aware**
* Training must **preserve Level identity**
* Any design that collapses these distinctions violates the architecture

This document is foundational.
Any system variant that contradicts it is considered a different model.
