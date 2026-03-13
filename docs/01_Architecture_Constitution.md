# Architecture Constitution

**Document Type:** Authoritative Specification (Normative)  
**Effective date:** 2026-03-10  
**Authority:** This document defines the non-negotiable architectural invariants for IRIS-math v2.

---

## System Invariants & Non-Negotiables

### 0. Document Positioning

#### 0.1 Purpose

This document defines the immutable architectural constraints for **IRIS-math v2**:

- document-native,
- multimodal,
- verifier-centered,
- mathematically structured,
- single-trunk.

The target is not a generic chatbot and not a benchmark-locked contest solver.

#### 0.2 Documentation-First Transition

The repository is currently in a documentation-first transition:

- active v2 documents define the target architecture,
- baseline code may still expose pre-v2 scaffolding,
- transition lag does not change the target contract.

#### 0.3 Benchmark Boundary

Benchmarks do not define the architecture.
They may participate in training **only** under the tiered policy in `docs/07_Data_Constitution.md`.

---

### 1. Trunk Invariants

#### 1.1 Single Trunk

- IRIS-math v2 must have exactly one primary parameter-bearing semantic trunk.
- The trunk is the main locus of representation formation, control, recovery conditioning, and learned adaptation.
- A second high-capacity semantic network that can compete with or replace the trunk is forbidden.

#### 1.2 Allowed Trunk Families

The trunk may be attention-based, SSM-based, or hybrid, so long as it still behaves as a single trunk under one primary training objective family.

#### 1.3 Peripheral Modules

Tokenizers, parsers, OCR systems, retrievers, verifiers, and other peripherals are allowed only if they do not become the semantic brain.

---

### 2. State IR Invariants

#### 2.1 Canonical Internal State

- State IR is the canonical internal representation shared across levels.
- Raw parser traces, raw tool outputs, or unmanaged side channels must not bypass State IR.

#### 2.2 Math-Native Semantics

State IR must represent mathematical work state, not generic chat memory.
It must support:

- problem framing,
- symbol grounding,
- constraints,
- frontier management,
- lemma applicability,
- verification,
- control state.

#### 2.3 Controlled Evolution

Changes to slot inventory, ordering, or mandatory semantics require explicit versioned updates to `docs/02_State_IR_Spec.md`.

---

### 3. Learnable Routing and Control Invariants

#### 3.1 Learned-by-Default Control

- Routing, gating, recovery, termination, and resource-allocation decisions must be learnable by default.
- Hard-coded orchestration may exist only as clearly labeled temporary technical debt guardrails.

#### 3.2 Recovery Is Learned Policy

Failure recovery must not collapse into a static retry script.
Recovery is a model behavior informed by verification and credit routing.

---

### 4. Level Interface and Mounting Invariants

#### 4.1 Stable External Numbering

The active external interface set remains `L0-L6` during the first-round v2 rewrite.
The semantic meaning of these levels is math-specific and defined in `docs/03_Level_Contracts_L0-L6.md`.

#### 4.2 Interface Persistence

Each level interface must exist even when the mounted implementation is a stub.
Removing a level interface or its diagnostics surface is an architectural violation.

---

### 5. Credit Assignment and Verification Invariants

#### 5.1 Level-Addressable Credit

The system must preserve level-addressable credit assignment.
Outcome, recovery, and learning signals cannot be flattened into opaque end-to-end blame.

#### 5.2 Verifier-Centered Loop

Verification is first-class.
The architecture must support:

- local and global validity checks,
- counterexample or contradiction probing,
- confidence calibration,
- targeted recovery conditioned on credit routing.

---

### 6. Benchmarks, Tools, and Documents

#### 6.1 Benchmarks

- Benchmarks may inform curriculum and train-visible structural signals only through the declared tiering policy in `docs/07_Data_Constitution.md`.
- Benchmarks must not become the sole optimization objective.
- Strict held-out evaluation must remain available.

#### 6.2 Tools

- Tools may generate data, provide verification evidence, or supply monitoring artifacts.
- Tools must not replace trunk-learned reasoning or control.

#### 6.3 Documents

Documents are first-class input objects.
PDF, scanned notes, formula regions, tables, and diagrams are legitimate inputs once normalized through the canonical parsing pipeline.

---

### 7. Summary

IRIS-math v2 must always preserve:

- one primary trunk,
- a canonical math-native State IR,
- learnable routing, gating, recovery, and termination,
- persistent level interfaces `L0-L6`,
- verifier-centered credit assignment,
- tool usage without tool-first cognition,
- benchmark tiering without benchmark lock-in.

---

## What This Model Is Explicitly **NOT**

### 1. Not a Tool-First Agent

Parsers, search utilities, formalizers, and verifiers may contribute signals.
They do not define the system's intelligence substrate.

### 2. Not a Benchmark-Locked Contest Solver

The architecture is not defined by ARC, AIMO, FrontierMath, or any other single benchmark family.

### 3. Not a Symbolic-First DSL Core

The system may produce or consume symbolic artifacts, but it is not a handwritten symbolic executor with a neural wrapper.

### 4. Not a Deterministic If/Else Controller

Routing, strategy switching, recovery, and stopping cannot be hard-coded as routine policy.

### 5. Not a Second-Trunk Architecture

No large parallel semantic subsystem may compete with the trunk.

### 6. Not a Generic Chat-First VLM

Multimodality exists to support mathematical documents, formulas, diagrams, and proof-bearing artifacts, not to maximize generic chat breadth.

---

## Single Trunk Contract & Allowed Variations

### 1. Definition

The trunk is the only high-capacity semantic substrate responsible for:

- integrating State IR,
- conditioning strategy and control,
- fusing retrieval and verification signals,
- producing updated work state.

### 2. Allowed Variations

Allowed families include:

- attention-only,
- SSM-only,
- attention/SSM hybrids,
- retrieval-augmented trunks where retrieval remains peripheral and learnedly controlled.

### 3. Prohibited Patterns

Forbidden patterns include:

- dual-core "encoder + solver" systems,
- parser/solver pipelines that bypass State IR,
- separate semantic control brains,
- benchmark-shaped trunk specialization.

### 4. Required Interfaces

The trunk must support semantically equivalent forms of:

- `encode(inputs) -> state`
- `step(state, control/context) -> state'`
- `decode(state) -> outputs or verifier-facing summaries`

The exact API shape may vary; the semantic contract may not.

---

## Routing, Gating, and Control Are Learnable

### 1. Core Principle

All behavior-affecting routing, gating, stopping, and recovery choices must be learnable and attributable.

### 2. Required Learned Surfaces

The architecture must expose learnable control for:

- level invocation,
- frontier selection,
- budget allocation,
- retrieval usage,
- recovery targeting,
- output / proof candidate selection.

### 3. Temporary Guardrails

Hard caps are allowed only as `TEMPORARY TECHNICAL DEBT` guardrails with removal criteria and intended learned replacements.

### 4. Compatibility Requirement

Control signals must remain compatible with:

- training feedback,
- level-addressable credit routing,
- calibration,
- regression diagnostics.

---

## Related Active Documents

- `docs/02_State_IR_Spec.md`
- `docs/03_Level_Contracts_L0-L6.md`
- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/07_Data_Constitution.md`
- `docs/09_Training_Profiles_and_Scaling.md`
- `docs/18_Optimization_and_Learning_Contract.md`
- `docs/19_Runtime_and_Task_Adjudication_Semantics.md`
