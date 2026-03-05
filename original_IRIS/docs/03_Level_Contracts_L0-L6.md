# Level Contracts L0-L6

**Document Type:** Normative Contract (Level Interfaces)  
**Effective date:** 2026-02-27  
**Replaces (removed on 2026-02-27):** the previous split Level contract docs under `docs/Level Contracts/` (L0-1, L2, L3-4, L5-6).

**Authority:** These contracts define interface existence, I/O surfaces, stub behavior, observability, and prohibited patterns per Level.

---

## Level 0-1 Contract

**IO Alignment, Representation Surface, and Short-Horizon Consistency**

---

### 0. Scope and Intent

This document defines the contract for Level 0-1 in a foundation-model setting.

- Level 0 handles input/output alignment and representation normalization.
- Level 1 handles short-horizon consistency, local aggregation, and lightweight state maintenance.
- This contract is benchmark-agnostic and task-agnostic.

---

### Authority & Optional Mounting

- This document defines a **Level Interface Contract**, not a fixed implementation or fixed capacity requirement.
- **Interface must exist**: even when this level is disabled in a checkpoint/config, its interface and I/O schema must remain available as a stub (no-op or low-capacity adapter).
- **Implementation may be mounted/disabled/replaced**: any mounted version must satisfy this contract (I/O consistency, observability, and credit attribution compatibility).
- Hard-orchestrated flow cannot replace learned control. If this level emits control/recovery signals, those signals must be learnable, trainable, and attributable.

---

### 1. Responsibilities

#### 1.1 Level 0 Responsibilities

- Convert raw input channels into State IR-aligned observation fields.
- Apply format normalization and schema-safe tokenization.
- Maintain modality-agnostic input handling (text/image/other structured sources).

#### 1.2 Level 1 Responsibilities

- Preserve local consistency across adjacent steps.
- Provide short-horizon memory/compression signals without taking over global reasoning.
- Expose uncertainty and representation quality signals for downstream levels.

#### 1.3 Peripheral Modules

- Tokenizers, patch embedders, and compression helpers are allowed as peripherals.
- Peripherals must not become a semantic decision core or bypass trunk-level reasoning.

---

### 2. Interface

#### Inputs

- `state_in`: canonical State IR (or schema-compatible reference/slice).
- `context_in`: optional external context.
- `control_in`: optional upstream control signals.
- `resource_budget`: optional limits (time/steps/memory).

#### Outputs

- `state_out`: State IR updates for observation/surface-form fields.
- `control_out`: optional downstream control hints (default neutral).
- `diagnostics`: observable signals including confidence/uncertainty, failure tags, credit hints, plus encoding stats (for example normalization ratio, compression ratio, OOV/unknown statistics when applicable).

#### Stub Behavior (when disabled)

- `state_out = state_in` (or minimal schema normalization only).
- `control_out` returns neutral/no-op.
- `diagnostics` must still emit a disabled marker and basic summary stats.

#### Observability & Logging (minimum)

- Must support sampled logging of input summary, output summary, key gates/logits (if present), and failure/error codes (if present).
- Attribution must be traceable to trunk contribution and this level contribution, including stub mode.

---

### 3. Prohibited Patterns

- Benchmark-shaped assumptions embedded into contract semantics.
- Fixed mandatory module classes as the only valid implementation path.
- Hard-coded semantic routing that bypasses learned control.
- Semantic closed loops outside trunk dynamics.

---

### 4. Compliance Checklist

1. Is the interface preserved even when disabled (with a valid stub)?
2. In disabled mode, does State IR remain consistent and unbroken?
3. Is there any semantic closed loop bypassing trunk (violation)?
4. Is minimum observability/diagnostics provided?
5. If control/routing signals exist, are they learned, trainable, and attributable?
6. Is any benchmark-specific assumption embedded in contract semantics?

---

**End of Level 0-1 Contract**

---

## Level 2 Contract

**Latent Procedure Induction and Mid-Level Program Structure**

---

### 0. Scope and Intent

This document defines the contract for Level 2 as a benchmark-agnostic, learned
procedure layer.

Level 2 is responsible for inducing reusable latent procedures from State IR and
providing structured intermediate reasoning signals without collapsing into a
handwritten solver.

---

### Authority & Optional Mounting

- This document defines a **Level Interface Contract**, not a fixed implementation or fixed capacity requirement.
- **Interface must exist**: even when this level is disabled in a checkpoint/config, its interface and I/O schema must remain available as a stub (no-op or low-capacity adapter).
- **Implementation may be mounted/disabled/replaced**: any mounted version must satisfy this contract (I/O consistency, observability, and credit attribution compatibility).
- Hard-orchestrated flow cannot replace learned control. If this level emits control/recovery signals, those signals must be learnable, trainable, and attributable.

---

### 1. Responsibilities

- Induce latent procedure candidates from canonical State IR.
- Emit reusable mid-level structure (for example subgoal sketches, constraint bundles, or action schemas in latent form).
- Support downstream control without forcing deterministic execution flow.

Level 2 must remain learned and differentiable in semantics. It must not become
an externalized symbolic engine.

---

### 2. Interface

#### Inputs

- `state_in`: canonical State IR (or schema-compatible reference/slice).
- `context_in`: optional external context.
- `control_in`: optional decomposition/control hints.
- `resource_budget`: optional limits (time/steps/memory).

#### Outputs

- `state_out`: State IR augmented with latent procedure candidates/program-like tokens.
- `control_out`: optional decomposition suggestions (non-binding, non-hardcoded).
- `diagnostics`: confidence/uncertainty, failure tags, credit hints, plus program quality signals (for example diversity, collapse indicators, consistency scores).

#### Stub Behavior (when disabled)

- `state_out = state_in` (or minimal schema normalization only).
- `control_out` returns neutral/no-op.
- `diagnostics` must still emit a disabled marker and basic summary stats.

#### Observability & Logging (minimum)

- Must support sampled logging of candidate summaries, scoring summaries, key gates/logits (if present), and failure/error codes (if present).
- Attribution must be traceable to trunk contribution and this level contribution, including stub mode.

---

### 3. Prohibited Patterns

- Handwritten DSL executor as primary semantics.
- Fixed search policy/flowchart embedded in this level.
- Rule-only planner behavior replacing learned procedure induction.
- Benchmark-shaped latent schema assumptions.

---

### 4. Compliance Checklist

1. Is the interface preserved even when disabled (with a valid stub)?
2. In disabled mode, does State IR remain consistent and unbroken?
3. Is there any semantic closed loop bypassing trunk (violation)?
4. Is minimum observability/diagnostics provided?
5. If control/routing signals exist, are they learned, trainable, and attributable?
6. Is any benchmark-specific assumption embedded in contract semantics?

---

**End of Level 2 Contract**

---

## Level 3-4 Contract

**Learned Decomposition, Multi-Step Control, and Resource Policy**

---

### 0. Scope and Intent

This document defines the contract for Level 3-4 as the learned policy layer for
multi-step decomposition and control.

- Level 3 focuses on decomposition and subgoal management.
- Level 4 focuses on multi-step control, budget allocation, branching, and
  repair-trigger policy.

Both levels are policy layers, not benchmark-specific schedulers.

---

### Authority & Optional Mounting

- This document defines a **Level Interface Contract**, not a fixed implementation or fixed capacity requirement.
- **Interface must exist**: even when this level is disabled in a checkpoint/config, its interface and I/O schema must remain available as a stub (no-op or low-capacity adapter).
- **Implementation may be mounted/disabled/replaced**: any mounted version must satisfy this contract (I/O consistency, observability, and credit attribution compatibility).
- Hard-orchestrated flow cannot replace learned control. If this level emits control/recovery signals, those signals must be learnable, trainable, and attributable.

---

### 1. Responsibilities

- Produce learned decomposition hints and subgoal progression signals.
- Emit learned control signals for next-step intent, branching preference, and
  budget usage.
- Coordinate multi-step progress without hardcoding an execution schedule.

---

### 2. Interface

#### Inputs

- `state_in`: canonical State IR (or schema-compatible reference/slice).
- `context_in`: optional external context and optional Level 2 procedure hints.
- `control_in`: optional upstream control signals.
- `resource_budget`: optional limits (time/steps/memory).

#### Outputs

- `state_out`: State IR updates for plan trace/subgoal-trace fields (if defined by schema).
- `control_out`: next-step intention, gate logits, branch weights, or equivalent learned control suggestions.
- `diagnostics`: confidence/uncertainty, failure tags, credit hints, plus step-efficiency, loop/stall indicators, and budget usage summary.

#### Stub Behavior (when disabled)

- `state_out = state_in` (or minimal schema normalization only).
- `control_out` returns neutral/no-op so trunk control remains live.
- `diagnostics` must still emit a disabled marker and basic summary stats.

#### Observability & Logging (minimum)

- Must support sampled logging of subgoal/control summaries, key gates/logits,
  branch/termination summaries, and failure/error codes (if present).
- Attribution must be traceable to trunk contribution and this level contribution, including stub mode.

---

### 3. Prohibited Patterns

- Rule-based scheduler as the default semantics.
- Hardcoded branch/termination policy replacing learned control.
- Deterministic flowchart that bypasses trunk-mediated control dynamics.
- Benchmark-shaped control assumptions encoded as contract behavior.

---

### 4. Compliance Checklist

1. Is the interface preserved even when disabled (with a valid stub)?
2. In disabled mode, does State IR remain consistent and unbroken?
3. Is there any semantic closed loop bypassing trunk (violation)?
4. Is minimum observability/diagnostics provided?
5. If control/routing signals exist, are they learned, trainable, and attributable?
6. Is any benchmark-specific assumption embedded in contract semantics?

---

**End of Level 3-4 Contract**

---

## Level 5-6 Contract

**Self-Evaluation, Recovery Policy, and Credit Attribution Hooks**

---

### 0. Scope and Intent

This document defines the contract for Level 5-6 as the learned self-evaluation
and recovery-attribution layer.

- Level 5 focuses on evaluation and verification signal quality.
- Level 6 focuses on recovery policy hooks and credit-assignment integration.

Both levels must remain learned, observable, and attribution-compatible.

---

### Authority & Optional Mounting

- This document defines a **Level Interface Contract**, not a fixed implementation or fixed capacity requirement.
- **Interface must exist**: even when this level is disabled in a checkpoint/config, its interface and I/O schema must remain available as a stub (no-op or low-capacity adapter).
- **Implementation may be mounted/disabled/replaced**: any mounted version must satisfy this contract (I/O consistency, observability, and credit attribution compatibility).
- Hard-orchestrated flow cannot replace learned control. If this level emits control/recovery signals, those signals must be learnable, trainable, and attributable.

---

### 1. Responsibilities

- Emit learned self-evaluation signals, including uncertainty and failure tags.
- Provide repair suggestions and credit-assignment hints as learned outputs.
- Support recovery policy without collapsing into hardcoded retry scripts.

---

### 2. Interface

#### Inputs

- `state_in`: canonical State IR (or schema-compatible reference/slice).
- `context_in`: optional external context, including behavior trace/output summary.
- `control_in`: optional upstream control signals.
- `resource_budget`: optional limits (time/steps/memory).

#### Outputs

- `state_out`: State IR updates with evaluation summary and credit hints.
- `control_out`: retry/reflect/branch gate logits or equivalent learned recovery suggestions (optional).
- `diagnostics`: confidence/uncertainty, failure tags, credit hints, error signatures, and recovery recommendation summary.

#### Stub Behavior (when disabled)

- `state_out = state_in` (or minimal schema normalization only).
- `control_out` returns neutral/no-op.
- `diagnostics` must still emit a disabled marker and basic summary stats.

#### Observability & Logging (minimum)

- Must support sampled logging of verification/recovery summaries, key gates/logits (if present), and failure/error codes (if present).
- Attribution must be traceable to trunk contribution and this level contribution, including stub mode.

---

### 3. Prohibited Patterns

- Hardcoded "if wrong then rerun N times" as default semantics.
- Rule-only verifier behavior that bypasses learned evaluation.
- Fixed recovery flowcharts replacing learned policy.
- Benchmark-shaped acceptance/rejection policy in contract semantics.

---

### 4. Compliance Checklist

1. Is the interface preserved even when disabled (with a valid stub)?
2. In disabled mode, does State IR remain consistent and unbroken?
3. Is there any semantic closed loop bypassing trunk (violation)?
4. Is minimum observability/diagnostics provided?
5. If control/routing signals exist, are they learned, trainable, and attributable?
6. Is any benchmark-specific assumption embedded in contract semantics?

---

**End of Level 5-6 Contract**
