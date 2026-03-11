# Training Run Governance

**Document Type:** Engineering Governance (Policy-Binding)  
**Scope:** Segment transactions, exactly-once resume, runtime lock, reproducibility controls, and provenance requirements for IRIS-math v2  
**Related active docs:** `docs/05_Eval_Metrics_Spec.md`, `docs/06_Regression_and_Phase_Gates.md`, `docs/07_Data_Constitution.md`, `docs/09_Training_Profiles_and_Scaling.md`, `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`, `docs/16_Verifier_and_Formalization_Stack.md`, `docs/17_Scaling_Promotion_and_Readiness.md`

---

## 0. Positioning and Authority

- This document is policy-binding for training operations.
- It does not override architecture, State IR, level, or credit-assignment contracts.
- Benchmark handling in training must follow `docs/07_Data_Constitution.md`; this document governs execution semantics, not benchmark policy by itself.
- Benchmark-family-specific allowances and forbidden uses are governed by `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.
- Verifier evidence interpretation is governed by `docs/16_Verifier_and_Formalization_Stack.md`.
- Profile-readiness claims are governed by `docs/17_Scaling_Promotion_and_Readiness.md`.

---

## 1. Purpose

Define the run-time governance required for long mathematical training runs:

- segment transaction semantics,
- exactly-once progress,
- reproducibility and runtime lock control,
- data-realization and decontamination policy capture,
- parser / formalizer / verifier provenance capture,
- resume drift diagnosis.

---

## 2. Governance Priorities

Training governance exists to preserve stable, attributable behavior distributions:

- failure-taxonomy observability,
- credit-routing quality,
- calibration quality,
- strategy diversity,
- contamination control,
- provenance coverage,
- resume consistency.

Outcome metrics and benchmark scores are secondary to these controls.

Benchmark policy summary:

- Tier 1 benchmark data may be train-visible only under the declared policy in `docs/07_Data_Constitution.md`.
- Tier 2 and Tier 3 remain evaluation surfaces.
- Governance must record which executable data-realization and decontamination policies were active for the run.
- Family-specific benchmark usage must also remain consistent with `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.

---

## 3. Definitions

**Training Segment**  
An atomic training unit that groups a fixed data slice, its traces, and its metric emissions.

**Segment Status**

- `PENDING`: execute path completed, apply not committed
- `APPLIED`: committed update

**Micro Step**  
One forward/backward pass on a microbatch.

**Optimizer Step**  
One applied update after a full accumulation window.

---

## 4. Segment Transaction Model

### 4.1 Execute vs Apply

**Execute** may consume compute and RNG but must not mutate committed global state.

**Apply** commits:

- model weights,
- optimizer state,
- behavior-affecting RNG,
- behavior-affecting memory / macro state,
- provenance identities relevant to the segment.

### 4.2 Boundary Rules

1. Accumulation windows must be fully contained inside a single segment.
2. Cross-segment accumulation is forbidden.
3. Segment boundaries must align with optimizer-step boundaries.
4. A segment counts as complete only after `APPLIED`.

---

## 5. Journal and Exactly-Once Rules

Journal schema remains `iris.segment_journal/v1` and must stay append-only.

### 5.1 Minimum Journal Record Fields

| Field | Type | Notes |
| --- | --- | --- |
| `schema` | string | `iris.segment_journal/v1` |
| `event_id` | int | Monotonic per run |
| `run_id` | string | Immutable run identity |
| `segment_id` | int | Monotonic segment index |
| `status` | enum | `PENDING` or `APPLIED` |
| `optimizer_step_id` | int | Target optimizer step |
| `dataset_slice_id` | string | Deterministic data-window id |
| `rng_hash_pre` | string | Hash at segment entry |
| `rng_hash_post` | string/null | Null for `PENDING`, required for `APPLIED` |
| `loss_hash` | string | Loss-summary hash |
| `grad_stats_hash` | string | Gradient-summary hash |
| `code_version_hash` | string | Code provenance |
| `config_hash` | string | Full config provenance |
| `runtime_lock_manifest_sha256` | string | Runtime provenance |
| `data_realization_policy_id` | string | Active `data_realization_policy/v1` id |
| `decontam_policy_id` | string | Active `decontam_policy/v1` id |
| `benchmark_family_policy_refs` | string/list | Active `benchmark_family_policy/v1` refs when benchmark-derived material is present |
| `parser_provenance_id` | string/null | Required for document-derived or multimodal segments; identifies the canonical parse pipeline manifest |
| `parser_provenance_refs` | string/object/null | Required when layout, OCR, formula parsing, or semantic-unit typing surfaces are mounted; must resolve manifest ids for each active surface |
| `parse_config_fingerprint` | string/null | Required when document parsing is active; hashes behavior-affecting parse config |
| `ocr_layout_extractor_version` | string/null | Human-readable summary only; not authoritative without manifest refs |
| `formula_parser_version` | string/null | Human-readable summary only; not authoritative without manifest refs |
| `formalizer_provenance_id` | string/null | Required when natural-to-formal conversion is active |
| `formalizer_version` | string/null | Human-readable summary only; not authoritative without manifest ref |
| `verifier_provenance_id` | string/null | Required when verifier-generated labels or checks are active |
| `verifier_build_id` | string/null | Human-readable summary only; not authoritative without manifest ref |
| `checkpoint_ref` | string/null | Required for `APPLIED` |

### 5.2 Authority Rules

- Resume truth source is the last `APPLIED` event.
- `PENDING` means replay is required.
- Checkpoints without a matching `APPLIED` event are non-authoritative.

---

## 6. Checkpoint Atomicity and Commit Order

Commit protocol:

1. Append `PENDING` event
2. Execute segment
3. Build post-apply state in memory
4. Write checkpoint to a temp path
5. `fsync` checkpoint payload and metadata
6. Atomic rename to final path
7. `fsync` parent directory metadata
8. Append `APPLIED` event

### 6.1 Minimum Checkpoint Payload

- model weights
- optimizer state
- full RNG tree
- behavior-affecting memory / macro stats
- `segment_id_last_applied`
- `optimizer_step_id_last_applied`
- `dataset_slice_id_last_applied`
- `runtime_lock_manifest_id`
- `runtime_lock_manifest_sha256`
- `data_realization_policy_id`
- `decontam_policy_id`
- `benchmark_family_policy_refs`
- `parser_provenance_id`
- `parser_provenance_refs`
- `parse_config_fingerprint`
- `ocr_layout_extractor_version`
- `formula_parser_version`
- `formalizer_provenance_id`
- `formalizer_version`
- `verifier_provenance_id`
- `verifier_build_id`
- `journal_head_event_id`
- `journal_head_hash`
- `code_version_hash`
- `config_hash`

---

## 7. Behavior-Affecting RNG Governance

RNG is committed state and must be checkpointed/restored exactly.

Rules:

- Global RNG write-back during execute is forbidden.
- RNG tree hash pre/post must be journaled.
- Resume must restore full RNG state, not only a seed scalar.

Subsystem examples:

- model stochastic ops,
- routing/gating sampling,
- data shuffle / sampling,
- synthetic task generation,
- verifier probes / perturbations.

---

## 8. Data Slice Determinism

Data must be replayable by identity, not iterator position.

Required keys:

- `run_id`
- `segment_id`
- `micro_step_idx`
- `dataset_slice_id`
- `data_seed`

When the slice is document-derived or verifier-derived, the effective slice identity must also be stable with respect to:

- data-realization policy id,
- decontamination policy id,
- parser provenance id,
- parser provenance refs,
- parse config fingerprint,
- OCR/layout and formula-parser version summaries,
- formalizer provenance id,
- verifier provenance id.

---

## 9. Runtime Lock Policy

Runtime lock manifests remain mandatory.

Pinned surfaces must include:

- system / driver surface,
- Python + framework surface,
- compilation / XLA surface,
- verifier build provenance when the verifier is part of training-time supervision,
- parser / extractor build provenance when document parsing is part of the training data path,
- formalizer build provenance when natural-to-formal conversion is part of the training data path.

### 9.1 Resume Validation Policy

- `strict` is the default.
- `unsafe_resume` is disallowed after the baseline freeze point defined by the active phase/profile policy.

---

## 10. S8 Resume Consistency Alignment

When `S8` is active, validate crash classes:

1. during execute,
2. after execute and before checkpoint commit,
3. after checkpoint commit with journal reconciliation.

Resume artifacts must preserve:

- data slice identity,
- runtime lock identity,
- parser/formalizer/verifier provenance identity where relevant.

---

## 11. Technical Debt Boundaries

Hard controls are allowed only as guardrails.
Each such control must include:

- `TEMPORARY TECHNICAL DEBT` label,
- removal criterion,
- intended learned replacement.

---

## 12. Related Documents

- `docs/05_Eval_Metrics_Spec.md`
- `docs/06_Regression_and_Phase_Gates.md`
- `docs/07_Data_Constitution.md`
- `docs/09_Training_Profiles_and_Scaling.md`
