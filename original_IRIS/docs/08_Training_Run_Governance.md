# Training Run Governance

**Document Type:** Engineering Governance (Non-normative but Policy-Binding)  
**Scope:** Segment transactions, exactly-once resume, reproducibility controls, runtime lock enforcement, and S8 drift diagnostics  
**Related canonical docs:** `docs/06_Regression_and_Phase_Gates.md`, `docs/05_Eval_Metrics_Spec.md`, `docs/07_Data_Mixture_and_Ingestion.md`, `docs/09_Training_Profile_SingleH100_3B.md`  
**Source lineage:** Consolidated from legacy training governance notes/specs (removed on 2026-02-27).

---

## 0. Positioning and Authority

- This document is policy-binding for training operations and reproducibility.
- It does not override system invariants, State IR contracts, trunk contract, Level contracts, or the learnable-control contract.
- Where a rule requires a hard cap (e.g., max wall-clock), it is a guardrail only. If it affects semantics, it must be explicitly labeled `TEMPORARY TECHNICAL DEBT` with a removal criterion and intended learned replacement path.

---

## 1. Purpose

Define run-time governance for long training:

- Segment transaction semantics
- Exactly-once training progress (`no duplicate apply`, `no missed apply`)
- Runtime lock compliance
- Drift diagnosis and artifact requirements (especially `S8`)

---

## 2. Pretraining-First Governance Priorities

Pretraining is optimized for stable internal behavior distributions:

- Failure taxonomy observability and stability
- L6 credit routing quality (no collapse)
- Calibration quality (no degradation)
- Program diversity at L2
- Recovery signal quality through the L3/L6 loop

Outcome and cost metrics are secondary probes:

- Task success / benchmark score
- Cost and throughput

Secondary improvement cannot justify regression in primary process signals (see `docs/05_Eval_Metrics_Spec.md` and `docs/06_Regression_and_Phase_Gates.md`).

Benchmark boundaries:

- Benchmark datasets are regression probes, not training mixture components (see `docs/07_Data_Mixture_and_Ingestion.md`).
- Tools/benchmarks must not provide runtime truth authority, routing policy, or control decisions.

---

## 3. Definitions

**Training Segment**  
An atomic training unit that groups a fixed data slice, its traces, and its metric emissions.

**Segment Status**

- `PENDING`: execute path completed, apply not committed
- `APPLIED`: optimizer/state update committed

A segment is either fully applied or not applied at all.

**Micro Step**  
One forward/backward pass on a microbatch.

**Optimizer Step**  
One applied update after a full accumulation window.

---

## 4. Segment Transaction Model (Execute / Apply)

### 4.1 Segment State Model

- `PENDING`: execute completed, apply not committed.
- `APPLIED`: model/optimizer/RNG and behavior-affecting state committed.

A segment is either fully applied or replayed.

### 4.2 Execute vs Apply

**Execute (interruptible)**:

- Forward/rollout/verifier/loss/grad accumulation
- Consumes RNG
- Must not mutate committed global training state

**Apply (atomic intent)**:

- Update model weights and optimizer state
- Commit full RNG tree
- Commit behavior-affecting memory/macro state

### 4.3 Accumulation Boundary Rules

1. Accumulation windows must be fully contained inside a single segment.
2. Cross-segment accumulation is forbidden.
3. A segment boundary must align to optimizer-step boundary.
4. Segment completion is recognized only after `APPLIED`.

If interruption occurs before apply, the segment remains `PENDING` and must be replayed.

---

## 5. Journal and Exactly-Once Rules

Journal schema: `iris.segment_journal/v1` (append-only).

Storage:

- JSONL append-only file or append-only table
- No in-place overwrite for status transitions

### 5.1 Minimum Journal Record Fields

| Field | Type | Notes |
| --- | --- | --- |
| `schema` | string | `iris.segment_journal/v1` |
| `event_id` | int | Monotonic per run |
| `event_time` | string | ISO-8601 UTC |
| `run_id` | string | Immutable run identity |
| `segment_id` | int | Monotonic segment index |
| `status` | enum | `PENDING` or `APPLIED` |
| `optimizer_step_id` | int | Target optimizer step |
| `dataset_slice_id` | string | Deterministic data window id |
| `rng_hash_pre` | string | Hash of RNG tree at segment entry |
| `rng_hash_post` | string/null | Null for `PENDING`, required for `APPLIED` |
| `loss_hash` | string | Hash of segment loss summary |
| `grad_stats_hash` | string | Hash of grad summary for audit |
| `code_version_hash` | string | Code provenance |
| `config_hash` | string | Full config provenance |
| `runtime_lock_manifest_sha256` | string | Runtime provenance |
| `checkpoint_ref` | string/null | Required for `APPLIED` |

### 5.2 Authority Rules

- Resume truth source is `last APPLIED event`.
- `PENDING` indicates replay required.
- Checkpoints without matching `APPLIED` event are quarantined as non-authoritative.

---

## 6. Checkpoint Atomicity and Commit Order

Checkpoint backend requirement:

- Orbax/TensorStore path must follow `temp write -> fsync -> atomic rename`

Commit protocol (idempotent):

1. Append `PENDING` journal event for `segment_id`
2. Run `execute`
3. Build `apply` candidate state in memory
4. Write checkpoint to temp path (post-apply state)
5. `fsync` checkpoint payload and metadata
6. Atomic rename temp path to final checkpoint path
7. `fsync` parent directory metadata
8. Append `APPLIED` journal event (same `segment_id`) with `checkpoint_ref`

Resume reconciliation:

- If last journal status for newest segment is `PENDING`: replay segment, ignore newer uncommitted checkpoint artifacts
- If last journal status is `APPLIED`: restore from referenced checkpoint
- Stray checkpoints without matching `APPLIED` event are non-authoritative

### 6.1 Minimum Checkpoint Payload (Required)

- Model weights
- Optimizer state
- Full RNG tree
- Behavior-affecting memory/macro stats
- `segment_id_last_applied`
- `optimizer_step_id_last_applied`
- `dataset_slice_id_last_applied`
- `runtime_lock_manifest_id`
- `runtime_lock_manifest_sha256`
- `journal_head_event_id`
- `journal_head_hash`
- `code_version_hash`
- `config_hash`

---

## 7. Behavior-Affecting RNG Governance

RNG is committed state and must be checkpointed/restored exactly.

Rules:

- Global RNG write-back during `execute` is forbidden.
- RNG tree hash pre/post must be journaled.
- Resume must restore full RNG tree, not only a seed scalar.

### 7.1 RNG Ownership Table (v1)

| Subsystem | Key Path | Consumed In | Commit Rule |
| --- | --- | --- | --- |
| Model stochastic ops (dropout, etc.) | `rng.model.*` | `execute` | Promote post-key only on `APPLIED` |
| Routing/gating sampling | `rng.control.*` | `execute` | Promote post-key only on `APPLIED` |
| Data shuffle/sampling | `rng.data.*` | `execute` | Promote post-key only on `APPLIED` |
| Synthetic task generation | `rng.synthetic.*` | `execute` | Promote post-key only on `APPLIED` |
| Verifier perturbation/probes | `rng.verifier.*` | `execute` | Promote post-key only on `APPLIED` |

---

## 8. Data Slice Determinism Spec

Data must be replayable by identity, not by iterator position.

Required keys:

- `run_id`
- `segment_id`
- `micro_step_idx`
- `dataset_slice_id`
- `data_seed`

Deterministic mapping rule:

```text
batch = SelectBatch(run_id, dataset_slice_id, segment_id, micro_step_idx, data_seed)
```

Operational constraints:

- Segment definition binds a fixed data slice.
- “next batch” semantics are invalid for resumable segments.
- Shuffle must derive only from deterministic keys above.
- `dataset_slice_id` and `data_seed` must be present in checkpoint metadata.

---

## 9. Runtime Lock Policy

Runtime lock manifest is required and pinned per validated set.

### 9.1 Lock Surfaces

The following must be pinned as one tested set:

System / driver surface (must be recorded):

- GPU model
- NVIDIA driver version
- CUDA runtime version
- cuDNN version (if present/used by runtime)
- OS + kernel (or container image id)

Python + JAX surface (must be pinned):

- `python`
- `jax`
- `jaxlib` (CUDA-compatible build; record build tag/hash)
- `flax` (NNX API)
- `optax`
- `orbax-checkpoint`
- `numpy`

Optional but recommended pins:

- `ml_dtypes`
- `tensorstore` (if checkpoint backend requires it)

Compilation / XLA surface (must be recorded):

- `XLA_FLAGS` (full string)
- JAX/XLA env vars that can affect numerics/compilation (record full key/value pairs), e.g.:
  - `JAX_ENABLE_X64`, `JAX_DEFAULT_MATMUL_PRECISION`, `JAX_DISABLE_JIT`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_PREALLOCATE`

### 9.2 Resume Validation Policy

- `strict` (default): manifest id and sha must match active runtime lock
- `unsafe_resume` (explicit): allowed only before Phase C baseline freeze; must be logged as technical debt artifact
- Phase C+ baseline rule: `unsafe_resume` is disallowed

### 9.3 Upgrade Policy

- No ad-hoc package bump in active training runs.
- Upgrade only as a full tested set (never single-package drift).
- Phase C baseline freeze: **no upgrades** unless treated as a baseline rebuild with full activated-suite regression artifacts.
- Resume from checkpoint requires manifest id+sha match; mismatch is rejected in Phase C+.

### 9.4 Runtime Lock Manifest Format (v1, JSON)

Store a single manifest artifact per validated stack. Minimum required fields:

```json
{
  "schema": "iris.runtime_lock_manifest/v1",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "phase": "A|B|C|D|E",
  "host": {
    "os": "...",
    "kernel": "...",
    "gpu": "...",
    "nvidia_driver": "...",
    "cuda_runtime": "...",
    "cudnn": "..."
  },
  "python": { "version": "...", "packages": [{ "name": "jax", "version": "...", "hash": "..." }] },
  "jax": {
    "jax": "...",
    "jaxlib": "...",
    "jaxlib_build": "...",
    "xla_flags": "...",
    "env": { "JAX_ENABLE_X64": "...", "JAX_DEFAULT_MATMUL_PRECISION": "..." }
  }
}
```

### 9.5 Checkpoint Metadata Requirement

Every checkpoint must record:

- `runtime_lock_manifest_id`
- `runtime_lock_manifest_sha256`

This is required for `S8` resume investigations and to prevent silent “same checkpoint, different runtime” drift.

---

## 10. S8 Resume Consistency Alignment

When S8 is active, validate crash classes:

1. Crash during `execute` (before apply)
2. Crash after `execute` and before checkpoint commit (`pre-commit`)
3. Crash after checkpoint commit (`post-commit`) with journal reconciliation

Segment-aligned comparisons should use:

- `task.validity_score`
- `task.confidence`
- `failure.credit`

S8 failure diagnosis must classify suspected source:

- `runtime_drift`
- `rng_drift`
- `data_slice_drift`
- `optimizer_state_drift`

Cadence binding (profile-specific): follow `docs/09_Training_Profile_SingleH100_3B.md`.

---

## 11. Technical Debt Boundaries

Allowed hard controls are guardrails only (for example max segment wall-clock, max recovery retries).

If a hard control is introduced, it must include:

- `TEMPORARY TECHNICAL DEBT` label
- Removal criterion
- Intended learned replacement path

---

## 12. 不確定 Items

- 不確定: Some JAX/XLA ops may still show minor numeric variation across driver/XLA minor versions.
- Mitigation: strict runtime lock manifest + fixed XLA/env surfaces + S8 drift gates.
- Default policy: do not treat cross-manifest resume as valid baseline evidence.

---

## 13. Related Documents

- `docs/06_Regression_and_Phase_Gates.md`
- `docs/05_Eval_Metrics_Spec.md`
- `docs/07_Data_Mixture_and_Ingestion.md`
- `docs/09_Training_Profile_SingleH100_3B.md`
