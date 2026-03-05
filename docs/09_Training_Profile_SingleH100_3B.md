# Training Profile: Single H100 3B

**Document Type:** Design Note Profile (Non-normative)  
**Scope:** Single-card baseline values for architecture validation and early pretraining  
**Governance boundary:** Transaction/resume/runtime-lock semantics live in `docs/08_Training_Run_Governance.md` (this profile binds fixed values only)  
**Source lineage:** Consolidated from legacy single-H100 planning docs (removed on 2026-02-27).

---

## 0. Change Declaration (Baseline)

- Change class: `Targeted fix`
- Target failure categories / signals:
  - `F_SEARCH` stability risks caused by long-run single-card budget pressure and noisy small-batch updates
  - `F_EVAL` / `eval.calibration_error` drift risks from unstable mixed-precision and resume behavior
- Expected impact:
  - No architectural contract changes
  - Better training-run reproducibility and resume consistency (`S8`)

---

## 1. Scope

Single-card profile for architecture validation and early pretraining:

- Hardware: `1x H100 80GB`
- Distributed strategy: none (`no DP`, `no TP`)
- Training mode: from-scratch pretraining
- Stack direction: `JAX + Flax NNX + Optax + Orbax`

This is not a multi-node production profile.

---

## 2. Fixed Engineering Decisions

| Category | Decision |
| --- | --- |
| Framework | `JAX` |
| Model API | `Flax NNX` (not Linen) |
| Model size target | `~3B parameters` |
| Compute precision | `BF16` compute on H100 |
| Master weights | `FP32` |
| LN / logits | `FP32` |
| Accumulators | `FP32` |
| Activation strategy | Block-level `jax.remat` from start (see `OD-08`) |
| Shape policy | Formal training uses single bucket `1024`; `512/1024` allowed for compile warmup only (see `OD-07`) |
| Main context length | `1024` |
| Microbatch | `2` |
| Gradient accumulation | Enabled |
| Cross-segment accumulation | Forbidden |
| Runtime lock policy | Lock manifest v1 required; Phase `C+` upgrades frozen unless baseline rebuild (see `OD-02`) |

---

## 3. Optimizer and LR Schedule

| Item | Value |
| --- | --- |
| Optimizer | `AdamW` |
| `beta1` | `0.9` |
| `beta2` | `0.95` |
| `eps` | `1e-8` |
| `weight_decay` | `0.1` |
| Gradient clipping | Global norm `1.0` |
| Peak LR | `3e-4` |
| Warmup | `2000 optimizer steps` |
| Decay | Cosine |
| Final LR | `3e-5` |

---

## 4. Effective Batch Design

| Item | Value |
| --- | --- |
| `seq_len` | `1024` |
| `microbatch` | `2` |
| Tokens per forward | `2048` |
| Target tokens per optimizer step | `~1,000,000` |
| Accumulation steps | `~488` (`2048 * 488 = 999,424`) |

Notes:

- Effective batch is achieved via accumulation only.
- Segment boundaries must not cut through an accumulation window.

---

## 5. Checkpoint/Resume Profile Bindings

Governance details live in `docs/08_Training_Run_Governance.md`. This section binds profile values:

| Item | Policy |
| --- | --- |
| Small checkpoint | Every `100` optimizer steps |
| Full checkpoint | Every `1000` optimizer steps |
| Segment sizing | Target `40` minutes; hard max `45` minutes; must align to optimizer-step boundary |
| Segment journal | Append-only |
| Segment semantics | Two-phase `PENDING -> APPLIED` |
| Segment journal schema | `iris.segment_journal/v1` append-only |
| Resume authority | Last `APPLIED` journal event (`PENDING` must replay) |
| Resume consistency | Must pass `S8` when activated by phase policy |
| Runtime lock manifest | Must record `runtime_lock_manifest_id` and `runtime_lock_manifest_sha256` in checkpoint metadata |
| Runtime mismatch policy | `strict` resume required; Phase `C+` forbids `unsafe_resume` |
| Segment-end evaluation | One eval snapshot at each segment end (lite metrics allowed) |
| Checkpoint retention | Three-tier: recent `5` segments + daily `1` + per-phase milestone `1` permanent |

---

## 6. 3B Topology Recommendation (Training Profile Layer)

| Parameter | Value |
| --- | --- |
| Layers | `28` |
| Hidden dim | `2560` |
| Attention heads | `20` |
| Head dim | `128` |
| FFN multiplier | `4` |
| Vocab | `100k` fixed, with protected IR/control sub-vocab |

---

## 7. Data Policy Binding

This profile assumes:

- Pure LM: `90%`
- IR-aligned synthetic: `10%`
- Benchmark datasets: `0%` in training (regression probe only)

Source of truth:

- `docs/07_Data_Mixture_and_Ingestion.md`

---

## 8. Technical Debt Guardrails

The following hard controls are temporary guardrails and must not become semantic policy:

1. Fixed single training bucket (`1024`)
2. Compile-warmup bucket restriction (`512/1024` only)
3. Hard clip (`global norm = 1.0`)

Removal criteria:

- Stable compile behavior and cache hit rate under planned dynamic-shape alternatives
- No regression in primary process gates:
  - `failure.credit.collapse_rate`
  - `eval.calibration_error`
  - `prog.diversity`
  - `search.termination_margin`

Intended learned replacement:

- Learned budget/control adaptation remains in L3/L6 policies; guardrails remain safety caps only.

---

## 9. Regression and Artifact Expectations

For architecture/training-impacting updates under this profile:

- Keep phase declaration explicit in artifacts (`phase = C|D|E` as applicable)
- Maintain baseline/tolerance profile integrity (`tolerance_profile_id` frozen at Phase `B -> C`, tightening-only afterward)
- Run activated suites per `docs/06_Regression_and_Phase_Gates.md`
- Block on any hard-gate violation

Eval cadence is fixed:

- Per segment: lite probes `S1/S2` + `S3` when active
- Every `24h` or `200` optimizer steps (whichever first): full activated suites
- Before phase gate: mandatory full regression
- `S8` run must include crash-injection coverage for `execute/pre-commit/post-commit`

---

## 10. Appendix A: Resolved Decision Register

| ID | Status | Fixed Value | Failure/Metric Risk |
| --- | --- | --- | --- |
| `OD-01` | closed | `tokenizer.vocab_size = 100000`; protected IR/control sub-vocab required; `rep.tokenizer.ir_fragmentation_rate` target `0.0` | `F_REP/F_PROC`, `S5/S6/S7`, `prog.diversity` |
| `OD-02` | closed | lock manifest v1 required (driver/CUDA/cuDNN/jaxlib build/XLA flags/env vars/packages); upgrades frozen after Phase `C` baseline unless baseline rebuild | `S1`, `S8` |
| `OD-03` | closed | segment boundary = optimizer-step boundary; no cross-accumulation; wall-clock target `40` min, hard max `45` min; one eval snapshot at each segment end | `S8`, journal replay |
| `OD-04` | closed | retention tiers: `recent=5 segments`, `daily=1`, `phase_milestone=1 permanent`; checkpoint must include model/optimizer/RNG/journal head/lock manifest id+sha | recovery reliability, ops risk, `S8` |
| `OD-05` | closed | per segment: lite probes `S1/S2` + `S3` when active; full activated suites every `24h` or `200` optimizer steps (whichever first); mandatory full regression before phase gate | drift detection latency |
| `OD-06` | closed | freeze tolerance profile at Phase `B -> C` promotion; tightening allowed, relaxation forbidden; epsilon persisted in versioned JSON artifact | gate ambiguity, false pass risk |
| `OD-07` | closed | debug/compile warmup may use `512/1024`; formal training uses single context bucket `1024`; bucket curriculum only in dedicated phase | distribution shift, compile cache churn |
| `OD-08` | closed | remat policy fixed to block-level through Phase `B`; remat changes require baseline checkpoint + full activated-suite regression + runtime manifest bump | `S1` numeric drift, `S8` resume drift |
| `OD-09` | closed | segment journal schema v1 is append-only; minimum fields include `run_id, segment_id, status, optimizer_step_id, dataset_slice_id, rng_hash_pre/post, loss_hash, grad_stats_hash, code/config hash, runtime_lock_manifest_sha256` | replay ambiguity, double-apply risk, `S8` attribution gaps |
| `OD-10` | closed | checkpoint commit protocol fixed: `PENDING append -> execute -> apply-in-memory -> temp write -> fsync -> atomic rename -> APPLIED append`; resume authority is last `APPLIED` journal event | crash consistency, exactly-once violations |
| `OD-11` | closed | RNG and data replay governance fixed: behavior-affecting RNG keys are checkpointed and committed only on `APPLIED`; deterministic data mapping key is `(run_id, dataset_slice_id, segment_id, micro_step_idx, data_seed)` | hidden distribution drift, non-replayable segments |
| `OD-12` | closed | `S8` requires crash-injection classes (`execute`, `pre-commit`, `post-commit`) plus drift diagnosis labels (`runtime/rng/data/optimizer`) | weak resume diagnostics, slow root-cause isolation |

---

## 11. Appendix B: Profile Change Declaration Template

Use this template for any profile delta:

```text
change_class = pure_refactor | targeted_fix | capability_expansion
target_failure_categories = ...
expected_metric_impact = ...
phase = A|B|C|D|E
tolerance_profile_id = ...
runtime_lock_manifest_delta = yes|no
regression_suites_to_run = S1..S8 (activated subset)
technical_debt_guardrails = none | listed_with_removal_criteria
```

---

## 12. Appendix C: Closure Records (Resolved)

```text
[OD-01] status=closed
decision=tokenizer.vocab_size=100000; protected_ir_control_sub_vocab=required; rep.tokenizer.ir_fragmentation_rate_target=0.0
effective_date=2026-02-26
phase=C
expected_metric_impact=lower F_REP/F_PROC risk from control-token fragmentation; improved prog.diversity stability
regression_artifacts=S5,S6,S7 diffs with tokenizer.vocab_size metadata

[OD-02] status=closed
decision=runtime_lock_manifest_v1_required; phase_c_plus_upgrade_policy=frozen_unless_baseline_rebuild; checkpoint_metadata_add=[runtime_lock_manifest_id,runtime_lock_manifest_sha256]
effective_date=2026-02-26
phase=C
expected_metric_impact=lower resume drift and numeric drift risk
regression_artifacts=S1,S8 before/after aligned reports

[OD-03] status=closed
decision=segment_boundary=optimizer_step; cross_segment_accumulation=forbidden; segment_wall_clock_target_min=40; segment_wall_clock_hard_max_min=45; segment_end_eval_snapshot=required
effective_date=2026-02-26
phase=C
expected_metric_impact=improved replay determinism and failure localization
regression_artifacts=S8 segment-aligned comparisons

[OD-04] status=closed
decision=checkpoint_retention=[recent:5_segments,daily:1,phase_milestone:1_permanent]; checkpoint_minimum_payload=[model,optimizer,rng,journal_head,runtime_lock_manifest_ref]
effective_date=2026-02-26
phase=C
expected_metric_impact=improved crash recovery robustness without single-point rollback risk
regression_artifacts=retention policy logs + S8 resume checks

[OD-05] status=closed
decision=probe_lite_per_segment=[S1,S2,S3_if_active]; full_regression_interval=[24h_or_200_optimizer_steps_whichever_first]; pre_phase_gate_full_regression=mandatory
effective_date=2026-02-26
phase=C
expected_metric_impact=faster structural drift detection with bounded eval cost
regression_artifacts=scheduled probe logs + full suite reports

[OD-06] status=closed
decision=tolerance_profile_freeze_point=phase_B_to_C_promotion; tolerance_relaxation=forbidden; tightening=allowed; epsilon_artifact=versioned_json
effective_date=2026-02-26
phase=B/C
expected_metric_impact=prevents chronic drift masking and gate ambiguity
regression_artifacts=tolerance profile JSON + phase declaration metadata

[OD-07] status=closed
decision=training_context_bucket=1024_single; warmup_buckets=[512,1024]; mixed_bucket_curriculum=separate_phase_only
effective_date=2026-02-26
phase=C
expected_metric_impact=lower hidden distribution shift from mixed bucket usage
regression_artifacts=S2/S3/S7 comparisons under fixed bucket

[OD-08] status=closed
decision=remat_granularity=block_level_through_phase_B; remat_change_requires=[baseline_checkpoint,full_activated_suite_regression,runtime_manifest_bump]
effective_date=2026-02-26
phase=A/B/C
expected_metric_impact=controls S1 numeric order drift and S8 resume inconsistency
regression_artifacts=S1,S8 regressions across remat change boundary

[OD-09] status=closed
decision=segment_journal_schema=iris.segment_journal/v1_append_only; required_fields=[run_id,segment_id,status,optimizer_step_id,dataset_slice_id,rng_hash_pre,rng_hash_post,loss_hash,grad_stats_hash,code_version_hash,config_hash,runtime_lock_manifest_sha256]
effective_date=2026-02-27
phase=C
expected_metric_impact=improves replay auditability and prevents silent apply-count ambiguity
regression_artifacts=S8 replay logs + journal integrity report

[OD-10] status=closed
decision=checkpoint_commit_order=[pending_append,execute,apply_in_memory,temp_write,fsync,atomic_rename,applied_append]; resume_authority=last_applied_journal_event
effective_date=2026-02-27
phase=C
expected_metric_impact=prevents duplicate apply and partial-commit recovery errors
regression_artifacts=S8 crash-injection matrix + checkpoint integrity checks

[OD-11] status=closed
decision=rng_commit=apply_only; behavior_affecting_rng_checkpoint=required; data_replay_key=[run_id,dataset_slice_id,segment_id,micro_step_idx,data_seed]
effective_date=2026-02-27
phase=C
expected_metric_impact=reduces resumed-path failure-distribution drift
regression_artifacts=S8 rng/data drift diagnostics + deterministic replay probes

[OD-12] status=closed
decision=s8_crash_classes=[execute,pre_commit,post_commit]; s8_drift_labels=[runtime_drift,rng_drift,data_slice_drift,optimizer_state_drift]
effective_date=2026-02-27
phase=C/D/E
expected_metric_impact=faster and attributable resume inconsistency triage
regression_artifacts=segment-aligned S8 reports with drift-label breakdown
```

---

## 13. Related Documents

- `docs/08_Training_Run_Governance.md`
- `docs/07_Data_Mixture_and_Ingestion.md`
- `docs/06_Regression_and_Phase_Gates.md`
- `docs/05_Eval_Metrics_Spec.md`
