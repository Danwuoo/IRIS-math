# Phase C Gate Report (Strict JAX)

- Document Type: Design Note (Non-normative)
- Generated At (UTC): 2026-03-13T13:59:20.232436Z
- Phase: C
- Baseline ID: toy-baseline
- Tolerance Profile ID: toy-default
- Change Class: Capability expansion (IRIS-math v2 documentation-first transition)
- Overall Regression Status: **FAIL**

## 1) Suite Status
- S1: FAIL
- S2: FAIL
- S3: PASS
- S4: FAIL
- S5: FAIL
- S6: PASS
- S7: PASS
- S8: PASS
- S8_h100_packet: PASS

## 2) Blocking Violations
- [S1] smoke runtime checks: Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s1_smoke.py", line 15, in <module>
    from iris.trunk import SingleTrunk
  File "C:\Users\wurre\Desktop\IRIS-math\src\iris\trunk\__init__.py", line 1, in <module>
    from .single_trunk import (
  File "C:\Users\wurre\Desktop\IRIS-math\src\iris\trunk\single_trunk.py", line 6, in <module>
    import jax
ModuleNotFoundError: No module named 'jax'
- [S2] structural contract checks: Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_structural.py", line 84, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_structural.py", line 20, in main
    state = _build_state(hidden_dim=hidden_dim)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_structural.py", line 15, in _build_state
    return StateIR(T=zeros(1), G=zeros(1), O=zeros(1), R=zeros(0), X=zeros(0), M=zeros(0))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: StateIR.__init__() got an unexpected keyword argument 'T'; Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\src\iris\runtime\jax_runtime.py", line 8, in assert_jax_runtime
    import flax  # noqa: F401
    ^^^^^^^^^^^
ModuleNotFoundError: No module named 'flax'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_mounted.py", line 94, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_mounted.py", line 37, in main
    assert_jax_runtime(
  File "C:\Users\wurre\Desktop\IRIS-math\src\iris\runtime\jax_runtime.py", line 12, in assert_jax_runtime
    raise RuntimeError(
RuntimeError: Strict JAX runtime required. Install jax/flax/optax before running this command.; Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_structural.py", line 84, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_structural.py", line 20, in main
    state = _build_state(hidden_dim=hidden_dim)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_structural.py", line 15, in _build_state
    return StateIR(T=zeros(1), G=zeros(1), O=zeros(1), R=zeros(0), X=zeros(0), M=zeros(0))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: StateIR.__init__() got an unexpected keyword argument 'T'; Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\src\iris\runtime\jax_runtime.py", line 8, in assert_jax_runtime
    import flax  # noqa: F401
    ^^^^^^^^^^^
ModuleNotFoundError: No module named 'flax'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_mounted.py", line 94, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "C:\Users\wurre\Desktop\IRIS-math\scripts\s2_mounted.py", line 37, in main
    assert_jax_runtime(
  File "C:\Users\wurre\Desktop\IRIS-math\src\iris\runtime\jax_runtime.py", line 12, in assert_jax_runtime
    raise RuntimeError(
RuntimeError: Strict JAX runtime required. Install jax/flax/optax before running this command.
- [S4] concept.isolation_score / concept.leakage_score (baseline+tolerance): 16 concept-level tolerance violations detected
- [S5] paired.invariance.gap / asymmetry rate (baseline+tolerance): 2 paired-representation tolerance violations detected

## 3) S8 Crash-Class Coverage Matrix (Local Packet)
| Path | Coverage | Status | Runtime Lock ID |
| --- | --- | --- | --- |
| uninterrupted | Yes | PASS | 605d19d5dc97 |
| execute_crash | Yes | PASS | 605d19d5dc97 |
| pre_commit_crash | Yes | PASS | 605d19d5dc97 |
| post_commit_crash | Yes | PASS | 605d19d5dc97 |

## 4) S8 Drift Diagnosis Summary (vs uninterrupted)
- uninterrupted: runtime_drift=False, rng_drift=False, data_slice_drift=False, optimizer_state_drift=False, failure_credit_l1=0.0
- execute_crash: runtime_drift=False, rng_drift=False, data_slice_drift=False, optimizer_state_drift=False, failure_credit_l1=0.0
- pre_commit_crash: runtime_drift=False, rng_drift=False, data_slice_drift=False, optimizer_state_drift=False, failure_credit_l1=0.0
- post_commit_crash: runtime_drift=False, rng_drift=False, data_slice_drift=False, optimizer_state_drift=False, failure_credit_l1=0.0

## 5) H100 Packet Status
- S8 status for H100 packet: **PASS**
- Coverage: uninterrupted=True, execute_crash=True, pre_commit_crash=True, post_commit_crash=True

## 6) Notes
- S8 local packet drift_clear=True
- S8 h100 packet status=PASS
- strict_suite_exec=True
- reuse_existing_suite_artifacts=False

## 7) Completion Checklist
- Mandatory docs consulted: `docs/00_INDEX.md`, `docs/10_Glossary_and_Normative_Status.md`, `docs/13_Goals_and_Success_Criteria.md`, `docs/07_Data_Constitution.md`, `docs/01_Architecture_Constitution.md`, `docs/02_State_IR_Spec.md`, `docs/03_Level_Contracts_L0-L6.md`, `docs/04_Credit_Assignment_and_Recovery.md`, `docs/18_Optimization_and_Learning_Contract.md`, `docs/19_Runtime_and_Task_Adjudication_Semantics.md`, `docs/05_Eval_Metrics_Spec.md`, `docs/06_Regression_and_Phase_Gates.md`, `docs/08_Training_Run_Governance.md`, `docs/09_Training_Profiles_and_Scaling.md`, `docs/14_Multimodal_Document_Pipeline.md`, `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`, `docs/16_Verifier_and_Formalization_Stack.md`, `docs/17_Scaling_Promotion_and_Readiness.md`
- Change class: `Capability expansion (IRIS-math v2 documentation-first transition)`
- Expected failure-category impact: Targeted closure for Phase C gate blocking failures.
- Technical debt guardrails introduced: none
- Termination: `Blocked`
