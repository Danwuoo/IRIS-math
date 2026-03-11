# Phase E Execution Design Note

**Document Type:** Design Note (Non-normative)  
**Normative Boundaries:** This note does not override `docs/01`~`docs/08`.  
**Archive Warning:** This note preserves baseline ARC/baseline-profile execution context. It contains legacy ARC-specific semantics and probe-only assumptions and must not be used as active authority for IRIS-math v2.

## 1. Scope

Phase E implementation adds:

- Pure LM 90% streaming data path (`pure_lm_90_v1` profile).
- Deterministic dataset slice planning for S8 replay alignment.
- `arc-agi-benchmarking` integration as a regression probe (not training data).
- A dedicated Phase E hard gate entrypoint (`scripts/phase_e_gate.py`).

Out of scope:

- No edits to RO architecture contracts (`docs/01`~`docs/04`).
- No edits to vendored benchmark internals under `tools/arc-agi-benchmarking/`.

## 2. Gate Flow

Phase E gate sequence:

1. Run S1/S2 suites.
2. Run/consume ARC benchmark probe artifact (`arc_benchmark_probe.json`).
3. Execute Phase D suite core (`S3`~`S8`) under `phase="E"` context.
4. Apply additional Phase E probe hard checks:
   - Probe status must be `PASS`.
   - Baseline non-regression must be true.
   - Baseline and IRIS scoring artifacts must exist.
5. Emit gate artifacts and markdown report.

Any blocking failure sets `regression.status=FAIL`.

## 3. Probe Artifact Schema

`arc_benchmark_probe.json` (`iris.arc_benchmark_probe/v1`) includes:

- `status`
- `probe_a_baseline.export`
- `probe_a_baseline.scoring`
- `probe_b_iris.export`
- `probe_b_iris.scoring`
- `baseline_non_regression`
- `block_reasons`

Phase E gate consumes this artifact as verifier evidence and records pass/fail under `suite_status.PhaseEProbe`.

## 4. Baseline Freeze Workflow

- Default baseline id: `phase-e-v1`.
- Default tolerance profile: `phase-e-default`.
- First baseline creation uses `--freeze-baseline`.
- Without `--freeze-baseline`, missing baseline artifacts are treated as blocking in hard-fail mode.

## 5. Hard-Fail Policy

`phase_e_gate.py` defaults to hard-fail semantics:

- Any `ON` suite failure blocks.
- Missing ARC probe artifact blocks.
- Missing benchmark scoring artifacts block.
- Baseline non-regression failure blocks.

## 6. Key Artifacts

Phase E run expects these artifacts at minimum:

- `summary_report.json`
- `resume_consistency_packet.json`
- `arc_benchmark_probe.json`
- `phase_e_gate_report.md`
- baseline diff artifacts from S3/S4/S5
