# P1 Readiness Packet

- Profile: P1
- Baseline ID: p1-readiness-fixed-baseline
- Tolerance Profile ID: tp_p1_bootstrap
- Run Gate Status: **FAIL**
- Promotion Status: **BLOCKED**
- Consecutive Gate-Passed Runs: 0/3

## Hard-Gate Surfaces
- contam.strict_holdout_leakage_score: FAIL (current=None, baseline=None, delta=None) -> current value is unavailable; baseline value is unavailable
- eval.calibration_error: PASS (current=0.08, baseline=0.08, delta=0.0) -> within tolerance
- eval.false_accept_rate: PASS (current=0.05, baseline=0.05, delta=0.0) -> within tolerance
- failure.credit.collapse_rate: FAIL (current=0.14285714285714285, baseline=0.14285714285714285, delta=0.0) -> current value 0.142857 exceeds the <= 0.02 ceiling
- provenance.parser_coverage: PASS (current=1.0, baseline=1.0, delta=0.0) -> within tolerance
- provenance.verifier_coverage: PASS (current=1.0, baseline=1.0, delta=0.0) -> within tolerance
- rep.document.parse_completeness: FAIL (current=0.9299999999999999, baseline=0.9299999999999999, delta=0.0) -> current value 0.93 is below the >= 0.97 floor
- task.document_grounding_score: FAIL (current=0.95, baseline=0.95, delta=0.0) -> delta 0 is below the required +0.01 improvement

## Residual Blockers
- governed_training_run: Checkpoint schema is not iris.training_checkpoint/v2; readiness evidence remains synthetic or partial.
- strict_holdout_leakage_audit: No strict held-out leakage audit artifact was provided.
- sidecar_document_pipeline_debt: 1 strict held-out document fixtures still depend on sidecar-backed normalization.
