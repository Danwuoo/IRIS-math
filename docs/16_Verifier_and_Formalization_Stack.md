# Verifier and Formalization Stack

**Document Type:** Active Companion Authority  
**Scope:** Verifier stack layers, evidence classes, training-time versus eval-time use, false accept / false reject policy, and interfaces to `L3/L4/L6`  
**Boundary:** This document defines verifier roles and evidence semantics. It does not lock one backend mix across neural, symbolic, or formal tools.

---

## 0. Purpose and Positioning

IRIS-math v2 is verifier-centered, but verifier importance must not stay implicit or scattered.

This document defines:

- what the verifier stack is composed of,
- how verifier evidence is classified,
- how training-time labels differ from eval-time evidence,
- how false accept and false reject cases are handled,
- how verifier outputs interface with recovery, retrieval, and credit routing.

---

## 1. Verifier Stack Layers

The active verifier stack is organized into four layers.

| Layer | Role | Typical Outputs |
| --- | --- | --- |
| Local validity / step checking | Check local transitions, algebraic legality, type/scope consistency, and immediate rule use | step-validity signals, local contradictions, missing-justification markers |
| Proof-gap / obligation tracking | Detect unresolved obligations, hidden assumptions, unsupported leaps, and incomplete branches | gap scores, obligation lists, hidden-assumption risk |
| Contradiction / counterexample probes | Search for contradictions, edge-case failures, counterexamples, or branch inconsistencies | counterexample candidates, contradiction hits, branch-risk summaries |
| Formal checker bridge | Connect natural or semi-formal content to formal or near-formal checking surfaces | formal-check verdicts, translation confidence, proof-bridge evidence |

No single layer is sufficient on its own to define verifier maturity.

### 1.1 Bootstrap Maturity Ladder (`P1-P3`)

Verifier maturity is phase- and profile-dependent.
The current bootstrap expectation is:

| Profile | Local Validity | Gap Tracking | Counterexample Probe | Formal Bridge |
| --- | --- | --- | --- | --- |
| `P1` | mounted step-level local validity checker or equivalent step-aligned head | mounted gap / obligation tracker | lightweight symbolic-first probe | `partial_mount` only; semi-formal skeleton allowed |
| `P2` | mounted and regression-tracked | mounted and used for targeted recovery | hybrid neural proposal plus checker is allowed | narrow-domain formal bridge may be mounted |
| `P3` | mounted and promotion-relevant | mounted and promotion-relevant | hybrid probe expected on active proof-bearing eval surfaces | formal checker bridge enters official eval surface |

Rules:

1. `P1` is not allowed to claim verifier maturity if local validity and gap tracking remain pure stubs.
2. `P1` false accept mitigation should prefer stronger symbolic or checker-backed filtering over more permissive thresholds.
3. `P3` readiness is not met if formal bridge evidence exists only as an unused side artifact.

### 1.2 Mounted vs `partial_mount` vs Stub

- `mounted`: the verifier surface emits auditable evidence that is consumed by training or eval-time control
- `partial_mount`: the surface runs on a bounded slice, emits provenance-bearing evidence, but is not yet a full blocking surface
- `stub`: interface exists but produces no evidence useful for attribution or promotion

`partial_mount` is allowed for the formal bridge in `P1`, but it does not satisfy frontier-facing readiness on its own.

---

## 2. Evidence Classes

Verifier evidence should be logged and reasoned about by class:

1. **Local validity evidence**  
   Evidence that a step or local transformation is legal.

2. **Gap evidence**  
   Evidence that a proof or derivation still lacks support.

3. **Counterexample evidence**  
   Evidence that a claim or branch fails under contradiction or adversarial probing.

4. **Formal bridge evidence**  
   Evidence produced when a statement or proof segment survives a formal or semi-formal checking bridge.

Evidence should also carry:

- provenance,
- coverage scope,
- strength/confidence,
- whether it is positive, negative, or mixed evidence.

---

## 3. Training-Time Labels vs Eval-Time Evidence

### 3.1 Training-Time Verifier Labels

Training-time labels may be used to supervise:

- local validity classifiers,
- proof-gap detectors,
- counterexample contrast pairs,
- natural-to-formal alignment,
- recovery targeting.

Training-time labels are allowed to be partial, noisy, or weakly supervised if:

- provenance is explicit,
- label quality is declared,
- the labels are not misrepresented as final proof truth.

### 3.2 Eval-Time Verifier Evidence

Eval-time verifier evidence is stricter.

Its job is to support:

- final validity judgments,
- confidence calibration,
- failure-credit routing,
- regression gates,
- readiness and promotion claims.

Eval-time evidence must therefore be auditable and stable enough to survive adversarial or reformulated cases.

Training labels and eval evidence must not be conflated.

---

## 4. False Accept / False Reject Policy

### 4.1 False Accept

False accept means invalid reasoning or output is accepted as valid.

Policy:

- treat persistent false accept as a top-priority blocker,
- route credit primarily through `L6`, while preserving lower-level blame when upstream defects caused the acceptance,
- strengthen contradiction / counterexample / formal-bridge checks before relaxing thresholds.

False accept is more dangerous than low-confidence abstention because it corrupts downstream learning and promotion claims.

### 4.2 False Reject

False reject means valid reasoning or output is rejected.

Policy:

- investigate missing coverage, weak bridge alignment, or overly conservative local/global checks,
- preserve the distinction between "insufficient evidence" and "evidence of invalidity",
- allow recovery or stronger verification passes when justified.

False reject matters, but it does not justify hiding false accept problems.

---

## 5. Interfaces to Levels

### 5.1 Interface to `L3`

`L3` consumes verifier evidence to decide:

- whether to continue, backtrack, or switch strategy,
- whether recovery should target parse, structuring, strategy, memory, or abstraction,
- whether more budget is warranted.

`L3` should not reduce verifier evidence to a single hard-coded retry policy.

### 5.2 Interface to `L4`

`L4` uses verifier evidence to assess:

- whether retrieved lemmas really apply,
- whether a memory hit is unsafe,
- whether document-local facts should dominate over a tempting retrieval.

Verifier evidence should strengthen applicability audit, not replace it.

### 5.3 Interface to `L6`

`L6` aggregates verifier evidence into:

- validity judgments,
- confidence and calibration,
- `failure.credit`,
- false accept / false reject accounting.

`L6` is the canonical emitter of outcome-side verifier interpretation.

---

## 6. Maturity Signals

The verifier stack is considered materially mature only when:

- local validity signals are stable,
- proof-gap detection can expose hidden assumptions without collapsing into noise,
- counterexample probes find real failures often enough to be useful,
- formal bridge evidence can be consumed without destroying throughput or provenance discipline,
- false accept and false reject behavior are both measurable and actionable.

This maturity is a prerequisite for frontier-facing scaling claims.

---

## 7. Explicit Non-Goals

This document does not:

- mandate one theorem prover,
- require every output to be fully formalized,
- freeze the neural/symbolic/formal backend ratio,
- turn the verifier into the sole intelligence substrate.

---

## 8. Related Documents

- `docs/04_Credit_Assignment_and_Recovery.md`
- `docs/05_Eval_Metrics_Spec.md`
- `docs/06_Regression_and_Phase_Gates.md`
- `docs/17_Scaling_Promotion_and_Readiness.md`
