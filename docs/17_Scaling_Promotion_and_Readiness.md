# Scaling Promotion and Readiness

**Document Type:** Active Companion Authority  
**Scope:** Capability-readiness rules for promotion across `P1-P4`, readiness dimensions, and definitions of institution maturity for IRIS-math v2  
**Boundary:** This document governs promotion readiness. It does not replace the hardware/purpose profile family descriptions in `docs/09_Training_Profiles_and_Scaling.md` or the phase gates in `docs/06_Regression_and_Phase_Gates.md`.

---

## 0. Purpose and Positioning

Parameter count alone is not a promotion contract.

This document separates:

- **hardware/purpose profile families** from `docs/09`,
- **capability-readiness promotion** across those families.

Movement from `P1` to `P4` is justified only when the program is institutionally ready, not merely because a larger model is available.

---

## 1. Readiness Dimensions

Profile promotion is judged against these readiness dimensions:

1. **Benchmark governance alive**  
   Benchmark-family use is registered, tiered, and auditable.

2. **Contamination audit active**  
   Leakage checks, homologous split policy, and held-out integrity are functioning.

3. **Document robustness stable**  
   OCR/layout/document-grounded behavior remains useful under reformulation and modality shifts.

4. **Verifier evidence mature**  
   Local/global checks, counterexample probes, and formal bridges supply usable evidence.

5. **Provenance reproducible**  
   Parser, formalizer, verifier, and runtime provenance remain stable and replayable.

6. **Failure distribution stable**  
   Outcome improvements do not come from collapsed failure attribution or hidden regression.

Readiness claims should use metrics and tolerances already defined in `docs/05`, `docs/06`, `docs/08`, and `docs/09`.

---

## 2. Promotion Contract

### 2.1 `P1 -> P2`

Promotion is justified only if:

- the `P1` institution validator can run the complete governance loop,
- benchmark tiering is explicit and auditable,
- document parsing is materially useful rather than cosmetic,
- verifier signals are stable enough to influence recovery and evaluation,
- contamination controls are active and reproducible.

Blockers:

- unresolved leakage,
- missing provenance,
- verifier instability,
- document pipeline that does not survive basic reformulation pressure.

### 2.2 `P2 -> P3`

Promotion is justified only if:

- strategy diversity and theorem reuse improve without failure-credit collapse,
- document robustness survives harder OCR/layout and long-context pressure,
- formal or semi-formal signals produce usable verifier gains,
- benchmark gains transfer to homologous held-out surfaces,
- training governance remains reproducible at the larger envelope.

Blockers:

- benchmark lock-in,
- rising false accept with score gains,
- loss of provenance completeness,
- unstable long-context document behavior.

### 2.3 `P3 -> P4`

Promotion is justified only if:

- verifier maturity is strong enough for frontier-facing claims,
- strict held-out benchmark families remain genuinely informative,
- long-context proof and document behavior are stable under strong stress,
- contamination governance still works at the larger scale,
- failure distributions remain diagnosable rather than flattened.

Blockers:

- unverifiable frontier claims,
- strict held-out leakage uncertainty,
- scaling that outruns the verifier stack,
- growth in model size without corresponding institution maturity.

---

## 3. Institution Solved

`Institution solved` means the `P1` validator can stably run:

- the declared data constitution,
- benchmark tier disclosure,
- contamination audit,
- parser / formalizer / verifier provenance tracking,
- regression and phase-gate artifacts,
- verifier-conditioned validity and failure-credit reporting.

This is the minimum condition for treating the program as operationally real.

---

## 4. Verifier Mature Enough for Frontier Scaling

The verifier stack is mature enough for frontier scaling only when:

- local validity checks are reliable,
- global proof-gap detection is useful on hard tasks,
- contradiction or counterexample probes catch meaningful failure modes,
- formal bridge evidence can participate without breaking governance or throughput assumptions,
- false accept and false reject behavior are both understood well enough to support promotion claims.

If this condition is not met, frontier-scale promotion is premature even when larger hardware is available.

---

## 5. Relationship to Phases and Profiles

This document does not replace:

- `A-E` phase promotion in `docs/06_Regression_and_Phase_Gates.md`,
- hardware and purpose family descriptions in `docs/09_Training_Profiles_and_Scaling.md`.

Interpretation rule:

- `docs/06` answers whether the program is phase-ready,
- `docs/09` answers what a profile family is for,
- this document answers whether promotion to a larger profile family is justified.

---

## 6. Final Rule

No profile promotion claim is valid if it cannot explain readiness in terms of:

- benchmark governance,
- contamination audit,
- document robustness,
- verifier maturity,
- provenance reproducibility,
- stable failure distributions.
