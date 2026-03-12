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

### 2.4 Consolidated Profile Gating Table

This table is an operational shorthand for the active contracts in `docs/05`, `docs/06`, `docs/09`, `docs/13`, `docs/15`, and `docs/16`.
It does not replace metric names, family-specific benchmark rules, or phase-suite activation rules.

Interpretation rules:

- Treat a profile as gate-passed only when the relevant active suites in `docs/06_Regression_and_Phase_Gates.md` pass under a fixed declared `baseline_id` and `tolerance_profile_id`.
- Treat the profile-specific hard-gate metric bundles in `docs/05_Eval_Metrics_Spec.md` as the default surface sets for readiness review unless a stricter packet is declared.
- Treat `tp_p1_bootstrap` through `tp_p4_bootstrap` in `docs/05_Eval_Metrics_Spec.md` as the default numeric tolerance profiles for routine promotion review unless a stricter profile is declared.
- Treat benchmark rows below as posture summaries, not exceptions to the family-specific Tier 1 / Tier 2 / Tier 3 firewalls in `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`.
- Treat outcome claims as invalid if they violate the success-model red lines in `docs/13_Goals_and_Success_Criteria.md`.

| Profile | Minimum Institution State | Capability Gates That Must Be Alive | Benchmark Posture | Claim Boundary |
| --- | --- | --- | --- | --- |
| `P1` | `institution solved` loop is real: data constitution, benchmark tier disclosure, contamination audit, parser / formalizer / verifier provenance, regression artifacts, and `failure.credit` reporting all run stably | document parsing is materially useful rather than cosmetic; local validity and gap tracking are mounted enough to influence recovery or evaluation; false accept is not worsening; document grounding and provenance coverage remain stable | `AIMO`, `Omni-MATH`, and `miniF2F` may appear only through their registered tier policies; Tier 2 remains observe-only during tuning; original `FrontierMath` remains untouched strict held-out | may claim institution validity and useful document / verifier loop closure; must not claim frontier competitiveness, `AIMO private` strength, or final scaling recipe validity |
| `P2` | `P1` institutions remain stable at the larger envelope; homologous split, paired reformulation, and benchmark disclosure machinery stay reproducible | strategy diversity, theorem reuse, OCR / layout robustness, partial formalization gains, and counterexample-probe usefulness improve without `failure.credit` collapse | Tier 2 aggregate gains must accompany any Tier 1-visible family gains; benchmark lock-in is a blocker; `FrontierMath` is still original-family strict held-out, with only separately registered derivative families allowed for train-visible use | may claim benchmark-adjacent generalization and partial formalization gains; must not claim `AIMO private-like` readiness or original `FrontierMath` strength |
| `P3` | full governance loop survives harder scale: `S1-S8` artifacts, resume consistency, contamination audit, and provenance completeness remain stable | long-context proof handling, verifier-memory cooperation, and search / recovery gains are visible on hard tasks; formal bridge evidence is promotion-relevant; false accept remains controlled | `AIMO` private-like or post-cutoff held-out surfaces become meaningful only as post-lock evaluation; `Omni-MATH` and `miniF2F` must improve on held-out as well as homologous surfaces; original `FrontierMath` remains strict held-out | may claim frontier-readiness signals and non-trivial high-difficulty math competence; must not treat early frontier signals alone as sufficient justification for `120B` claims |
| `P4` | institution maturity survives frontier scale: benchmark governance, contamination audit, document robustness, verifier maturity, provenance reproducibility, and stable failure distributions all remain informative | verifier stack is mature enough for frontier claims; long-context document and proof robustness remain stable under strong stress; false accept / false reject accounting stays actionable | only here may strict held-out frontier surfaces support headline outcome claims; `AIMO private`, `Omni-MATH`, `miniF2F`, and original `FrontierMath` still require homologous held-out support plus verifier evidence | this is the first profile where final outcome-facing goals such as `AIMO 3 private: 50/50` may be claimed, and only when they remain aligned with capability and system evidence |

### 2.5 Default Promotion Packet

Unless a stricter program rule is declared, a routine `P1 -> P2`, `P2 -> P3`, or `P3 -> P4` promotion packet should include:

1. three consecutive gate-passed runs under the same declared `baseline_id` and `tolerance_profile_id`,
2. the relevant hard-gate surface bundle from `docs/05_Eval_Metrics_Spec.md`,
3. the matching bootstrap tolerance profile from `docs/05_Eval_Metrics_Spec.md` unless a stricter named profile was used,
4. declared benchmark-family posture, including which families were Tier 1-visible and which surfaces were `observe_only`,
5. verifier-maturity evidence sufficient for the claimed profile boundary, including the frontier hard-gate surfaces from `docs/16_Verifier_and_Formalization_Stack.md` when relevant,
6. explicit residual blockers or open uncertainties, if any remain.

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
