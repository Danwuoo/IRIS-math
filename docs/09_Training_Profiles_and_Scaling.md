# Training Profiles and Scaling

**Document Type:** Active Profile Note  
**Scope:** Staged scaling profiles for IRIS-math v2  
**Boundary:** Exact per-run hyperparameters live in run manifests and configs; this document defines hardware/purpose profile families and intended purpose. Promotion authority lives in `docs/15_Scaling_Promotion_and_Readiness.md`; learning-objective semantics live in `docs/16_Optimization_and_Learning_Contract.md`.

---

## 1. Role of This Document

This document replaces the old single-card baseline note.

Its job is to define the active profile matrix for staged scaling:

- what each profile is for,
- what evidence each profile must produce,
- how data constitution and governance expectations shift with scale,
- what not to over-claim from early profiles.

The `3B` profile is an institution validator, not the final truth of the program.
Profiles are hardware/purpose families, not automatic capability stages.

---

## 2. Active Profile Matrix

| Profile | Hardware Envelope | Model Band | Primary Purpose |
| --- | --- | --- | --- |
| `P1` | `1x H100 80GB` | `~3B` | Validate institutions: State IR learnability, parser utility, verifier signal, contamination controls |
| `P2` | `1-8x H200 NVL` | `~7B / 14B` | Expand strategy diversity, theorem reuse, partial formalization, document robustness |
| `P3` | `16x H200 SXM` | `30B+` | Long-context proof handling, multi-branch search, stronger verifier-memory interplay |
| `P4` | `1-8x B200` | `70B / 120B exploratory` | Frontier integration, strong abstraction transfer, unified multimodal math reasoning |

Movement from `P1` to `P4` is governed by capability readiness in `docs/15_Scaling_Promotion_and_Readiness.md`, not by parameter count alone.

---

## 3. Profile Intent and Evidence

### 3.1 `P1` — `1x H100 3B`

This profile exists to answer:

- can the v2 State IR be learned at all,
- does document parsing materially help,
- does the verifier emit stable signals,
- does benchmark tiering remain auditable,
- does contamination control actually work.

Do not use `P1` as evidence that the final scaling recipe is solved.

### 3.2 `P2` — `1-8x H200 NVL 7B / 14B`

This profile should show:

- better strategy diversity,
- better theorem / lemma reuse,
- improved OCR / document robustness,
- non-trivial gains in partial formalization.

### 3.3 `P3` — `16x H200 SXM 30B+`

This profile is where the program starts to test:

- long proof dependencies,
- cross-page theorem tracing,
- stronger verifier-memory cooperation,
- non-trivial search benefits.

### 3.4 `P4` — `1-8x B200 70B / 120B exploratory`

This profile is reserved for:

- strong abstraction transfer,
- frontier-task generalization,
- deep verifier integration,
- fully native multimodal math behavior.

Do not claim that `120B` is justified unless the verifier, data quality, and contamination discipline are already strong.

---

## 4. Data Constitution Expectations by Profile

| Profile | Data Emphasis |
| --- | --- |
| `P1` | Validate Pool C / D / E institutions and benchmark tier disclosure |
| `P2` | Expand Pool B / C / D coverage with stronger formal and document balance |
| `P3` | Increase long-form Pool D and verifier-rich Pool C coverage |
| `P4` | Preserve strict Tier 2 / Tier 3 discipline while scaling frontier difficulty |

Every profile must declare:

- `data_realization_policy_id`,
- realized pool weights,
- benchmark family policy refs,
- decontamination policy id,
- parser provenance ids / refs,
- formalizer provenance id,
- verifier provenance id.

### 4.1 Initial Realization Envelopes

These envelopes are bootstrap defaults for the first executable `data_realization_policy/v1` objects.
They are not promotion gates and do not override run-level regression review.
They are the human-readable summary of the normative bootstrap matrix in `docs/07_Data_Constitution.md` and must not diverge from that document.
Each cell is `token_weight / record_weight`.

| Profile | Pool A | Pool B | Pool C | Pool D | Pool E | Tier 1 Cap | Weak-Supervision Cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `P1` | `20 / 25` | `10 / 10` | `25 / 25` | `35 / 20` | `10 / 20` | `<= 5 / <= 5` | `<= 15 / <= 15` |
| `P2` | `15 / 20` | `15 / 15` | `25 / 25` | `30 / 20` | `15 / 20` | `<= 8 / <= 8` | `<= 20 / <= 18` |
| `P3` | `10 / 15` | `15 / 15` | `30 / 30` | `30 / 20` | `15 / 20` | `<= 8 / <= 8` | `<= 20 / <= 18` |
| `P4` | not precommitted | not precommitted | not precommitted | not precommitted | not precommitted | readiness-reviewed only | readiness-reviewed only |

Rules:

1. `P1-P3` values are starting envelopes, not frozen recipes.
2. `Pool D` record weight is intentionally lower than token weight because long documents distort token-only accounting.
3. `P4` must be justified by readiness evidence rather than inherited ratios.

---

## 5. Guardrails

1. Do not treat benchmark-heavy training as the default recipe.
2. Do not extrapolate `120B` claims from `3B` institution checks.
3. Do not freeze one permanent mixture recipe into this document.
4. Do not let profile notes override governance or architecture contracts.
5. Do not auto-promote profiles by size alone.

---

## 6. Required Profile Metadata

Every run that claims alignment with one of these profiles should emit at least:

- `profile_id`
- hardware description
- model-size band
- phase
- `default_learning_objective_bundle_map` for the active profile family
- runtime lock manifest id / sha
- tokenizer fingerprint
- `data_realization_policy_id`
- `learning_objective_bundle_id`
- `learning_objective_bundle_resolution_source`
- `decontam_policy_id`
- benchmark family policy refs
- `parser_provenance_id`
- `parser_provenance_refs`
- `parse_config_fingerprint`
- OCR/layout and formula-parser version summaries
- `formalizer_provenance_id`
- `formalizer_version`
- `verifier_provenance_id`
- `verifier_build_id`

---

## 7. Relationship to Other Documents

- `docs/07_Data_Constitution.md` defines what data is allowed and how benchmark tiers work.
- `docs/08_Training_Run_Governance.md` defines transaction, resume, and provenance rules.
- `docs/16_Optimization_and_Learning_Contract.md` defines which objective families and curriculum activations may be claimed for a profile.
- `docs/06_Regression_and_Phase_Gates.md` defines what suites and gates must pass.
- `docs/15_Scaling_Promotion_and_Readiness.md` defines whether promotion between profile families is justified.
