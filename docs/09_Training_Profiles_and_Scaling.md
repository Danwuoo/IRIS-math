# Training Profiles and Scaling

**Document Type:** Active Profile Note  
**Scope:** Staged scaling profiles for IRIS-math v2  
**Boundary:** Exact per-run hyperparameters live in run manifests and configs; this document defines profile families and intended purpose.

---

## 1. Role of This Document

This document replaces the old single-card baseline note.

Its job is to define the active profile matrix for staged scaling:

- what each profile is for,
- what evidence each profile must produce,
- how data constitution and governance expectations shift with scale,
- what not to over-claim from early profiles.

The `3B` profile is an institution validator, not the final truth of the program.

---

## 2. Active Profile Matrix

| Profile | Hardware Envelope | Model Band | Primary Purpose |
| --- | --- | --- | --- |
| `P1` | `1x H100 80GB` | `~3B` | Validate institutions: State IR learnability, parser utility, verifier signal, contamination controls |
| `P2` | `1-8x H200 NVL` | `~7B / 14B` | Expand strategy diversity, theorem reuse, partial formalization, document robustness |
| `P3` | `16x H200 SXM` | `30B+` | Long-context proof handling, multi-branch search, stronger verifier-memory interplay |
| `P4` | `1-8x B200` | `70B / 120B exploratory` | Frontier integration, strong abstraction transfer, unified multimodal math reasoning |

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

- realized pool weights,
- Tier 1 usage policy,
- decontamination policy id,
- parser provenance,
- formalizer provenance,
- verifier build provenance.

---

## 5. Guardrails

1. Do not treat benchmark-heavy training as the default recipe.
2. Do not extrapolate `120B` claims from `3B` institution checks.
3. Do not freeze one permanent mixture recipe into this document.
4. Do not let profile notes override governance or architecture contracts.

---

## 6. Required Profile Metadata

Every run that claims alignment with one of these profiles should emit at least:

- `profile_id`
- hardware description
- model-size band
- phase
- runtime lock manifest id / sha
- tokenizer fingerprint
- data-constitution policy id
- parser provenance id
- OCR/layout extractor version
- formalizer version
- verifier build id

---

## 7. Relationship to Other Documents

- `docs/07_Data_Constitution.md` defines what data is allowed and how benchmark tiers work.
- `docs/08_Training_Run_Governance.md` defines transaction, resume, and provenance rules.
- `docs/06_Regression_and_Phase_Gates.md` defines what suites and gates must pass.
