# Glossary and Normative Status

**Document Type:** Design Note (Non-normative)  
**Purpose:** Shared authority map and vocabulary for the IRIS-math v2 transition

---

## 1. Status Labels

Use these labels consistently.

- **Authoritative Specification (Normative):** system-level architectural contract.
- **Canonical Specification (Normative):** exact representational or responsibility contract.
- **Normative Contract:** binding interface / behavior rules.
- **Canonical Binding Policy:** binding vocabulary, workflow, or data-governance policy.
- **Engineering Governance (Policy-Binding):** run-time and reproducibility rules that do not override architecture.
- **Active Strategy Note:** prioritized directional guidance for v2 migration; informs rewrites until superseded by active contracts.
- **Documentation-First Transition:** active docs lead while implementation may temporarily lag.
- **Archive:** preserved for baseline history only; not active guidance.

---

## 2. Active Authority Map

### 2.1 Active Strategy and Navigation

| Document | Status | Role |
| --- | --- | --- |
| `docs/數學模型建議.md` | Active Strategy Note | Prioritization and design direction for IRIS-math v2 |
| `docs/00_INDEX.md` | Design Note | Navigation and reading paths |
| `docs/10_Glossary_and_Normative_Status.md` | Design Note | Authority map and shared vocabulary |

### 2.2 Active v2 Contracts and Policies

| Document | Status | Role |
| --- | --- | --- |
| `docs/01_Architecture_Constitution.md` | Authoritative Specification | Single-trunk architecture, learnable control, benchmark/tool boundaries |
| `docs/02_State_IR_Spec.md` | Canonical Specification | Math-native State IR v2 |
| `docs/03_Level_Contracts_L0-L6.md` | Normative Contract | Level responsibilities and interfaces |
| `docs/04_Credit_Assignment_and_Recovery.md` | Canonical Specification | Failure taxonomy, credit routing, targeted recovery |
| `docs/05_Eval_Metrics_Spec.md` | Canonical Binding Policy | Metrics vocabulary and field semantics |
| `docs/06_Regression_and_Phase_Gates.md` | Canonical Binding Policy | Suites, gates, tolerance and phase policy |
| `docs/07_Data_Constitution.md` | Canonical Binding Policy | Data pools, benchmark tiering, contamination and provenance rules |
| `docs/08_Training_Run_Governance.md` | Engineering Governance | Segment transactions, resume, runtime lock, provenance fields |
| `docs/09_Training_Profiles_and_Scaling.md` | Active Profile Note | Staged scaling profiles and profile intent |

### 2.3 Archive

| Document | Status | Role |
| --- | --- | --- |
| `docs/11_Phase_D_Diagnostics_Design_Note.md` | Archive | Baseline Phase D note |
| `docs/12_Phase_E_Execution_Design_Note.md` | Archive | Baseline Phase E note |

---

## 3. Transition Terms

- **Documentation-First Transition:** the repository state where active v2 docs are authoritative even if `src/` still contains baseline-aligned code.
- **Baseline Implementation:** existing code paths or tests that predate the active v2 contracts.
- **Archive Compatibility Signal:** an old metric or harness kept temporarily to monitor drift during migration.

---

## 4. Core Terms

- **IRIS-math v2:** the active target, a document-native, multimodal, verifier-centered mathematical foundation model.
- **Data Constitution:** the binding policy that defines allowed training pools, benchmark tiers, contamination controls, and provenance rules.
- **Problem Frame:** State IR slot describing task type, output form, assumptions, domain, and source anchors.
- **Symbol Table:** State IR slot describing variables, constants, functions, sets, scopes, and types.
- **Constraint Graph:** State IR slot for equalities, inequalities, dependencies, incidences, recurrences, and logical links.
- **Proof / Program Frontier:** State IR slot describing subgoals, hypotheses, candidate strategies, active branches, and unresolved obligations.
- **Memory / Lemma Interface:** State IR slot for retrieved lemmas, match conditions, applicability audit, and mismatch notes.
- **Verifier State:** State IR slot for local validity, gap risk, counterexample risk, and branch consistency.
- **Control State:** State IR slot for continue/backtrack/reparse/switch/stop behavior, budget, and escalation state.
- **Applicability Audit:** the requirement that retrieved lemmas or examples must carry explicit match conditions and mismatch evidence, not just similarity scores.
- **Benchmark Tier 1:** train-visible benchmark pool used only under declared curriculum or structural-signal policy.
- **Benchmark Tier 2:** homologous held-out evaluation set, distributionally related but train-hidden.
- **Benchmark Tier 3:** strict held-out frontier evaluation, never train-visible.
- **Parser Provenance:** versioned identity of OCR, layout, formula, or document parsers used to create canonical artifacts.
- **Formalizer Provenance:** versioned identity of natural-to-formal or semi-formal conversion tooling.
- **Verifier Build Provenance:** build identity and version of proof checker, counterexample engine, or formal verifier used in data or evaluation.
- **Failure Taxonomy:** the stable failure codes `F_REP`, `F_PROC`, `F_SEARCH`, `F_MEM`, `F_ABS`, `F_EVAL`.
- **Level (`L0-L6`):** externally stable responsibility interfaces used during first-round v2 migration.

---

## 5. Boundary Rules

1. Active v2 docs override baseline implementation behavior.
2. Active strategy notes help choose direction, but they do not override active contracts once those contracts exist.
3. Archive docs never override active v2 docs.
4. Benchmark usage is governed by `docs/07_Data_Constitution.md`, not by legacy blanket bans.
5. `src/` may lag the docs during transition, but the lag must be explicit and temporary.
