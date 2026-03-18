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
- **Active Profile Note:** active profile-family guidance that does not itself define promotion readiness.
- **Active Companion Authority:** active scoped authority that elaborates the core v2 spine without replacing stable external interfaces.
- **Documentation-First Transition:** active docs lead while implementation may temporarily lag.
- **Archive:** preserved for baseline history only; not active guidance.

---

## 2. Active Authority Map

### 2.1 Active Navigation

| Document | Status | Role |
| --- | --- | --- |
| `docs/00_INDEX.md` | Design Note | Navigation and reading paths |
| `docs/10_Glossary_and_Normative_Status.md` | Design Note | Authority map and shared vocabulary |

### 2.2 Active v2 Core Spine and Primary Policy

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
| `docs/09_Training_Profiles_and_Scaling.md` | Active Profile Note | Hardware/purpose profile families and profile intent |

### 2.3 Active v2 Companion Docs

| Document | Status | Role |
| --- | --- | --- |
| `docs/11_Goals_and_Success_Criteria.md` | Active Companion Authority | Program goals, success surfaces, red-line failure criteria |
| `docs/12_Multimodal_Document_Pipeline.md` | Active Companion Authority | Document-native pipeline, modality normalization, canonical anchors |
| `docs/13_Benchmark_Registry_and_Tiering_Playbook.md` | Active Companion Authority | Benchmark family registry, tiering playbook, allowed and forbidden family uses |
| `docs/14_Verifier_and_Formalization_Stack.md` | Active Companion Authority | Verifier stack roles, evidence classes, false accept / false reject policy |
| `docs/15_Scaling_Promotion_and_Readiness.md` | Active Companion Authority | Capability-readiness promotion rules across profile families |
| `docs/16_Optimization_and_Learning_Contract.md` | Active Companion Authority | Learning-objective families, curriculum activation, and level-addressable optimization contract |
| `docs/17_Runtime_and_Task_Adjudication_Semantics.md` | Active Companion Authority | Runtime-cycle semantics, task-family adjudication, and terminal-status policy |

## 3. Transition Terms

- **Documentation-First Transition:** the repository state where active v2 docs are authoritative even if `src/` still contains baseline-aligned code.
- **Baseline Implementation:** existing code paths or tests that predate the active v2 contracts.
- **Archive Compatibility Signal:** an old metric or harness kept temporarily to monitor drift during migration.
- **Core Spine:** the active architectural / representational / policy backbone in `docs/01` through `docs/09`.
- **Companion Authority:** an active cross-cutting document that closes operational gaps without expanding the external level ids, failure taxonomy, suite ids, or phase ids.

---

## 4. Core Terms

- **IRIS-math v2:** the active target, a document-native, multimodal, verifier-centered mathematical foundation model.
- **Goals and Success Criteria:** the binding success model for core capabilities, system behavior, external outcomes, and red-line failure conditions.
- **Data Constitution:** the binding policy that defines allowed training pools, benchmark tiers, contamination controls, and provenance rules.
- **Data Realization Policy (`data_realization_policy/v1`):** executable policy object that binds a profile/phase to pool allocations, role usage, quality gates, Tier 1 caps, and referenced benchmark/decontamination policies.
- **Benchmark Family Registry:** the authoritative table defining each benchmark family's allowed tiers, train-visible granularity, held-out sources, and forbidden uses.
- **Benchmark Family Policy (`benchmark_family_policy/v1`):** executable family object that declares allowed tiers, train-visible units, homology axes, source-lineage firewalls, held-out sources, default task-family posture, adjudication-policy attachments, tuning visibility, derivative-family refs, and forbidden uses.
- **Decontam Policy (`decontam_policy/v1`):** executable policy object that declares parallel fingerprint layers, normalization rules, duplicate thresholds, homologous split rules, and leakage-audit method.
- **Learning Objective Bundle (`learning_objective_bundle/v1`):** executable optimization contract object that binds a profile/phase to active objective families, level surface maps, control-learning mode, verifier conditioning, and failure-replay policy; runtime resolves exactly one active bundle and may not merge multiple bundles on the fly.
- **Default Learning Objective Bundle Map:** the profile-family mapping from `phase` to default `learning_objective_bundle_id` used when a run manifest does not name an explicit bundle.
- **Provenance Manifest (`provenance_manifest/v1`):** immutable backend identity object for parser, extractor, formalizer, or verifier surfaces referenced by training or evaluation artifacts.
- **Task Adjudication Policy (`task_adjudication_policy/v1`):** executable runtime contract object that binds a task family to verifier mode, evidence requirements, acceptance/rejection rules, abstention rules, and benchmark-family overlays; terminal adjudication resolves exactly one policy per attempt.
- **Task Family Resolution Source:** the canonical source tag recording whether task-family classification came from the `PF` classifier, explicit item metadata, eval-manifest override, or benchmark-family default.
- **Task Adjudication Policy Resolution Source:** the canonical source tag recording whether the active adjudication policy came from item-level policy, suite-level override, suite default, benchmark-family default, or global task-family default.
- **Train-Visible Record Admission Contract:** the minimum required metadata carried by any train-visible record before it may enter a run manifest.
- **Dual-Weight Realization:** the requirement that a data realization declares both `token_weight` and `record_weight`, especially for long-document pools.
- **Homology Axes:** the declared structural axes, such as `problem_type`, `theorem_family`, `proof_pattern`, or `difficulty_band`, that define how Tier 2 homologous evaluation matches Tier 1 exposure.
- **Source-Lineage Firewall:** the required source-group separation that prevents the same contest series, source corpus, formal library lineage, author lineage, or derivative-family lineage from crossing Tier 1 / Tier 2 / Tier 3.
- **Cluster Exclusion Key:** the fingerprint or family key used to keep homologous Tier 1 / Tier 2 clusters disjoint even when they match on declared homology axes.
- **Proof-Pattern:** the abstract strategy family or proof-shape cluster used to define homologous evaluation without exposing exact benchmark items.
- **Tuning Observe-Only Surface:** a benchmark-family-specific aggregate surface that may be read only at fixed declared cadence for regression or readiness checks, not for item-level tuning.
- **Tuning-Blocked Surface:** any benchmark-derived metric, item, explanation, or slice that may not be used for checkpoint selection, curriculum shaping, prompt shaping, or hyperparameter selection.
- **Problem Frame:** the singleton State IR frame record describing task type, target, required output, problem-global assumptions, and source anchors.
- **Symbol Table:** the State IR slot whose minimal mutable unit is a `symbol_entry` carrying scope, binding state, type status, and unresolved-reference state in-band.
- **Constraint Graph:** the State IR slot whose canonical unit is a typed relation tuple or hyperedge rather than a binary-edge-only graph.
- **Proof / Program Frontier:** the State IR slot whose canonical entry kinds are `branch`, `subgoal`, `obligation`, `hypothesis`, and `strategy_candidate`.
- **Memory / Lemma Interface:** the State IR slot for retrieved lemmas or patterns, each with a binding map and explicit applicability audit.
- **Derived Abstraction:** a reusable invariant, macro, compression summary, or lemma-like object emitted by `L5` that must land canonically in `LM`, `FR`, or `CG` rather than in hidden side metadata.
- **Verifier State:** the State IR slot for verifier evidence objects plus consistency-summary objects.
- **Control State:** the singleton State IR control record with symbolic control actions, explicit budget, uncertainty, escalation, runtime status, and adjudication state.
- **Runtime Status:** the canonical persisted `CS` status vocabulary: `in_progress`, `candidate_ready`, `accepted`, `rejected`, `abstained`, `budget_exhausted`.
- **Adjudication State:** the `CS` sub-object that records which `task_adjudication_policy/v1` is active, whether a candidate is adjudication-ready, and which evidence or blockers are decisive.
- **Adjudication Status:** the canonical persisted adjudication vocabulary: `pending`, `ready`, `accepted`, `rejected`, `abstained`, `blocked`.
- **State IR Entry:** an identified mutable object inside a non-singleton State IR slot group, or an identified nested object inside `PF` or `CS`.
- **Deferred Field:** a field that may be absent or marked `pending` at object creation, but whose incompleteness must remain explicit until it is resolved or intentionally dropped.
- **Scope Ref:** the canonical `{ scope_kind, scope_id, parent_scope_id? }` shape used to identify problem-global, branch-local, or quote-local scope.
- **Relation Tuple:** the canonical `CG` object form carrying `relation_type`, ordered arguments, and relation-specific qualifiers.
- **Control Action:** the symbolic control object in `CS` that identifies actions such as `continue`, `backtrack`, `reparse`, `switch_strategy`, or `stop`, optionally with learned scores.
- **Applicability Audit:** the requirement that retrieved lemmas or examples must carry explicit match conditions, binding alignment, and mismatch evidence, not just similarity scores.
- **Benchmark Tier 1:** train-visible benchmark pool used only under declared curriculum or structural-signal policy.
- **Benchmark Tier 2:** homologous held-out evaluation set, distributionally related but train-hidden.
- **Benchmark Tier 3:** strict held-out frontier evaluation, never train-visible.
- **Parser Provenance:** versioned identity of OCR, layout, formula, or document parsers used to create canonical artifacts.
- **Formalizer Provenance:** versioned identity of natural-to-formal or semi-formal conversion tooling.
- **Verifier Build Provenance:** build identity and version of proof checker, counterexample engine, or formal verifier used in data or evaluation.
- **Verifier Evidence:** auditable signals in the canonical classes `local_validity`, `gap`, `counterexample`, and `formal_bridge` that can influence validity judgment, credit routing, or recovery.
- **Capability Readiness:** the cross-metric state where benchmark governance, contamination control, document robustness, verifier maturity, provenance reproducibility, and failure distribution are stable enough to justify profile promotion.
- **Institution Solved:** the point at which the `P1` institution validator can stably run the full data / provenance / verifier / contamination / regression loop.
- **Failure Taxonomy:** the stable failure codes `F_REP`, `F_PROC`, `F_SEARCH`, `F_MEM`, `F_ABS`, `F_EVAL`.
- **Level (`L0-L6`):** externally stable responsibility interfaces used during first-round v2 migration.

---

## 5. Boundary Rules

1. Active v2 docs override baseline implementation behavior.
2. There is no separate active strategy-note authority above the active contract set; directional interpretation must come from the active contracts and companion docs.
3. Active companion docs elaborate scoped authority and are binding within that scope.
4. Deleted archive docs do not define active authority; legacy `StateIR(T,G,O,R,X,M)` or ARC-specific semantics remain non-authoritative even when referenced historically.
5. Benchmark usage is governed by `docs/07_Data_Constitution.md` and `docs/13_Benchmark_Registry_and_Tiering_Playbook.md`, not by legacy blanket bans.
6. `src/` may lag the docs during transition, but the lag must be explicit and temporary.
