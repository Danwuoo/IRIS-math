# Benchmark Registry and Tiering Playbook

**Document Type:** Active Companion Authority  
**Scope:** Benchmark-family registry, allowed tier placements, train-visible granularity, held-out governance, and forbidden uses for IRIS-math v2  
**Boundary:** This document governs benchmark families. It does not replace the global tier definitions in `docs/07_Data_Constitution.md`, phase/suite semantics in `docs/06_Regression_and_Phase_Gates.md`, or task-family adjudication semantics in `docs/17_Runtime_and_Task_Adjudication_Semantics.md`.

---

## 0. Purpose and Authority

Tiering rules are not complete until benchmark families are explicitly registered.

This document is the single authoritative registry for:

- which benchmark families exist as named governance objects,
- which tiers each family may occupy,
- what train-visible exposure is allowed,
- what default task-family posture and adjudication overlays apply,
- how homologous and strict held-out splits are defined,
- what uses remain forbidden.

Any benchmark family not registered here is not governance-complete for active v2 work.

---

## 1. Registry Extension Rule

Adding a benchmark family requires:

1. an entry in this registry,
2. declared train-visible granularity if Tier 1 is allowed,
3. a homologous held-out source if Tier 2 is allowed,
4. a strict held-out source if Tier 3 is allowed,
5. a declared decontamination method,
6. declared tuning-visible, observe-only, and tuning-blocked surfaces,
7. explicit forbidden uses.

Unregistered benchmark usage may not silently enter training or headline evaluation claims.

### 1.1 Executable Registry Object

Each registered family must resolve to a `benchmark_family_policy/v1` object.
The registry table below is a human-readable summary and does not replace object fields that are required for executable governance.

Minimum fields:

| Field | Requirement |
| --- | --- |
| `benchmark_family_id` | immutable family identity |
| `allowed_tiers` | allowed subset of `Tier 1/2/3` |
| `tier1_train_visible_units` | declared allowlist over `structural_label`, `process_fragment`, `theorem_or_problem_statement`, `proof_shape_fragment`, or `none` |
| `tier1_label_allowlist` | concrete structural labels such as problem family, theorem family, proof-pattern id, or difficulty band |
| `tier1_fragment_allowlist` | concrete process or proof-shape fragment types that may be train-visible |
| `tier2_homologous_source_id` | train-hidden homologous source identity when Tier 2 is allowed |
| `homology_axes` | ordered axes such as `problem_type`, `theorem_family`, `proof_pattern`, or `difficulty_band` used to build Tier 2 |
| `source_lineage_firewall` | source grouping that may not cross Tier 1 / Tier 2 / Tier 3 |
| `cluster_exclusion_key` | fingerprint or family key used to keep homologous clusters disjoint |
| `tier3_strict_holdout_source_id` | strict held-out source identity when Tier 3 is allowed |
| `decontam_policy_id` | referenced `decontam_policy/v1` |
| `default_task_family_map` | family-level default mapping from item subtype or eval surface to canonical task family |
| `task_adjudication_policy_refs` | declared default `task_adjudication_policy/v1` refs used by the family |
| `benchmark_family_adjudication_overrides` | family-specific tightening rules that specialize, but do not replace, task-family adjudication |
| `tuning_visible_surfaces` | surfaces that may be inspected during curriculum shaping or early diagnostics |
| `tuning_observe_only_surfaces` | aggregate surfaces allowed only as bounded regression checks |
| `tuning_blocked_surfaces` | surfaces forbidden from routine tuning or leaderboard chasing |
| `derivative_family_refs` | separately declared derivative families when they exist |
| `forbidden_uses` | explicit forbidden uses for the family |

Rules:

1. Tier 3 surfaces must always appear in `tuning_blocked_surfaces`.
2. If Tier 1 is allowed, Tier 2 must exist and remain train-hidden.
3. Derivative families must carry distinct identities and may not be described as silent exposure to the original family.
4. Outcome-facing family usage is non-compliant until default task-family posture and adjudication-policy attachment are declared.

---

## 2. Benchmark Family Registry

### 2.1 Registry Summary

| Benchmark Family | Primary Role | Allowed Tier(s) | Tier 1 Mode | Default Task-Family Posture | Tier 2 Homology Signature | Strict Held-Out Source | Allowed Phases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `AIMO` | Outcome-facing contest family | `Tier 1`, `Tier 2`, `Tier 3` | structural labels + short process fragments only | `answer_only` by default; `answer_with_rationale` where rationale is explicitly scored or emitted | `problem_type` + `proof_pattern` + `difficulty_band`; theorem-family key only when recoverable | private-like or post-cutoff unseen contest pool | `B-E` |
| `Omni-MATH` | Curriculum + eval family | `Tier 1`, `Tier 2`, `Tier 3` | structural labels + process fragments + bounded proof-shape fragments | mixed family; item-level resolution required across answer-bearing and proof-bearing tasks | `problem_type` + `proof_pattern` + `difficulty_band`; theorem-family key when annotated | strict held-out plus reformulated held-out Omni-MATH-style split | `B-E` |
| `miniF2F` | Formal-bridge family | `Tier 1`, `Tier 2`, `Tier 3` | structural labels + theorem statements + proof-shape fragments + aligned formal snippets | `proof_semi_formal` or `formalization` depending bridge/checker surface | `theorem_family` + `proof_pattern` + `difficulty_band`; library lineage stays in the split firewall | strict held-out formal benchmark split | `B-E` |
| `FrontierMath` | Frontier outcome family | original family is `Tier 3` only; Tier 1 allowed only on separately declared derivative families | none on the original family | explicit item declaration required on the original family; derivative families must declare their own defaults | n/a on the original family; derivative family required if Tier 1 exists | original strict held-out FrontierMath set or equivalent unseen frontier pool | `D-E` |
| `ARC compatibility probes` | Compatibility-only transition family | `Tier 2`, `Tier 3` compatibility probes only | none | compatibility-only; not authoritative for active task-family adjudication | legacy compatibility split | strict compatibility probe split | `C-E` |

### 2.1.1 Benchmark-Family Adjudication Attachment

Rules:

1. A registered family must declare either:
   - default task-family mappings for its item subtypes, or
   - that item-level task-family resolution is mandatory.
2. A family may attach default `task_adjudication_policy/v1` refs or benchmark-family tightening overlays, but those overlays may only tighten the resolved task-family policy.
3. Mixed families such as `Omni-MATH` may not rely on one family-wide adjudication rule across heterogeneous item types.
4. A family policy may require stronger abstention or stronger verifier modes than the task-family default, but it may not relax required evidence bundles defined by `docs/16` and `docs/19`.

### 2.2 Tier 1 Admission Matrix

This matrix answers what may become train-visible for the currently registered families.

| Family | Structural Labels | Difficulty Bands | Theorem / Problem Statements | Proof-Shape Fragments | Process Fragments | Full Problem + Full Official Solution |
| --- | --- | --- | --- | --- | --- | --- |
| `AIMO` | yes | yes | no | no | yes | no |
| `Omni-MATH` | yes | yes | no | yes | yes | no |
| `miniF2F` | yes | yes | yes | yes | yes | no |
| `FrontierMath` (original family) | no | no | no | no | no | no |
| `FrontierMath` derivative family | yes | yes | no from original family; derivative statements only if independently authored and separately registered | conditional | conditional | no |

### 2.3 AIMO

**Tier 1 allowlist**

- `tier1_label_allowlist`: domain, subdomain, problem archetype, output form, difficulty band
- `tier1_fragment_allowlist`: first representation step, branch-choice snippet, invariant proposal, verifier contrast pair
- `tier1_train_visible_units`: `structural_label`, `process_fragment`

**Tier 1 blocked**

- raw problem statements
- theorem statements reconstructed from official items
- proof-shape fragments that expose decisive terminal derivations
- full official solutions or answer-only labels detached from reasoning cleaning

**Tier 2 homologous policy**

- `homology_axes`: `problem_type`, `proof_pattern`, `difficulty_band`
- theorem-family key is optional and may be used only when it is recoverable from the annotation layer
- `source_lineage_firewall`: contest series, contest year, source snapshot, and derivative-family id must remain disjoint across tiers

**Tier 3 strict held-out**

- private-like or post-cutoff contest pool with no train-visible derivative lineage

**Decontam emphasis**

- theorem/problem fingerprinting
- proof-fragment fingerprinting
- contest/year/source-lineage quarantine
- reformulation and translation leakage audit

**Tuning visibility**

- `tuning_visible_surfaces`: Tier 1 usage ratios, contamination summaries, provenance coverage
- `tuning_observe_only_surfaces`: fixed-cadence family-level Tier 2 aggregate metrics and generalization gap
- `tuning_blocked_surfaces`: all Tier 3 surfaces, all item-level Tier 2 outputs, per-contest breakdowns, and any hidden solution text or proof traces from Tier 2 / Tier 3

**Adjudication posture**

- default task family is `answer_only`
- use `answer_with_rationale` only when the eval surface explicitly scores or requires rationale-bearing output
- family overlays may tighten abstention on unresolved answer validity but may not weaken answer-check evidence requirements

**Forbidden uses**

- architecture definition
- undisclosed full benchmark mixing
- sole promotion evidence
- repeated hyperparameter sweeps against the same AIMO Tier 2 slice

### 2.4 Omni-MATH

**Tier 1 allowlist**

- `tier1_label_allowlist`: domain, subdomain, problem type, difficulty band, answer form, proof-style tag
- `tier1_fragment_allowlist`: first-step structure, branch-choice snippet, invariant proposal, verifier contrast pair, bounded non-terminal proof-shape fragment
- `tier1_train_visible_units`: `structural_label`, `process_fragment`, `proof_shape_fragment`

**Tier 1 blocked**

- raw benchmark problem statements
- full end-to-end official solutions
- proof-shape fragments that preserve the decisive final construction or contradiction step

**Tier 2 homologous policy**

- `homology_axes`: `problem_type`, `proof_pattern`, `difficulty_band`
- theorem-family key must be added when Omni-MATH annotations or derived labels expose theorem-family information
- `source_lineage_firewall`: source corpus, restatement cluster, and derivative-family id must remain disjoint across tiers

**Tier 3 strict held-out**

- strict held-out plus adversarially reformulated held-out split

**Decontam emphasis**

- normalized problem fingerprinting
- solution-fragment fingerprinting
- style-family separation
- homologous restatement and translation split audit

**Tuning visibility**

- `tuning_visible_surfaces`: Tier 1 usage ratios, contamination summaries, provenance coverage
- `tuning_observe_only_surfaces`: fixed-cadence family-level Tier 2 aggregate metrics and generalization gap
- `tuning_blocked_surfaces`: all Tier 3 surfaces, all item-level Tier 2 outputs, homology-axis-aligned held-out slices, and full held-out solution traces

**Adjudication posture**

- `Omni-MATH` is mixed-family by default
- item-level task-family resolution is required across answer-bearing and proof-bearing items
- family overlays may tighten verifier mode or abstention on proof-bearing items but may not collapse all items into one answer-only policy

**Forbidden uses**

- using Tier 1 exposure as final evidence
- hidden curriculum leakage
- family-specific leaderboard chasing on Tier 2 or Tier 3

### 2.5 miniF2F

**Tier 1 allowlist**

- `tier1_label_allowlist`: domain, library lineage, theorem family id, proof-pattern id, difficulty band, tactic-depth band
- `tier1_fragment_allowlist`: theorem statement, aligned natural-to-formal snippet, semi-formal proof skeleton, tactic-family outline with terminal chain removed, verifier contrast pair
- `tier1_train_visible_units`: `structural_label`, `process_fragment`, `theorem_or_problem_statement`, `proof_shape_fragment`

**Tier 1 blocked**

- full proof scripts as a dominant supervised corpus
- theorem statement plus official proof pair as an undisclosed end-to-end benchmark channel
- held-out theorem statements or checker traces crossing the source-lineage firewall

**Tier 2 homologous policy**

- `homology_axes`: `theorem_family`, `proof_pattern`, `difficulty_band`
- library lineage is mandatory as a split constraint and should also be reported because theorem families can collapse within one library
- `source_lineage_firewall`: formal library lineage, theorem identity cluster, proof lineage, and derivative-family id must remain disjoint across tiers

**Tier 3 strict held-out**

- strict held-out formal benchmark split with theorem-identity and proof-lineage exclusion

**Decontam emphasis**

- formal statement hashing
- alpha-renamed theorem fingerprinting
- proof-skeleton fingerprinting
- natural-language paraphrase audit when semi-formal or natural statements are exposed

**Tuning visibility**

- `tuning_visible_surfaces`: Tier 1 usage ratios, contamination summaries, provenance coverage
- `tuning_observe_only_surfaces`: fixed-cadence family-level Tier 2 aggregate metrics, proof-validity calibration, and theorem-family generalization gap
- `tuning_blocked_surfaces`: all Tier 3 surfaces, all item-level Tier 2 theorem texts, proof scripts, checker traces, and library-specific hidden leaderboards

**Adjudication posture**

- default task families are `proof_semi_formal` and `formalization`
- checker-backed or bridge-backed evidence is mandatory on the surfaces that claim formalization success
- family overlays may tighten bridge coverage and abstention rules but may not accept theorem plausibility without the declared checker surface

**Forbidden uses**

- provenance-free formal data mixing
- collapsing formal and held-out sources
- theorem-family-specific tuning loops on hidden theorem families

### 2.6 FrontierMath

**Original family policy**

- `allowed_tiers`: `Tier 3`
- `tier1_train_visible_units`: `none`
- the original FrontierMath family is never train-visible and never provides tuning-visible surfaces

**Derivative-family exception**

- Tier 1 is allowed only on a separately declared derivative family with its own `benchmark_family_id`
- the derivative family must be independently authored or independently sourced rather than lightly rewritten from official FrontierMath items
- the derivative family must keep original FrontierMath statements, solutions, and proof traces completely out of train-visible data
- derivative-family Tier 1 exposure is limited to structural labels, difficulty bands, and separately registered bounded proof-shape or process fragments
- if a derivative family allows proof-shape or process fragments, it must also declare Tier 2 with `homology_axes` including `proof_pattern`

**Derivative-family registration checklist**

- attach explicit independence evidence such as source snapshot lineage, authorship basis, or independent curation note
- attach a reformulation-leakage audit artifact against the original FrontierMath family before Tier 1 admission
- declare author/source/project lineage in the firewall, not only topic or method cluster
- keep original official statements, explanations, and checker traces outside train-visible curation even when derivative families exist
- do not expose derivative-family Tier 2 observe-only metrics until registration artifacts are complete

**Tier 3 strict held-out**

- original FrontierMath remains the strict held-out surface even when a derivative family exists

**Decontam emphasis**

- official statement fingerprinting
- frontier-topic and method-cluster fingerprinting
- author/source-lineage firewall
- reformulation leakage audit between original and derivative families

**Tuning visibility**

- `tuning_visible_surfaces`: none on the original family
- `tuning_observe_only_surfaces`: none on the original family; derivative families may expose fixed-cadence aggregate Tier 2 metrics only after separate registration
- `tuning_blocked_surfaces`: all original FrontierMath metrics, items, explanations, and slice breakdowns during tuning, checkpoint selection, or curriculum shaping

**Adjudication posture**

- original FrontierMath requires explicit item-level task-family declaration on outcome-facing eval
- derivative families must publish their own task-family defaults and adjudication overlays
- frontier-family overlays may tighten abstention and verifier requirements, but they may not relax task-family evidence bundles

**Forbidden uses**

- direct benchmark training on the original family
- claiming original FrontierMath remains strict held-out after hidden derivative use without separate family registration
- using original FrontierMath score curves for routine hyperparameter or prompt tuning

### 2.7 ARC Compatibility Probes

Compatibility probes remain unchanged:

- `allowed_tiers`: `Tier 2`, `Tier 3`
- `tier1_train_visible_units`: `none`
- no train-return path
- not valid as primary math promotion evidence

**Transition boundary**

- ARC compatibility probes default to `observe_only` once active math-native concept-isolation and reformulation packets exist for the same regression axis
- ARC probes may block promotion only when they expose drift that is also corroborated by active math-native surfaces
- ARC probe improvement may not be cited as a sufficient positive signal for `P3` or `P4` promotion

**Retirement trigger**

- once two consecutive promotion packets show stable active math-native `S4/S5` coverage for the corresponding axis, ARC compatibility probes move to appendix-only reporting
- by `P4`, ARC compatibility probes must not appear as a primary promotion surface

---

## 3. Playbook Rules

### 3.1 Tier 1 Rule

If a family is allowed in Tier 1:

- the train-visible granularity must be declared up front,
- exposure must be disclosed in run manifests,
- the family must still retain train-hidden homologous evaluation,
- Tier 1 material must not dominate the full supervised mixture.

### 3.2 Tier 2 Rule

Tier 2 is mandatory whenever Tier 1 is allowed.

Tier 2 must answer:

- did the model generalize beyond train-visible benchmark derivatives,
- did benchmark exposure distort nearby generalization,
- did curriculum gains survive homologous hold-out pressure.

### 3.3 Tier 3 Rule

Tier 3 remains the only strict benchmark surface for frontier claims.

Tier 3 material must:

- remain train-hidden,
- remain isolated from curriculum tuning loops,
- be protected by the family-specific decontamination method.

### 3.4 Tuning Visibility Rule

Benchmark-aware tuning must obey the executable family policy.

Minimum rules:

- Tier 1 surfaces may be used only within the declared train-visible granularity.
- Tier 2 surfaces may support periodic regression and generalization checks only through declared `tuning_observe_only_surfaces`; item-level or homology-axis-aligned drill-down is blocked.
- Tier 3 surfaces are blocked from curriculum tuning and routine leaderboard optimization.

### 3.4.1 Benchmark-Adjudication Overlay Rule

1. Benchmark-family overlays may tighten abstention posture, verifier mode, or evidence requirements, but they may not weaken a resolved task-family policy.
2. Benchmark-family overlays may not use hidden item-level ground truth as a silent acceptance teacher.
3. If a family changes its adjudication overlay in a behavior-affecting way, the `benchmark_family_policy/v1` object must receive a new immutable identity.

### 3.4.2 Tuning-Blocked Surfaces Once Benchmark Exposure Exists

Once a family has Tier 1 exposure, the following remain tuning-blocked unless a stricter family rule already blocks more:

- all Tier 3 metrics before post-lock evaluation
- all Tier 3 item texts, solutions, checker traces, and error analyses
- all item-level Tier 2 outputs
- Tier 2 slice breakdowns that directly mirror `problem_type`, `theorem_family`, `proof_pattern`, or other declared homology axes
- repeated checkpoint or hyperparameter selection against a fixed Tier 2 split
- train-data curation triggered directly by Tier 2 / Tier 3 misses

---

## 4. Allowed Derivative Families

Derivative families are allowed only when declared as distinct data objects.

Examples:

- structural labels,
- process fragments,
- theorem-style tags,
- formalization snippets,
- homologous synthetic restatements.

Derivative families must not be silently described as if the original benchmark remained untouched.
They must also retain explicit independence and leakage-audit artifacts in the surrounding governance packet, even when those artifacts live outside the executable family object itself.

---

## 5. Phase and Promotion Boundary

This document governs benchmark-family use.

It does not by itself:

- decide `A-E` phase promotion,
- decide `P1-P4` profile promotion,
- waive verifier evidence requirements,
- waive contamination audit requirements.

Those decisions remain in:

- `docs/06_Regression_and_Phase_Gates.md`
- `docs/14_Verifier_and_Formalization_Stack.md`
- `docs/15_Scaling_Promotion_and_Readiness.md`

---

## 6. Final Rule

If benchmark usage cannot be explained in terms of:

- declared family registry entry,
- declared tier placement,
- declared train-visible granularity,
- declared task-family posture,
- declared adjudication overlays,
- declared decontamination method,
- declared held-out sources,
- declared tuning-visible, observe-only, and tuning-blocked surfaces,

then the benchmark usage is non-compliant for active v2 work.
