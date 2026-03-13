# State IR Spec

**Document Type:** Canonical Specification (Normative)<br>
**Effective date:** 2026-03-11<br>
**Version:** 2.2<br>
**Revision note:** Clarifies `L5` landing semantics, control-state runtime/adjudication fields, and binding to task adjudication without changing the seven-slot inventory.<br>
**Authority:** This document defines the active IRIS-math v2 State IR.

---

## 1. Purpose and Transition Scope

State IR is the canonical internal work-state for IRIS-math v2.

It exists to represent mathematical reasoning over:

- text,
- documents,
- formulas,
- diagrams,
- formal or semi-formal artifacts,
- verifier feedback.

The repository is in a documentation-first transition.
During this period, baseline implementation may still expose legacy scaffolding.
Such scaffolding is transitional and does not replace the v2 contract defined here.

---

## 2. Core Principles

1. **Single Canonical Work State**  
   All semantic reasoning state must map into this State IR.

2. **Math-Native Semantics**  
   State IR represents mathematical work structure, not generic conversation state.

3. **Verifier Visibility**  
   Verification and proof-validity signals are first-class parts of state.

4. **Control Visibility**  
   Strategy switching, backtracking, stopping, and budget state must be representable.

5. **Cross-Modal Grounding**  
   Text, document layout, formulas, and diagrams must normalize into one coherent state contract.

6. **Versioned Evolution**  
   Slot inventory, ordering, or mandatory semantics cannot drift silently.

---

## 3. Canonical Slot System

State IR v2 uses seven fixed-order slot groups.

| Slot | Symbol | Cardinality | Minimal mutable unit | Purpose |
| --- | --- | --- | --- | --- |
| Problem Frame | `PF` | exactly 1 | singleton frame record plus identified nested assumptions / anchor refs | Task type, target, output contract, global assumptions, source anchors |
| Symbol Table | `SY` | `0..N_sy` | one `symbol_entry` | Variables, constants, functions, sets, types, scopes, unresolved references |
| Constraint Graph | `CG` | `0..N_cg` | one `constraint_relation` | Equalities, inequalities, dependencies, incidences, recurrences, relation tuples |
| Proof / Program Frontier | `FR` | `0..N_fr` | one `frontier_entry` | Hypotheses, subgoals, candidate strategies, branches, obligations |
| Memory / Lemma Interface | `LM` | `0..N_lm` | one `lemma_binding` | Retrieved lemmas, match conditions, applicability audit, mismatch notes |
| Verifier State | `VS` | `0..N_vs` | one `verifier_entry` | Local validity, gap evidence, counterexample evidence, consistency summaries |
| Control State | `CS` | exactly 1 | singleton control record plus identified `control_action` entries | Continue/backtrack/reparse/switch/stop intent, budget, escalation, runtime status, adjudication |

No additional top-level slot groups are permitted.

---

## 4. Canonical Sequence Construction

The fixed slot order is:

```text
Z = [ PF ; SY_1...SY_n ; CG_1...CG_m ; FR_1...FR_k ; LM_1...LM_p ; VS_1...VS_q ; CS ]
```

Rules:

- `PF` and `CS` must always be present.
- `SY`, `CG`, `FR`, `LM`, and `VS` may be empty.
- Relative ordering within each slot group must remain stable within a reasoning cycle.
- Reordering inside a slot group is a state mutation and must be explicitly produced.
- Every mutable entry must carry a stable slot-local id.
- Cross-slot references must use stable ids, not list position.

---

## 5. Common Object Model

### 5.1 Common Reference Types

State IR uses three common reference shapes.

```text
state_ref = { slot, entry_id, field_path? }
scope_ref = { scope_kind, scope_id, parent_scope_id? }
anchor_ref = { anchor_id, anchor_role, support_kind, confidence? }
```

Rules:

1. `state_ref` points to another State IR object or subfield and is the only compliant semantic cross-link shape.
2. `scope_ref` is required anywhere scope matters. `scope_kind` must distinguish at least `problem_global`, `branch_local`, and `quote_local` when those cases occur.
3. `anchor_ref.anchor_id` must resolve to the canonical anchor object governed by `docs/14_Multimodal_Document_Pipeline.md`.
4. State IR stores anchor references, not duplicated full anchor payloads.

### 5.2 Field Obligation Classes

Every schema field belongs to one of three obligation classes.

| Class | Meaning |
| --- | --- |
| `required_at_creation` | Must exist when the object first appears in State IR |
| `deferred_allowed` | May be absent or marked `pending` at creation, but the object must expose incompleteness explicitly |
| `optional_diagnostic` | Useful for observability or ranking but not required for schema compliance |

Rules:

1. A `required_at_creation` field may not be silently inferred from another channel.
2. A `deferred_allowed` field is not a free omission. The owning object must remain visibly incomplete until the field is filled or intentionally dropped.
3. A consumer may not treat an object with unresolved required semantics as a settled mathematical fact.

### 5.3 Common Mutation Rules

Mutation operations are append, patch, retire, or explicit reorder of identified objects.

Rules:

1. New objects must declare their slot-local id at creation time.
2. Deferred fields may be patched later without changing the object id.
3. Retiring, superseding, or rejecting an object is a state mutation and must be explicit.
4. `candidate`, `unresolved`, `blocked`, or `rejected` objects remain in-band State IR objects; they must not be moved into hidden side channels.

---

## 6. Detailed Slot Schemas

### 6.1 Problem Frame (`PF`)

`PF` is a singleton frame record.
It carries task-global framing information, not branch-local proof state.

```text
PF = {
  task_type,
  target_spec,
  required_output,
  problem_assumptions,
  domain_tags?,
  source_anchor_refs,
  frame_status
}
```

| Field | Obligation | Notes |
| --- | --- | --- |
| `task_type` | `required_at_creation` | Controlled vocabulary such as `prove`, `compute`, `construct`, `decide`, `formalize`, `find_counterexample` |
| `target_spec` | `required_at_creation` | Target statement, object, or requested deliverable |
| `required_output.output_kind` | `required_at_creation` | Proof, scalar answer, construction, witness, formal artifact, counterexample, classification, or analogous output family |
| `required_output.answer_channel` | `required_at_creation` | Natural-language proof, formula, program, formal statement, structured object, or mixed output |
| `required_output.formality_level` | `deferred_allowed` | Informal, semi-formal, formal, or profile-specific equivalent |
| `required_output.format_constraints` | `deferred_allowed` | Output shape restrictions such as exact value, bounded explanation, theorem-proof format, or theorem-formalization target |
| `required_output.verifier_mode` | `deferred_allowed` | Verifier surface expected to judge success; it must resolve before terminal adjudication under `docs/19_Runtime_and_Task_Adjudication_Semantics.md` |
| `problem_assumptions[]` | `deferred_allowed` | May start empty, but the field itself must exist |
| `domain_tags[]` | `deferred_allowed` | Algebra, geometry, number theory, analysis, combinatorics, or finer domain tags |
| `source_anchor_refs[]` | `required_at_creation` | Must exist as a field; for document-grounded tasks the list may not be empty |
| `frame_status` | `required_at_creation` | At minimum distinguishes `draft`, `active`, `reparsed`, or `resolved` style frame states |

Each `problem_assumption` object must carry:

```text
problem_assumption = {
  assumption_id,
  normalized_claim,
  origin_kind,
  status,
  source_anchor_refs?
}
```

Rules:

1. `PF` holds problem-global assumptions only. Branch-local assumptions belong in `FR`.
2. `task_type` and `required_output` may not remain implicit even when the source text is ambiguous.
3. `source_anchor_refs` are references to canonical anchors, not embedded parser traces.

### 6.2 Symbol Table (`SY`)

`SY` is a collection of `symbol_entry` objects.
An unresolved reference remains an in-band `symbol_entry`; it is not a separate side channel.

```text
symbol_entry = {
  sy_id,
  surface_form,
  entity_kind,
  scope_ref,
  binding_state,
  type_status,
  canonical_name?,
  type_expr?,
  bound_to_sy_id?,
  candidate_bindings?,
  source_anchor_refs?
}
```

| Field | Obligation | Notes |
| --- | --- | --- |
| `sy_id` | `required_at_creation` | Stable symbol-entry id |
| `surface_form` | `required_at_creation` | Observed symbol string or normalized label |
| `entity_kind` | `required_at_creation` | `variable`, `constant`, `function`, `predicate`, `set`, `index`, `operator`, `parameter`, or equivalent |
| `scope_ref` | `required_at_creation` | Must identify where the symbol is valid |
| `binding_state` | `required_at_creation` | At minimum supports `bound`, `free`, `placeholder`, `unresolved`, `shadowed` |
| `type_status` | `required_at_creation` | At minimum supports `typed`, `partially_typed`, `unknown`, `conflicted` |
| `canonical_name` | `deferred_allowed` | Canonicalized identity when different from surface form |
| `type_expr` | `deferred_allowed` | Domain, codomain, sort, or richer type expression |
| `bound_to_sy_id` | `deferred_allowed` | Resolved binding target when this entry aliases or resolves to another symbol |
| `candidate_bindings[]` | `optional_diagnostic` | Candidate resolutions ranked or filtered by structuring |
| `source_anchor_refs[]` | `optional_diagnostic` | Anchor refs back to definitions or mentions |

Rules:

1. Unresolved references are represented by `binding_state = unresolved`, not by omitting the symbol from `SY`.
2. `scope_ref` is mandatory because scope errors are first-class `F_REP` signals.
3. `type_expr` may be deferred, but `type_status` may not.

### 6.3 Constraint Graph (`CG`)

`CG` is a collection of `constraint_relation` objects.
The canonical unit is a typed relation tuple or hyperedge, not a binary-edge-only graph.

```text
constraint_relation = {
  cg_id,
  relation_type,
  arguments,
  relation_status,
  qualifiers?,
  source_anchor_refs?,
  supporting_vs_refs?
}
```

| Field | Obligation | Notes |
| --- | --- | --- |
| `cg_id` | `required_at_creation` | Stable constraint id |
| `relation_type` | `required_at_creation` | Equality, inequality, divisibility, modular relation, membership, implication, incidence, recurrence, transformation, dependency, or equivalent |
| `arguments[]` | `required_at_creation` | Ordered argument list of `state_ref`s, normalized terms, or literals |
| `relation_status` | `required_at_creation` | At minimum supports `asserted`, `derived`, `candidate`, `disputed`, `retracted` |
| `qualifiers` | `deferred_allowed` | Comparator direction, modulus, quantifier guard, recurrence index, orientation, or other relation-specific qualifiers |
| `source_anchor_refs[]` | `optional_diagnostic` | Anchors back to source statements or diagram regions |
| `supporting_vs_refs[]` | `optional_diagnostic` | Verifier evidence refs that strengthen or weaken the relation |

Rules:

1. Binary typed edges are permitted only as a derived serialization of a `constraint_relation` when the relation is truly binary and the mapping is lossless.
2. `arguments[]` must preserve order because many mathematical relations are not symmetric.
3. Diagram-derived or parser-derived relations enter as `relation_status = candidate` until later structuring or verification strengthens them.

### 6.4 Proof / Program Frontier (`FR`)

`FR` is a collection of `frontier_entry` objects.
Its canonical entry kinds are `branch`, `subgoal`, `obligation`, `hypothesis`, and `strategy_candidate`.

```text
frontier_entry = branch | subgoal | obligation | hypothesis | strategy_candidate
```

Each entry kind uses the following minimum schema.

```text
branch = {
  branch_id,
  branch_status,
  local_scope_ref,
  parent_branch_id?,
  strategy_family?,
  summary?
}

subgoal = {
  subgoal_id,
  branch_id,
  goal_kind,
  target_payload,
  goal_status,
  blocking_obligation_ids?
}

obligation = {
  obligation_id,
  branch_id,
  attached_to_ref,
  obligation_kind,
  obligation_status,
  required_evidence_class?
}

hypothesis = {
  hypothesis_id,
  branch_id,
  normalized_claim,
  origin_kind,
  hypothesis_status
}

strategy_candidate = {
  strategy_id,
  branch_id,
  strategy_family,
  candidate_status,
  precondition_refs?,
  score?
}
```

Rules:

1. `branch` is the minimal control-bearing frontier context and must carry `local_scope_ref`.
2. `subgoal` is the minimal semantic target to be satisfied.
3. `obligation` is the minimal verifier-visible unmet justification unit. It may attach to a branch, subgoal, or other frontier object through `attached_to_ref`.
4. Branch-local assumptions live in `hypothesis` entries under `FR`, not in `PF`.
5. Abandoned branch summaries remain in-band `branch` objects with `branch_status = abandoned`; they are not silently deleted.

### 6.5 Memory / Lemma Interface (`LM`)

`LM` is a collection of `lemma_binding` objects.
Similarity-only retrieval is non-compliant; every usable item needs an explicit applicability audit.

```text
lemma_binding = {
  lm_id,
  memory_kind,
  source_ref,
  claim_signature,
  binding_map,
  applicability_audit,
  retrieval_signal?,
  source_anchor_refs?
}

applicability_audit = {
  audit_status,
  required_conditions,
  satisfied_condition_refs?,
  mismatch_reasons?,
  verifier_evidence_refs?
}
```

| Field | Obligation | Notes |
| --- | --- | --- |
| `lm_id` | `required_at_creation` | Stable memory-binding id |
| `memory_kind` | `required_at_creation` | `lemma`, `definition`, `example`, `pattern`, `prior_branch`, `derived_abstraction`, or equivalent |
| `source_ref` | `required_at_creation` | Memory source identity or external record reference |
| `claim_signature` | `required_at_creation` | Normalized statement, theorem key, pattern signature, or equivalent content descriptor |
| `binding_map` | `required_at_creation` | Mapping from retrieved symbols or slots into current `SY` / `FR` context; may be empty but must exist |
| `applicability_audit.audit_status` | `required_at_creation` | At minimum supports `unchecked`, `provisional`, `applicable`, `blocked`, `rejected` |
| `applicability_audit.required_conditions[]` | `required_at_creation` | Preconditions or side conditions needed for safe use |
| `applicability_audit.satisfied_condition_refs[]` | `deferred_allowed` | Refs showing which conditions are already met |
| `applicability_audit.mismatch_reasons[]` | `deferred_allowed` | Required once status becomes `blocked` or `rejected` |
| `applicability_audit.verifier_evidence_refs[]` | `optional_diagnostic` | Verifier evidence that supports or blocks use |
| `retrieval_signal` | `optional_diagnostic` | Similarity score, rank, or other retrieval-side diagnostics |
| `source_anchor_refs[]` | `optional_diagnostic` | Anchors into local documents or cited sources |

Rules:

1. A retrieved item may not be treated as safely usable unless `applicability_audit.audit_status` is at least `provisional`.
2. `binding_map` is mandatory because applicability without variable alignment is underspecified.
3. When a retrieval is blocked or rejected, `mismatch_reasons[]` must become explicit.

### 6.5A Canonical Landing Rules for `L5` Abstraction Outputs

`L5` may not emit behavior-affecting abstractions into hidden side channels.
Its outputs must land canonically in existing State IR slots according to semantic role:

1. **Reusable abstraction, macro, or lemma-like summary**  
   Lands in `LM` as a `lemma_binding` with `memory_kind = derived_abstraction`, `pattern`, `lemma`, or equivalent, plus `source_ref` back to the originating branch, proof fragment, or prior state.

2. **Branch-scoped active invariant or compressed local assumption**  
   Lands in `FR` as a `hypothesis` or `strategy_candidate` when the abstraction is only valid within the current proof/program branch.

3. **Claim-bearing invariant relation**  
   Lands in `CG` as one or more `constraint_relation` objects when the abstraction materially changes the accepted mathematical relation set.

Rules:

1. If an abstraction is simultaneously reusable and currently active in a branch, the reusable and branch-scoped effects must be represented separately rather than overloaded into one hidden object.
2. L5-originated abstractions that constrain control, retrieval, verification, or final output must be visible in `LM`, `FR`, or `CG`; diagnostic-only compression metadata is insufficient.
3. Applicability and scope remain explicit even for internally generated abstractions; local convenience summaries may not masquerade as globally valid lemmas.

### 6.6 Verifier State (`VS`)

`VS` is a collection of `verifier_entry` objects.
Local validity, gap, counterexample, and formal-bridge evidence are separate evidence objects.
Consistency is represented as a summary object that references evidence, not as a replacement for it.

```text
verifier_entry = verifier_evidence | consistency_summary

verifier_evidence = {
  vs_id,
  evidence_class,
  target_ref,
  verdict,
  polarity,
  coverage_scope,
  strength,
  provenance_ref,
  linked_obligation_refs?
}

consistency_summary = {
  vs_id,
  summary_kind,
  target_ref,
  based_on_vs_ids,
  consistency_status,
  confidence,
  provenance_ref
}
```

| Field | Obligation | Notes |
| --- | --- | --- |
| `vs_id` | `required_at_creation` | Stable verifier-entry id |
| `evidence_class` | `required_at_creation` for `verifier_evidence` | Must align with `local_validity`, `gap`, `counterexample`, or `formal_bridge` evidence families from `docs/16_Verifier_and_Formalization_Stack.md` |
| `summary_kind` | `required_at_creation` for `consistency_summary` | At minimum supports `branch_consistency` or `global_consistency` style summaries |
| `target_ref` | `required_at_creation` | What the evidence or summary is about |
| `verdict` | `required_at_creation` for `verifier_evidence` | Positive, negative, mixed, or profile-specific equivalent |
| `polarity` | `required_at_creation` for `verifier_evidence` | Whether the evidence supports, weakens, or contradicts the target |
| `coverage_scope` | `required_at_creation` | Step, claim, subgoal, branch, or global scope |
| `strength` / `confidence` | `required_at_creation` | Scalar or banded confidence representation |
| `provenance_ref` | `required_at_creation` | Verifier build, checker, or probe provenance reference |
| `linked_obligation_refs[]` | `deferred_allowed` | Obligations affected by the evidence |
| `based_on_vs_ids[]` | `required_at_creation` for `consistency_summary` | Evidence ids supporting the summary |
| `consistency_status` | `required_at_creation` for `consistency_summary` | Consistent, inconsistent, unstable, or equivalent |

Rules:

1. `VS` must keep evidence-class granularity aligned with `docs/16_Verifier_and_Formalization_Stack.md`.
2. Consistency summaries may compress evidence for control or reporting, but they do not replace the underlying evidence objects.
3. Verifier evidence without provenance is non-compliant.

### 6.7 Control State (`CS`)

`CS` is a singleton control record.
Its canonical contract is symbolic action state with optional learned scores plus explicit runtime/adjudication status.
Raw logits alone are not a compliant State IR representation.

```text
CS = {
  selected_action,
  action_candidates?,
  budget_state,
  runtime_status,
  uncertainty_state,
  escalation_state,
  adjudication_state?,
  recovery_target?
}

control_action = {
  action_id,
  action_type,
  target_ref?,
  target_level?,
  trigger_vs_refs?,
  selection_score?,
  action_status
}

budget_state = {
  global_step_budget_remaining,
  branch_expansion_budget_remaining?,
  verifier_probe_budget_remaining?,
  reparse_budget_remaining?
}

adjudication_state = {
  task_adjudication_policy_id,
  adjudication_status,
  decisive_vs_refs?,
  blocking_reason?
}
```

| Field | Obligation | Notes |
| --- | --- | --- |
| `selected_action` | `required_at_creation` | Canonical currently selected control action |
| `selected_action.action_type` | `required_at_creation` | Must support at least `continue`, `backtrack`, `reparse`, `switch_strategy`, `stop` |
| `selected_action.action_status` | `required_at_creation` | Proposed, selected, executing, completed, blocked, or equivalent |
| `selected_action.target_ref` | `deferred_allowed` | Required when the action is directed at a specific branch, subgoal, or symbol scope |
| `selected_action.target_level` | `deferred_allowed` | Useful for targeted recovery against `L0-L6` |
| `selected_action.trigger_vs_refs[]` | `optional_diagnostic` | Verifier evidence or consistency refs motivating the choice |
| `selected_action.selection_score` | `optional_diagnostic` | Learned score, calibrated score, or compact logit summary |
| `action_candidates[]` | `optional_diagnostic` | Ranked or filtered alternatives considered by the controller |
| `budget_state.global_step_budget_remaining` | `required_at_creation` | Minimum required budget field |
| `budget_state.branch_expansion_budget_remaining` | `deferred_allowed` | Optional finer-grained search budget |
| `budget_state.verifier_probe_budget_remaining` | `deferred_allowed` | Optional finer-grained verifier budget |
| `budget_state.reparse_budget_remaining` | `deferred_allowed` | Optional finer-grained reparse budget |
| `runtime_status` | `required_at_creation` | Must use one of `in_progress`, `candidate_ready`, `accepted`, `rejected`, `abstained`, `budget_exhausted` as governed by `docs/19_Runtime_and_Task_Adjudication_Semantics.md` |
| `uncertainty_state` | `required_at_creation` | Controller-visible uncertainty or indecision state |
| `escalation_state` | `required_at_creation` | Whether stronger verification, wider search, or abstention escalation is active |
| `adjudication_state.task_adjudication_policy_id` | `deferred_allowed` | Required once terminal task adjudication is attempted |
| `adjudication_state.adjudication_status` | `deferred_allowed` | Must use one of `pending`, `ready`, `accepted`, `rejected`, `abstained`, `blocked` under the active task adjudication policy |
| `adjudication_state.decisive_vs_refs[]` | `optional_diagnostic` | Verifier evidence objects that materially determined the adjudication |
| `adjudication_state.blocking_reason` | `deferred_allowed` | Required when adjudication fails or remains blocked despite a candidate output |
| `recovery_target` | `deferred_allowed` | Current intended repair locus or failure-target summary |

Rules:

1. `stop`, `switch_strategy`, and `reparse` are symbolic control actions, not out-of-band booleans.
2. Learned scores may accompany actions, but a score vector without a symbolic action identity is non-compliant.
3. Finer-grained budgets may be deferred, but `global_step_budget_remaining` may not.
4. `selected_action.action_type = stop` does not by itself imply accepted termination; terminal success or failure must be represented through `runtime_status` and, when available, `adjudication_state`.

---

## 7. Cross-Modal Anchoring and External Artifact Normalization

Document parsers, OCR systems, formalizers, verifiers, and external tools do not write raw outputs directly into the trunk.

They must first be normalized into:

- canonical parse artifacts governed by `docs/07_Data_Constitution.md`,
- canonical anchors governed by `docs/14_Multimodal_Document_Pipeline.md`,
- State IR slot content governed by this document.

State IR may retain anchor refs, provenance refs, and normalized summaries, but not raw unmanaged parser traces or tool logs.

---

## 8. Mutability Rules

1. New slot entries may be created only by learned or contract-governed modules.
2. Deletion, retirement, rejection, or consolidation must be explicit.
3. Control and verifier state are part of the same work-state contract, not out-of-band metadata.
4. Transition adapters that project baseline code into v2 slots are temporary and must be labeled as such.
5. Promoting an entry from `candidate` or `unresolved` to a stronger state is a first-class mutation and must not happen implicitly.

---

## 9. Cross-Level Contract

All levels `L0-L6`:

- must consume this State IR or a schema-faithful slice of it,
- must produce updated State IR or scalar signals derived from it,
- must not invent alternative semantic state channels,
- must not bypass the slot ordering defined above.

Typical level pressure is:

- `L0/L1`: primarily create or patch `PF`, `SY`, and `CG`,
- `L2/L3`: primarily expand `FR` and select `CS`,
- `L4`: primarily add or reject `LM` bindings,
- `L5`: primarily emit reusable abstractions into `LM`, branch-scoped invariant summaries into `FR`, and claim-bearing invariant relations into `CG`,
- `L6`: primarily emit `VS` objects and influence `CS` through verifier evidence and adjudication-ready summaries.

Program traces or symbolic artifacts may exist outside State IR as auxiliary objects, but any behavior-affecting summary must be represented back into `FR`, `LM`, `VS`, or `CS`.

---

## 10. Explicit Non-Goals

State IR is not:

- a raw OCR dump,
- a raw tool log,
- a benchmark-specific action schema,
- a human-readable proof script requirement,
- a second semantic trunk,
- a generic assistant conversation buffer.

---

## 11. Versioning and Compliance

- This specification is versioned.
- Any change to the seven slot groups, their order, or their mandatory semantics requires a major revision.
- Changes to entry kinds, required fields, or obligation classes within an existing slot require an explicit minor revision and contract diff.
- Silent fallback to baseline `T/G/O/R/X/M` semantics is not compliant with the active target, even if temporary adapters still exist in code.
