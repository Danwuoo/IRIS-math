# IRIS-math v2 Documentation Index

**Document Type:** Design Note (Non-normative)  
**Purpose:** Navigation entrypoint for the active IRIS-math v2 document set  
**Non-Override Clause:** This index does not override active contracts.

---

## 1. Current Operating State

The repository is in a **documentation-first transition**.

- **Active target:** `IRIS-math v2`
- **Authority source:** active v2 documents under `docs/`
- **Implementation reality:** parts of `src/` still reflect the baseline IRIS scaffold and must be treated as transition-state code
- **Numbering rule:** the `docs/` root reserves a single active numbering scheme; archived legacy materials live under `docs/archive/`

When active v2 docs and baseline implementation disagree, the docs win.

---

## 2. Active v2 Document Set

### 2.1 Navigation

1. `docs/00_INDEX.md`
2. `docs/10_Glossary_and_Normative_Status.md`

### 2.2 Active Core Spine and Primary Policy

3. `docs/11_Goals_and_Success_Criteria.md`
4. `docs/07_Data_Constitution.md`
5. `docs/01_Architecture_Constitution.md`
6. `docs/02_State_IR_Spec.md`
7. `docs/03_Level_Contracts_L0-L6.md`
8. `docs/04_Credit_Assignment_and_Recovery.md`
9. `docs/05_Eval_Metrics_Spec.md`
10. `docs/06_Regression_and_Phase_Gates.md`
11. `docs/08_Training_Run_Governance.md`
12. `docs/09_Training_Profiles_and_Scaling.md`

### 2.3 Active Companion Docs

13. `docs/12_Multimodal_Document_Pipeline.md`
14. `docs/13_Benchmark_Registry_and_Tiering_Playbook.md`
15. `docs/14_Verifier_and_Formalization_Stack.md`
16. `docs/15_Scaling_Promotion_and_Readiness.md`
17. `docs/16_Optimization_and_Learning_Contract.md`
18. `docs/17_Runtime_and_Task_Adjudication_Semantics.md`

## 3. Mandatory Reading Paths

### 3.1 Default Implementation Path

1. `docs/00_INDEX.md`
2. `docs/10_Glossary_and_Normative_Status.md`
3. `docs/11_Goals_and_Success_Criteria.md`
4. `docs/07_Data_Constitution.md`
5. `docs/01_Architecture_Constitution.md`
6. `docs/02_State_IR_Spec.md`
7. `docs/03_Level_Contracts_L0-L6.md`
8. `docs/04_Credit_Assignment_and_Recovery.md`
9. `docs/16_Optimization_and_Learning_Contract.md`
10. `docs/17_Runtime_and_Task_Adjudication_Semantics.md`

### 3.2 Conditional Companion Reading

Add the following when relevant:

- `docs/12_Multimodal_Document_Pipeline.md` for document parsing, OCR/layout, diagram handling, multimodal ingestion, and document-grounded reasoning work
- `docs/13_Benchmark_Registry_and_Tiering_Playbook.md` for benchmark use, contamination controls, homologous split policy, or training/eval governance involving benchmark families
- `docs/14_Verifier_and_Formalization_Stack.md` for verifier behavior, formalization, proof validity, false accept / false reject policy, or recovery behavior tied to verifier evidence
- `docs/15_Scaling_Promotion_and_Readiness.md` for scaling, profile promotion, readiness claims, or profile-to-profile advancement
- `docs/16_Optimization_and_Learning_Contract.md` for learning-objective design, level-addressable losses, curriculum activation, or control/verifier learning semantics
- `docs/17_Runtime_and_Task_Adjudication_Semantics.md` for runtime loop behavior, stopping semantics, task-family adjudication, verifier-mode binding, or accepted-output policy

Add `docs/05`, `docs/06`, `docs/08`, and `docs/09` whenever the task touches metrics, regression, training governance, scaling, or eval policy.

### 3.3 Training / Evaluation Path

1. `docs/10_Glossary_and_Normative_Status.md`
2. `docs/11_Goals_and_Success_Criteria.md`
3. `docs/07_Data_Constitution.md`
4. `docs/13_Benchmark_Registry_and_Tiering_Playbook.md`
5. `docs/14_Verifier_and_Formalization_Stack.md`
6. `docs/16_Optimization_and_Learning_Contract.md`
7. `docs/17_Runtime_and_Task_Adjudication_Semantics.md`
8. `docs/05_Eval_Metrics_Spec.md`
9. `docs/06_Regression_and_Phase_Gates.md`
10. `docs/08_Training_Run_Governance.md`
11. `docs/09_Training_Profiles_and_Scaling.md`
12. `docs/15_Scaling_Promotion_and_Readiness.md`

---

## 4. Authority and Transition Reminder

- `docs/01` through `docs/09` define the active v2 core spine and primary policy surfaces.
- `docs/11` through `docs/17` are active companion docs that elaborate goals, multimodal pipeline rules, benchmark family governance, verifier stack policy, scaling readiness, learning semantics, and runtime/adjudication semantics.
- There is no separate active strategy-note authority above the contract set.
- `docs/codex_plan/*` is workflow guidance only.

---

## 5. Legacy / Rename Mapping

Legacy materials have been moved under `docs/archive/` so the `docs/` root exposes only the active numbering scheme.

| Legacy Path | Active Destination | Status |
| --- | --- | --- |
| `docs/11_Phase_D_Diagnostics_Design_Note.md` | `docs/archive/11_Phase_D_Diagnostics_Design_Note.md` | archived legacy design note |
| `docs/12_Phase_E_Execution_Design_Note.md` | `docs/archive/12_Phase_E_Execution_Design_Note.md` | archived legacy design note |
| `docs/13_Goals_and_Success_Criteria.md` | `docs/11_Goals_and_Success_Criteria.md` | renumbered; legacy copy archived at `docs/archive/13_Goals_and_Success_Criteria.md` |
| `docs/14_Multimodal_Document_Pipeline.md` | `docs/12_Multimodal_Document_Pipeline.md` | renumbered; legacy copy archived at `docs/archive/14_Multimodal_Document_Pipeline.md` |
| `docs/15_Benchmark_Registry_and_Tiering_Playbook.md` | `docs/13_Benchmark_Registry_and_Tiering_Playbook.md` | renumbered; legacy copy archived at `docs/archive/15_Benchmark_Registry_and_Tiering_Playbook.md` |
| `docs/16_Verifier_and_Formalization_Stack.md` | `docs/14_Verifier_and_Formalization_Stack.md` | renumbered; legacy copy archived at `docs/archive/16_Verifier_and_Formalization_Stack.md` |
| `docs/17_Scaling_Promotion_and_Readiness.md` | `docs/15_Scaling_Promotion_and_Readiness.md` | renumbered; legacy copy archived at `docs/archive/17_Scaling_Promotion_and_Readiness.md` |
| `docs/18_Optimization_and_Learning_Contract.md` | `docs/16_Optimization_and_Learning_Contract.md` | renumbered; legacy copy archived at `docs/archive/18_Optimization_and_Learning_Contract.md` |
| `docs/19_Runtime_and_Task_Adjudication_Semantics.md` | `docs/17_Runtime_and_Task_Adjudication_Semantics.md` | renumbered; legacy copy archived at `docs/archive/19_Runtime_and_Task_Adjudication_Semantics.md` |
| `docs/07_Data_Mixture_and_Ingestion.md` | `docs/07_Data_Constitution.md` | renamed |
| `docs/09_Training_Profile_SingleH100_3B.md` | `docs/09_Training_Profiles_and_Scaling.md` | renamed |
| `docs/數學模型建議.md` | retired; direction is now carried by the active contract set plus `docs/16` and `docs/17` | retired |

Earlier pre-consolidation mappings remain historical context only and should not drive new implementation work.
