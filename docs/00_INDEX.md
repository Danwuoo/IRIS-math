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
- **Archive material:** `docs/11` and `docs/12` remain for baseline history only

When active v2 docs and baseline implementation disagree, the docs win.

---

## 2. Active v2 Document Set

### 2.1 Strategy and Navigation

1. `docs/數學模型建議.md`
2. `docs/00_INDEX.md`
3. `docs/10_Glossary_and_Normative_Status.md`

### 2.2 Active Core Spine and Primary Policy

4. `docs/13_Goals_and_Success_Criteria.md`
5. `docs/07_Data_Constitution.md`
6. `docs/01_Architecture_Constitution.md`
7. `docs/02_State_IR_Spec.md`
8. `docs/03_Level_Contracts_L0-L6.md`
9. `docs/04_Credit_Assignment_and_Recovery.md`
10. `docs/05_Eval_Metrics_Spec.md`
11. `docs/06_Regression_and_Phase_Gates.md`
12. `docs/08_Training_Run_Governance.md`
13. `docs/09_Training_Profiles_and_Scaling.md`

### 2.3 Active Companion Docs

14. `docs/14_Multimodal_Document_Pipeline.md`
15. `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`
16. `docs/16_Verifier_and_Formalization_Stack.md`
17. `docs/17_Scaling_Promotion_and_Readiness.md`

### 2.4 Archive

- `docs/11_Phase_D_Diagnostics_Design_Note.md`
- `docs/12_Phase_E_Execution_Design_Note.md`

Archive notes may mention legacy `StateIR(T,G,O,R,X,M)` or ARC-specific semantics and are not active authority.

---

## 3. Mandatory Reading Paths

### 3.1 Default Implementation Path

1. `docs/數學模型建議.md`
2. `docs/00_INDEX.md`
3. `docs/10_Glossary_and_Normative_Status.md`
4. `docs/13_Goals_and_Success_Criteria.md`
5. `docs/07_Data_Constitution.md`
6. `docs/01_Architecture_Constitution.md`
7. `docs/02_State_IR_Spec.md`
8. `docs/03_Level_Contracts_L0-L6.md`
9. `docs/04_Credit_Assignment_and_Recovery.md`

### 3.2 Conditional Companion Reading

Add the following when relevant:

- `docs/14_Multimodal_Document_Pipeline.md` for document parsing, OCR/layout, diagram handling, multimodal ingestion, and document-grounded reasoning work
- `docs/15_Benchmark_Registry_and_Tiering_Playbook.md` for benchmark use, contamination controls, homologous split policy, or training/eval governance involving benchmark families
- `docs/16_Verifier_and_Formalization_Stack.md` for verifier behavior, formalization, proof validity, false accept / false reject policy, or recovery behavior tied to verifier evidence
- `docs/17_Scaling_Promotion_and_Readiness.md` for scaling, profile promotion, readiness claims, or profile-to-profile advancement

Add `docs/05`, `docs/06`, `docs/08`, and `docs/09` whenever the task touches metrics, regression, training governance, scaling, or eval policy.

### 3.3 Training / Evaluation Path

1. `docs/數學模型建議.md`
2. `docs/10_Glossary_and_Normative_Status.md`
3. `docs/13_Goals_and_Success_Criteria.md`
4. `docs/07_Data_Constitution.md`
5. `docs/15_Benchmark_Registry_and_Tiering_Playbook.md`
6. `docs/16_Verifier_and_Formalization_Stack.md`
7. `docs/05_Eval_Metrics_Spec.md`
8. `docs/06_Regression_and_Phase_Gates.md`
9. `docs/08_Training_Run_Governance.md`
10. `docs/09_Training_Profiles_and_Scaling.md`
11. `docs/17_Scaling_Promotion_and_Readiness.md`

---

## 4. Authority and Transition Reminder

- `docs/01` through `docs/09` define the active v2 core spine and primary policy surfaces.
- `docs/13` through `docs/17` are active companion docs that elaborate goals, multimodal pipeline rules, benchmark family governance, verifier stack policy, and scaling readiness.
- `docs/數學模型建議.md` is the active strategy note used to prioritize rewrites.
- `docs/11_Phase_D_Diagnostics_Design_Note.md` and `docs/12_Phase_E_Execution_Design_Note.md` are **archive notes**, not active reading requirements.
- `docs/codex_plan/*` is workflow guidance only.

---

## 5. Legacy / Rename Mapping

These entries are kept to preserve old references only.

| Legacy Path | Active Destination | Status |
| --- | --- | --- |
| `docs/07_Data_Mixture_and_Ingestion.md` | `docs/07_Data_Constitution.md` | renamed |
| `docs/09_Training_Profile_SingleH100_3B.md` | `docs/09_Training_Profiles_and_Scaling.md` | renamed |
| `docs/11_Phase_D_Diagnostics_Design_Note.md` | archive only | archived |
| `docs/12_Phase_E_Execution_Design_Note.md` | archive only | archived |

Earlier pre-consolidation mappings remain historical context only and should not drive new implementation work.
