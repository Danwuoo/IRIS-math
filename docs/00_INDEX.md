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

### 2.2 Active v2 Contracts and Policy

4. `docs/07_Data_Constitution.md`
5. `docs/01_Architecture_Constitution.md`
6. `docs/02_State_IR_Spec.md`
7. `docs/03_Level_Contracts_L0-L6.md`
8. `docs/04_Credit_Assignment_and_Recovery.md`
9. `docs/05_Eval_Metrics_Spec.md`
10. `docs/06_Regression_and_Phase_Gates.md`
11. `docs/08_Training_Run_Governance.md`
12. `docs/09_Training_Profiles_and_Scaling.md`

---

## 3. Mandatory Reading Paths

### 3.1 Default Implementation Path

1. `docs/數學模型建議.md`
2. `docs/00_INDEX.md`
3. `docs/10_Glossary_and_Normative_Status.md`
4. `docs/07_Data_Constitution.md`
5. `docs/01_Architecture_Constitution.md`
6. `docs/02_State_IR_Spec.md`
7. `docs/03_Level_Contracts_L0-L6.md`
8. `docs/04_Credit_Assignment_and_Recovery.md`

Add `docs/05`, `docs/06`, `docs/08`, and `docs/09` whenever the task touches metrics, regression, training governance, scaling, or eval policy.

### 3.2 Research / Architecture Path

1. `docs/數學模型建議.md`
2. `docs/10_Glossary_and_Normative_Status.md`
3. `docs/07_Data_Constitution.md`
4. `docs/01_Architecture_Constitution.md`
5. `docs/02_State_IR_Spec.md`
6. `docs/03_Level_Contracts_L0-L6.md`
7. `docs/04_Credit_Assignment_and_Recovery.md`

### 3.3 Training / Evaluation Path

1. `docs/數學模型建議.md`
2. `docs/10_Glossary_and_Normative_Status.md`
3. `docs/07_Data_Constitution.md`
4. `docs/05_Eval_Metrics_Spec.md`
5. `docs/06_Regression_and_Phase_Gates.md`
6. `docs/08_Training_Run_Governance.md`
7. `docs/09_Training_Profiles_and_Scaling.md`

---

## 4. Authority and Transition Reminder

- `docs/01` through `docs/08` define the active v2 target.
- `docs/09` is the active scaling/profile note for staged execution.
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
