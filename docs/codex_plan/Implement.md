# Implementation Runbook (IRIS-math v2)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Concrete validation commands for IRIS-math v2 migration work

---

## 0. Validation Focus

For documentation-migration work, validation should first prove:

- active references use the new doc paths,
- archive docs are not treated as active guidance,
- phase-gate mandatory docs reflect the new reading order,
- old failure-code and level ids remain externally stable.

---

## 1. Useful Commands

PowerShell path scan for renamed docs:

```powershell
$paths=@('AGENTS.md','README.md')
$paths += (Get-ChildItem docs,src,tests,scripts -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object FullName)
Select-String -Path $paths -Pattern '07_Data_Mixture_and_Ingestion\.md|09_Training_Profile_SingleH100_3B\.md'
```

Targeted regression test:

```powershell
python -m pytest -q tests/test_phase_c_gate_regression.py
```

Optional broader regression slice:

```powershell
python -m pytest -q tests/test_phase_c_gate_regression.py tests/test_phase_d_gate_regression.py tests/test_phase_e_gate_regression.py
```

---

## 2. Work Rules

- Keep docs and code references aligned in the same change.
- Prefer targeted tests first.
- If a mismatch between docs and code is intentionally left for the next wave, record it in `docs/codex_plan/Documentation.md`.
