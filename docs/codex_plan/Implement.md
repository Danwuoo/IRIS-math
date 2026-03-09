# Historical Baseline Runbook (Frozen)

**Status:** Frozen historical baseline runbook  
**Scope:** Archived implementation notes for the previous baseline IRIS control
surface  
**Do Not Use As Active Source:** For active IRIS-math work, use `AGENTS.md`,
`docs/00_INDEX.md`, `docs/10_Glossary_and_Normative_Status.md`,
`docs/13..16`, and the active files under `docs/codex_plan/`.

---

# Implementation Runbook (IRIS / Codex)

**Document Type:** Design Note (Non-normative)  
**Purpose:** Concrete environment + command checklist so Codex changes are runnable and verifiable  
**Non-Override Clause:** Does not override normative contracts.

---

## 0) OS / Hardware Reality Check

- **JAX GPU is Linux-first.** For an RTX 3050 on Windows, prefer **WSL2 (Ubuntu)** for GPU runs.
- Keep a **CPU-only smoke path** working at all times (fast `S1/S2`).
- For the scale target, align with `docs/09_Training_Profile_SingleH100_3B.md` (single H100).

---

## 1) Python Environment (Linux / WSL2 recommended)

Example (Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 1.1 Offline-ish installs using `wheels/linux_cp312/`

Minimal stack for early skeleton work:

```bash
python -m pip install --no-index --find-links wheels/linux_cp312 \
  numpy \
  pytest \
  jax \
  jaxlib \
  flax \
  optax \
  orbax-checkpoint
```

Optional (CUDA 12 plugin wheels are present in this repo):

```bash
python -m pip install wheels/linux_cp312/jax_cuda12_plugin-*.whl
python -m pip install wheels/linux_cp312/jax_cuda12_pjrt-*.whl
```

Quick sanity checks:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
python -c "import flax; import optax; import orbax.checkpoint as ocp; print('ok')"
```

---

## 2) Expected Commands (Codex should keep these working)

These are the intended "fast loop" commands Codex should wire up while implementing the skeleton:

```bash
python -m pytest -q
python scripts/s1_smoke.py --device cpu
python scripts/s2_structural.py
python scripts/train_toy.py --segments 2 --device cpu --output-dir artifacts/toy_train_cpu
python scripts/eval_toy.py --output-dir artifacts/toy_train_cpu
python scripts/train_toy.py --segments 1 --device gpu --output-dir artifacts/toy_train_gpu   # optional
```

Resume exactly-once demo:

```bash
python scripts/train_toy.py --segments 1 --output-dir artifacts/toy_resume --crash-point pre_commit --crash-segment 0
python scripts/train_toy.py --segments 1 --output-dir artifacts/toy_resume
```

Notes:

- `scripts/*` should stay thin; logic belongs under `src/iris/`.
- If any command cannot be supported yet, document the gap and the next milestone in `docs/codex_plan/Plan.md`.

---

## 3) "Don’t accidentally break contracts" checklist

- State IR token categories must remain exactly `{T,G,O,R,X,M}` in fixed order.
- All Levels `L0..L6` must be importable/instantiable; stubs must emit diagnostics.
- Avoid hard-coded semantic routing/termination; if a hard cap is needed, label `TEMPORARY TECHNICAL DEBT` + removal criterion.
- Benchmarks stay out of the training mixture (regression probes only).
