from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from iris.levels import LevelInput, build_level_stack
from iris.schema import AdjudicationState, BudgetState, ControlAction, ControlState, ProblemFrame, RequiredOutput, STATE_IR_TOKEN_ORDER, StateIR
from iris.train import build_document_pipeline_bundle, load_policy_bundle_for_profile_phase


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "p1_phase_c"


def _build_state(hidden_dim: int) -> StateIR:
    zero = np.zeros((hidden_dim,), dtype=np.float32)
    return StateIR(
        PF=ProblemFrame(
            task_type="prove",
            target_spec="bootstrap document proof",
            required_output=RequiredOutput(
                output_kind="proof",
                answer_channel="structured_object",
                formality_level="informal",
                verifier_mode="proof_gap_plus_local_validity",
            ),
            problem_assumptions=(),
            domain_tags=("bootstrap",),
            source_anchor_refs=(),
            frame_status="draft",
            vector=zero,
        ),
        CS=ControlState(
            selected_action=ControlAction(action_id="action-0", action_type="continue"),
            budget_state=BudgetState(
                global_step_budget_remaining=4,
                branch_expansion_budget_remaining=2,
                verifier_probe_budget_remaining=2,
                reparse_budget_remaining=1,
            ),
            runtime_status="in_progress",
            uncertainty_state="unknown",
            escalation_state="inactive",
            adjudication_state=AdjudicationState(
                task_adjudication_policy_id="task-family-proof-natural-language-default-v1",
                adjudication_status="pending",
            ),
            vector=zero,
        ),
    )


def _validate_l6_credit(diagnostics: dict) -> bool:
    credit = diagnostics.get("failure.credit")
    if not isinstance(credit, dict):
        return False
    keys = sorted(credit.keys())
    expected = [f"L{i}" for i in range(7)]
    if keys != expected:
        return False
    values = np.asarray([float(credit[level]) for level in expected], dtype=np.float64)
    return bool(np.all(values >= 0.0) and np.all(values <= 1.0) and np.isclose(np.sum(values), 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="S2M semantic mounted structural check for IRIS.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    fixture_root = _fixture_root()
    document_bundle = build_document_pipeline_bundle(
        fixture_root / "documents" / "bootstrap_identity.pdf",
        sidecar_path=fixture_root / "documents" / "bootstrap_identity.pdf.sidecar.json",
    )
    benchmark_policy = load_policy_bundle_for_profile_phase("P1", "C").benchmark_family_policies["miniF2F-v1"]

    state = _build_state(hidden_dim=args.hidden_dim)
    stack = build_level_stack(
        implementation="mounted",
        hidden_dim=args.hidden_dim,
        seed=0,
    )
    if sorted(stack.keys()) != [f"L{i}" for i in range(7)]:
        print("S2M FAIL: L0-L6 interfaces are incomplete.", file=sys.stderr)
        return 1

    current_state = state
    for level_id in [f"L{i}" for i in range(7)]:
        output = stack[level_id].run(
            LevelInput(
                state_in=current_state,
                context_in={
                    "document_bundle": document_bundle,
                    "benchmark_family_policy": benchmark_policy,
                    "task_family_override": "formalization" if level_id == "L6" else None,
                },
            )
        )
        current_state = output.state_out
        if output.diagnostics.get("disabled", True):
            print(f"S2M FAIL: {level_id} should be mounted (disabled=False).", file=sys.stderr)
            return 1
        if output.control_out.get("mode") != "mounted":
            print(f"S2M FAIL: {level_id} did not emit mounted control mode.", file=sys.stderr)
            return 1
        if int(output.diagnostics.get("contract_mutation_count", 0)) <= 0:
            print(f"S2M FAIL: {level_id} did not report contract-bearing mutation.", file=sys.stderr)
            return 1
        if level_id == "L6" and not _validate_l6_credit(output.diagnostics):
            print("S2M FAIL: L6 failure.credit is invalid.", file=sys.stderr)
            return 1

    sequence = current_state.to_canonical_sequence()
    if not np.isfinite(sequence).all():
        print("S2M FAIL: non-finite values detected.", file=sys.stderr)
        return 1
    if current_state.CS.runtime_status != "accepted":
        print("S2M FAIL: L6 did not produce canonical accepted runtime status.", file=sys.stderr)
        return 1
    if current_state.CS.adjudication_state is None or current_state.CS.adjudication_state.adjudication_status != "accepted":
        print("S2M FAIL: L6 did not produce canonical accepted adjudication status.", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "suite": "S2M",
                "state_ir_token_order": STATE_IR_TOKEN_ORDER,
                "levels": [f"L{i}" for i in range(7)],
                "runtime_status": current_state.CS.runtime_status,
                "task_adjudication_policy_id": current_state.CS.adjudication_state.task_adjudication_policy_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
