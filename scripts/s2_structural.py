from __future__ import annotations

import json
import sys

import numpy as np

import _bootstrap  # noqa: F401
from iris.levels import LevelInput, build_level_stack
from iris.runtime import resolve_task_semantics
from iris.schema import (
    AdjudicationState,
    BudgetState,
    ControlAction,
    ControlState,
    ProblemFrame,
    RequiredOutput,
    STATE_IR_TOKEN_ORDER,
    ScopeRef,
    StateIR,
    StateIRValidationError,
)
from iris.train import load_policy_bundle_for_profile_phase, resolve_learning_objective_bundle


def _build_state(hidden_dim: int) -> StateIR:
    zero = np.zeros((hidden_dim,), dtype=np.float32)
    return StateIR(
        PF=ProblemFrame(
            task_type="prove",
            target_spec="prove x = x",
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
            budget_state=BudgetState(global_step_budget_remaining=2),
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


def main() -> int:
    state = _build_state(hidden_dim=8)

    try:
        StateIR.from_ordered_sections(
            [
                ("SY", state.SY),
                ("PF", state.PF),
                ("CG", state.CG),
                ("FR", state.FR),
                ("LM", state.LM),
                ("VS", state.VS),
                ("CS", state.CS),
            ]
        )
        print("S2 FAIL: ordering violation was not rejected.", file=sys.stderr)
        return 1
    except StateIRValidationError:
        pass

    try:
        StateIR.from_token_map(
            {
                "PF": state.PF,
                "SY": state.SY,
                "CG": state.CG,
                "FR": state.FR,
                "LM": state.LM,
                "VS": state.VS,
                "CS": state.CS,
                "Q": (),
            }
        )
        print("S2 FAIL: unknown token category was not rejected.", file=sys.stderr)
        return 1
    except StateIRValidationError:
        pass

    if ScopeRef(scope_kind="global", scope_id="scope-0").scope_kind != "problem_global":
        print("S2 FAIL: legacy scope alias was not normalized.", file=sys.stderr)
        return 1

    bundle = load_policy_bundle_for_profile_phase("P1", "C")
    learning_bundle, bundle_source = resolve_learning_objective_bundle(profile_id="P1", phase="C")
    task_semantics = resolve_task_semantics(
        state.PF,
        benchmark_family_policy=bundle.benchmark_family_policies["miniF2F-v1"],
        item_task_family="formalization",
    )
    if not learning_bundle.learning_objective_bundle_id or bundle_source != "profile_phase_default":
        print("S2 FAIL: learning objective bundle did not resolve canonically.", file=sys.stderr)
        return 1
    if task_semantics.task_adjudication_policy_id != "minif2f-formalization-tight-v1":
        print("S2 FAIL: task adjudication policy did not resolve canonically.", file=sys.stderr)
        return 1

    level_stack = build_level_stack(implementation="stub")
    if sorted(level_stack.keys()) != [f"L{i}" for i in range(7)]:
        print("S2 FAIL: L0-L6 interfaces are incomplete.", file=sys.stderr)
        return 1

    current_state = state
    diagnostics = {}
    for level_id in [f"L{i}" for i in range(7)]:
        result = level_stack[level_id].run(LevelInput(state_in=current_state))
        diagnostics[level_id] = result.diagnostics
        current_state = result.state_out
        if not result.diagnostics.get("disabled", False):
            print(f"S2 FAIL: {level_id} stub did not report disabled marker.", file=sys.stderr)
            return 1
        if result.control_out.get("mode") != "neutral":
            print(f"S2 FAIL: {level_id} stub did not emit neutral control.", file=sys.stderr)
            return 1

    report = {
        "status": "PASS",
        "suite": "S2",
        "state_ir_token_order": STATE_IR_TOKEN_ORDER,
        "levels": list(diagnostics.keys()),
        "learning_objective_bundle_id": learning_bundle.learning_objective_bundle_id,
        "task_adjudication_policy_id": task_semantics.task_adjudication_policy_id,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
