from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from iris.levels import LevelInput, build_level_stack
from iris.metrics import build_canonical_metrics, neutral_failure_credit
from iris.schema import AdjudicationState, BudgetState, ControlAction, ControlState, ProblemFrame, RequiredOutput, StateIR
from iris.train import build_document_pipeline_bundle, load_policy_bundle_for_profile_phase, resolve_learning_objective_bundle


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "p1_phase_c"


def _initial_state(hidden_dim: int) -> StateIR:
    zero = np.zeros((hidden_dim,), dtype=np.float32)
    return StateIR(
        PF=ProblemFrame(
            task_type="prove",
            target_spec="bootstrap smoke proof",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 smoke check for P1 semantic mounted closure.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    fixture_root = _fixture_root()
    document_bundle = build_document_pipeline_bundle(
        fixture_root / "documents" / "bootstrap_identity.pdf",
        sidecar_path=fixture_root / "documents" / "bootstrap_identity.pdf.sidecar.json",
    )
    policy_bundle = load_policy_bundle_for_profile_phase("P1", "C")
    learning_bundle, _ = resolve_learning_objective_bundle(profile_id="P1", phase="C")

    level_stack = build_level_stack(
        implementation="mounted",
        hidden_dim=args.hidden_dim,
        seed=0,
    )
    current_state = _initial_state(hidden_dim=args.hidden_dim)
    l6_diagnostics = {}
    for level_id in [f"L{idx}" for idx in range(7)]:
        level_output = level_stack[level_id].run(
            LevelInput(
                state_in=current_state,
                context_in={
                    "document_bundle": document_bundle,
                    "benchmark_family_policy": policy_bundle.benchmark_family_policies["miniF2F-v1"],
                    "task_family_override": "formalization" if level_id == "L6" else None,
                },
            )
        )
        current_state = level_output.state_out
        if level_output.diagnostics.get("implementation_status") != "mounted":
            print(f"S1 FAIL: {level_id} did not report semantic mounted status.", file=sys.stderr)
            return 1
        if int(level_output.diagnostics.get("contract_mutation_count", 0)) <= 0:
            print(f"S1 FAIL: {level_id} did not report contract-bearing mutation.", file=sys.stderr)
            return 1
        if level_id == "L6":
            l6_diagnostics = dict(level_output.diagnostics)

    metrics = build_canonical_metrics(
        state=current_state,
        failure_credit=l6_diagnostics.get("failure.credit") or neutral_failure_credit(),
        task_validity_score=float(l6_diagnostics.get("task.validity_score", 0.0)),
        task_confidence=float(l6_diagnostics.get("task.confidence", 0.0)),
        extra={
            "phase": "C",
            "suite": "S1",
            "task.document_grounding_score": l6_diagnostics.get("task.document_grounding_score", 0.0),
            "provenance.parser_coverage": 1.0,
            "provenance.verifier_coverage": 1.0,
            "learning_objective_bundle_id": learning_bundle.learning_objective_bundle_id,
            "task_family": l6_diagnostics.get("task_family", ""),
            "task_adjudication_policy_id": l6_diagnostics.get("task_adjudication_policy_id", ""),
        },
    )

    sequence = current_state.to_canonical_sequence()
    if not np.isfinite(sequence).all():
        print("S1 FAIL: non-finite values detected.", file=sys.stderr)
        return 1
    if current_state.CS.runtime_status != "accepted":
        print("S1 FAIL: runtime did not reach canonical accepted status.", file=sys.stderr)
        return 1

    report = {
        "status": "PASS",
        "suite": "S1",
        "token_count": current_state.total_tokens,
        "runtime_status": current_state.CS.runtime_status,
        "task_adjudication_policy_id": current_state.CS.adjudication_state.task_adjudication_policy_id if current_state.CS.adjudication_state else "",
        "learning_objective_bundle_id": learning_bundle.learning_objective_bundle_id,
        "metrics_subset": {
            "task.validity_score": metrics["task.validity_score"],
            "task.confidence": metrics["task.confidence"],
            "task.document_grounding_score": metrics["task.document_grounding_score"],
        },
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
