from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.regression.p1_readiness import (
    _read_json,
    _write_json,
    append_readiness_history,
    build_p1_readiness_report,
    build_p1_readiness_snapshot,
    evaluate_p1_readiness,
)


def _resolve_roots(
    *,
    closure_root: Path | None,
    phase_d_dir: Path | None,
    phase_e_dir: Path | None,
    model_run_dir: Path | None,
) -> tuple[Path, Path, Path | None]:
    if closure_root is not None:
        closure_root = Path(closure_root)
        phase_d_dir = phase_d_dir or (closure_root / "phase_d")
        phase_e_dir = phase_e_dir or (closure_root / "phase_e")
        model_run_dir = model_run_dir or (closure_root / "work" / "model_run")
    if phase_d_dir is None or phase_e_dir is None:
        raise SystemExit("Provide --closure-root or both --phase-d-dir and --phase-e-dir.")
    return Path(phase_d_dir), Path(phase_e_dir), Path(model_run_dir) if model_run_dir is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fixed-baseline P1 readiness packet from Phase D/E artifacts."
    )
    parser.add_argument("--closure-root", type=Path, default=None)
    parser.add_argument("--phase-d-dir", type=Path, default=None)
    parser.add_argument("--phase-e-dir", type=Path, default=None)
    parser.add_argument("--model-run-dir", type=Path, default=None)
    parser.add_argument("--leakage-audit", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/p1_readiness_review"))
    parser.add_argument("--baseline-packet", type=Path, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--baseline-id", type=str, default="p1-readiness-fixed-baseline")
    parser.add_argument("--tolerance-profile-id", type=str, default="tp_p1_bootstrap")
    parser.add_argument(
        "--freeze-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the current readiness snapshot as the fixed baseline packet.",
    )
    parser.add_argument(
        "--hard-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when the current run is not a gate-passed readiness packet.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_packet = Path(args.baseline_packet) if args.baseline_packet is not None else (output_dir / "p1_baseline_snapshot.json")
    history_path = Path(args.history_path) if args.history_path is not None else (output_dir / "p1_readiness_history.jsonl")
    phase_d_dir, phase_e_dir, model_run_dir = _resolve_roots(
        closure_root=args.closure_root,
        phase_d_dir=args.phase_d_dir,
        phase_e_dir=args.phase_e_dir,
        model_run_dir=args.model_run_dir,
    )

    snapshot = build_p1_readiness_snapshot(
        phase_d_root=phase_d_dir,
        phase_e_root=phase_e_dir,
        model_run_dir=model_run_dir,
        leakage_audit_path=args.leakage_audit,
        baseline_id=args.baseline_id,
        tolerance_profile_id=args.tolerance_profile_id,
    )
    snapshot_path = output_dir / "p1_readiness_snapshot.json"
    _write_json(snapshot_path, snapshot)

    if args.freeze_baseline:
        _write_json(baseline_packet, snapshot)
        result = {
            "schema": "iris.readiness.p1_baseline_capture/v1",
            "status": "BASELINE_CAPTURED",
            "baseline_packet": str(baseline_packet.resolve()),
            "snapshot_path": str(snapshot_path.resolve()),
            "baseline_id": args.baseline_id,
            "tolerance_profile_id": args.tolerance_profile_id,
        }
        _write_json(output_dir / "p1_readiness_packet.json", result)
        (output_dir / "p1_readiness_packet.md").write_text(
            "# P1 Readiness Baseline\n\n"
            f"- Baseline packet: {baseline_packet.resolve()}\n"
            f"- Snapshot path: {snapshot_path.resolve()}\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if not baseline_packet.exists():
        raise SystemExit(
            "Baseline packet does not exist. Run once with --freeze-baseline to capture a fixed baseline snapshot."
        )

    baseline_snapshot = _read_json(baseline_packet)
    history_rows = []
    if history_path.exists():
        from iris.regression.p1_readiness import _read_jsonl  # local import to keep CLI path narrow

        history_rows = _read_jsonl(history_path)

    packet = evaluate_p1_readiness(
        snapshot=snapshot,
        baseline_snapshot=baseline_snapshot,
        history_rows=history_rows,
    )
    packet_path = output_dir / "p1_readiness_packet.json"
    _write_json(packet_path, packet)
    (output_dir / "p1_readiness_packet.md").write_text(
        build_p1_readiness_report(packet),
        encoding="utf-8",
    )
    append_readiness_history(history_path, packet)

    summary = {
        "status": packet.get("promotion_status", "BLOCKED"),
        "run_gate_status": packet.get("run_gate_status", "FAIL"),
        "packet_path": str(packet_path.resolve()),
        "baseline_packet": str(baseline_packet.resolve()),
        "history_path": str(history_path.resolve()),
        "consecutive_gate_passed_runs": packet.get("consecutive_gate_passed_runs", 0),
    }
    print(json.dumps(summary, sort_keys=True))
    if args.hard_fail:
        return 0 if str(packet.get("run_gate_status", "FAIL")).upper() == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
