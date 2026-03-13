from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.runtime import assert_jax_runtime
from iris.train import ToyTrainConfig, load_policy_bundle, run_toy_training


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy baseline pretrain entrypoint. "
            "Active v2 claims require a validated five-pool data policy bundle."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase_e_pretrain"))
    parser.add_argument("--run-id", type=str, default="phase-e-pretrain")
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--micro-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--tokens-per-micro-step", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="jax", choices=["jax"])
    parser.add_argument("--level-impl", type=str, default="jax_transition", choices=["mounted", "jax_transition"])
    parser.add_argument("--strict-jax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--level-alpha", type=float, default=0.1)
    parser.add_argument(
        "--data-source",
        type=str,
        default="hybrid_mixture",
        choices=["pure_lm_streaming", "hybrid_mixture"],
    )
    parser.add_argument(
        "--pure-lm-profile",
        type=Path,
        default=None,
        help="Path to Pure LM profile JSON (default uses built-in pure_lm_90_v1).",
    )
    parser.add_argument(
        "--data-policy-bundle",
        type=Path,
        default=None,
        help="Optional path to an active-v2 iris.data_policy_bundle/v1 JSON for validation and reporting.",
    )
    parser.add_argument("--tokenizer-id-or-path", type=str, required=True)
    parser.add_argument(
        "--streaming-mode",
        type=str,
        default="auto",
        choices=["auto", "hf_online", "local_snapshot"],
    )
    parser.add_argument("--snapshot-root", type=Path, default=None)
    parser.add_argument("--hybrid-pure-ratio", type=float, default=0.9)
    parser.add_argument("--baseline-id", type=str, default="phase-e-v1")
    parser.add_argument("--tolerance-profile-id", type=str, default="phase-e-default")
    parser.add_argument("--resume-path-id", type=str, default="uninterrupted")
    parser.add_argument(
        "--runtime-lock-manifest",
        type=Path,
        default=None,
        help="Path to a pinned runtime_lock_manifest.json for strict S8 replay.",
    )

    args = parser.parse_args()

    policy_bundle = None
    if args.data_policy_bundle is not None:
        try:
            policy_bundle = load_policy_bundle(args.data_policy_bundle)
        except Exception as error:
            print(f"DATA POLICY BUNDLE INVALID: {error}", file=sys.stderr)
            return 2

    assert_jax_runtime(
        device=args.device,
        require_gpu=bool(args.strict_jax and str(args.device).lower() == "gpu"),
    )

    config = ToyTrainConfig(
        output_dir=args.output_dir,
        run_id=args.run_id,
        segments=args.segments,
        micro_steps=args.micro_steps,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        data_seed=args.data_seed,
        device=args.device,
        phase="E",
        baseline_id=args.baseline_id,
        tolerance_profile_id=args.tolerance_profile_id,
        backend=args.backend,
        strict_jax=args.strict_jax,
        level_impl=args.level_impl,
        level_alpha=args.level_alpha,
        data_source=args.data_source,
        pure_lm_profile=args.pure_lm_profile,
        tokenizer_id_or_path=args.tokenizer_id_or_path,
        streaming_mode=args.streaming_mode,
        snapshot_root=args.snapshot_root,
        tokens_per_micro_step=args.tokens_per_micro_step,
        hybrid_pure_ratio=args.hybrid_pure_ratio,
        resume_path_id=args.resume_path_id,
        runtime_lock_manifest_path=args.runtime_lock_manifest,
    )

    try:
        summary = run_toy_training(config)
    except RuntimeError as error:
        print(f"TRAIN INTERRUPTED: {error}", file=sys.stderr)
        return 2

    if policy_bundle is not None:
        summary["data_policy_bundle_sha256"] = policy_bundle.bundle_sha256
        summary["data_realization_policy_id"] = (
            policy_bundle.data_realization_policy.data_realization_policy_id
        )
        summary["decontam_policy_id"] = policy_bundle.decontam_policy.decontam_policy_id

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
