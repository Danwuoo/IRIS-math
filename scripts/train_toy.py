from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.runtime import assert_jax_runtime
from iris.train import ToyTrainConfig, run_toy_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Run toy IRIS training with segment journal.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/toy_train"))
    parser.add_argument("--run-id", type=str, default="toy-run")
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--micro-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--phase", type=str, default="C")
    parser.add_argument("--backend", type=str, default="jax", choices=["jax"])
    parser.add_argument(
        "--level-impl",
        type=str,
        default="jax_transition",
        choices=["mounted", "stub", "mixed", "jax_transition"],
    )
    parser.add_argument("--strict-jax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--level-alpha", type=float, default=0.1)
    parser.add_argument(
        "--data-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "pure_lm_streaming", "hybrid_mixture"],
    )
    parser.add_argument(
        "--pure-lm-profile",
        type=Path,
        default=None,
        help="Path to Pure LM profile JSON. Default uses src/iris/train/data/profiles/pure_lm_90_v1.json.",
    )
    parser.add_argument(
        "--tokenizer-id-or-path",
        type=str,
        default=None,
        help="Required for pure_lm_streaming or hybrid_mixture.",
    )
    parser.add_argument(
        "--streaming-mode",
        type=str,
        default="auto",
        choices=["auto", "hf_online", "local_snapshot"],
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=None,
        help="Local snapshot root for streaming_mode=local_snapshot or auto fallback.",
    )
    parser.add_argument("--tokens-per-micro-step", type=int, default=128)
    parser.add_argument("--hybrid-pure-ratio", type=float, default=0.9)
    parser.add_argument(
        "--crash-point",
        type=str,
        choices=["none", "execute", "pre_commit", "post_commit"],
        default="none",
    )
    parser.add_argument("--crash-segment", type=int, default=-1)
    parser.add_argument("--resume-path-id", type=str, default="uninterrupted")
    parser.add_argument(
        "--runtime-lock-manifest",
        type=Path,
        default=None,
        help="Path to a pinned runtime_lock_manifest.json to reuse across runs (strict S8 flow).",
    )
    args = parser.parse_args()
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
        phase=str(args.phase).strip().upper(),
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
        crash_point=args.crash_point,
        crash_segment=args.crash_segment,
        resume_path_id=args.resume_path_id,
        runtime_lock_manifest_path=args.runtime_lock_manifest,
    )

    try:
        summary = run_toy_training(config)
    except RuntimeError as error:
        print(f"TRAIN INTERRUPTED: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
