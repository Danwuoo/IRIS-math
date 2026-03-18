from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.train import (
    HFRepoSpec,
    bootstrap_checkpoint_run,
    default_iris3b_config,
    iris3b_config_from_mapping,
    sync_checkpoint_run,
    sync_final_release,
    write_latest_pointer,
)


def _load_model_config(path: Path | None):
    if path is None:
        return default_iris3b_config()
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Model config must be a JSON object: {path}")
    return iris3b_config_from_mapping(payload)


def _configure_cache_env(cache_root: Path | None) -> None:
    if cache_root is None:
        return
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hf" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf" / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "hf" / "transformers"))


def _maybe_bootstrap_from_hf(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any((output_dir / name).exists() for name in ("segment_journal.jsonl", "checkpoints")):
        return
    if not args.download_latest:
        return
    pointer = bootstrap_checkpoint_run(
        repo=HFRepoSpec(repo_id=args.hf_checkpoint_repo_id),
        local_dir=output_dir,
        token=args.hf_token,
    )
    if pointer is not None:
        print(json.dumps({"bootstrap_status": "restored", "pointer": pointer}, sort_keys=True))


def _build_cycle_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cycle", help="Run one governed Kaggle training cycle.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/p1_3b"))
    parser.add_argument("--run-id", type=str, default="p1-iris3b-kaggle")
    parser.add_argument("--hf-checkpoint-repo-id", type=str, default="Danwuoo/IRIS-math")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--model-config-json", type=Path, default=None)
    parser.add_argument("--streaming-mode", type=str, default="auto", choices=["auto", "hf_online", "local_snapshot"])
    parser.add_argument("--snapshot-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--dataset-cache-limit-gib", type=int, default=50)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-workdir", type=Path, default=None)
    parser.add_argument("--max-cycle-minutes", type=int, default=350)
    parser.add_argument("--max-segments", type=int, default=10_000)
    parser.add_argument("--runtime-lock-manifest", type=Path, default=None)
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--strict-jax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-path-id", type=str, default="uninterrupted")
    parser.add_argument("--crash-point", type=str, default="none", choices=["none", "execute", "pre_commit", "post_commit"])
    parser.add_argument("--crash-segment", type=int, default=-1)
    parser.add_argument("--baseline-id", type=str, default="p1-readiness-fixed-baseline")
    parser.add_argument("--tolerance-profile-id", type=str, default="tp_p1_bootstrap")
    parser.add_argument("--download-latest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(command="cycle")


def _build_finalize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("finalize", help="Export and optionally publish the final Flax release.")
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/p1_3b"))
    parser.add_argument("--checkpoint-manifest-path", type=Path, default=None)
    parser.add_argument("--release-dir", type=Path, default=None)
    parser.add_argument("--model-config-json", type=Path, default=None)
    parser.add_argument("--tokenizer-root", type=Path, default=None)
    parser.add_argument("--streaming-manifest-path", type=Path, default=None)
    parser.add_argument("--readiness-packet-path", type=Path, default=None)
    parser.add_argument("--readiness-history-path", type=Path, default=None)
    parser.add_argument("--hf-final-repo-id", type=str, default="")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--require-readiness-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(command="finalize")


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal P1 Kaggle H100 IRIS 3B training and release entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_cycle_parser(subparsers)
    _build_finalize_parser(subparsers)
    args = parser.parse_args()

    if args.command == "cycle":
        from iris.train.iris3b_training import P1TrainConfig, run_p1_training_cycle

        _configure_cache_env(args.cache_root)
        _maybe_bootstrap_from_hf(args)
        model_config = _load_model_config(args.model_config_json)
        summary = run_p1_training_cycle(
            P1TrainConfig(
                output_dir=args.output_dir,
                run_id=args.run_id,
                baseline_id=args.baseline_id,
                tolerance_profile_id=args.tolerance_profile_id,
                manifest_path=args.manifest_path,
                streaming_mode=args.streaming_mode,
                cache_root=args.cache_root,
                snapshot_root=args.snapshot_root,
                tokenizer_dir=args.tokenizer_dir,
                tokenizer_workdir=args.tokenizer_workdir,
                max_cycle_minutes=args.max_cycle_minutes,
                max_segments=args.max_segments,
                dataset_cache_limit_gib=args.dataset_cache_limit_gib,
                hf_token=args.hf_token,
                runtime_lock_manifest_path=args.runtime_lock_manifest,
                device=args.device,
                strict_jax=args.strict_jax,
                resume_path_id=args.resume_path_id,
                crash_point=args.crash_point,
                crash_segment=args.crash_segment,
                model_config=model_config,
            )
        )
        latest_pointer_path = Path(args.output_dir) / "hf_latest_pointer.json"
        checkpoint_manifest_rel = Path(summary["checkpoint_manifest_path"]).resolve().relative_to(Path(args.output_dir).resolve())
        write_latest_pointer(
            output_path=latest_pointer_path,
            run_id=args.run_id,
            segment_id=int(summary["last_segment_id"]),
            checkpoint_manifest_path=f"checkpoints/{args.run_id}/{checkpoint_manifest_rel.as_posix()}",
            extra={
                "streaming_manifest_sha256": summary["streaming_manifest_sha256"],
                "tokenizer_manifest_ref": summary["tokenizer_manifest_ref"],
                "requested_streaming_mode": summary["requested_streaming_mode"],
                "effective_streaming_mode": summary["effective_streaming_mode"],
                "local_snapshot_manifest_ref": summary["local_snapshot_manifest_ref"],
            },
        )
        if args.sync_checkpoint:
            sync_checkpoint_run(
                repo=HFRepoSpec(repo_id=args.hf_checkpoint_repo_id),
                run_dir=args.output_dir,
                run_id=args.run_id,
                latest_pointer_path=latest_pointer_path,
                token=args.hf_token,
            )
        summary["hf_latest_pointer_path"] = str(latest_pointer_path)
        print(json.dumps(summary, sort_keys=True))
        return 0

    from iris.train.iris3b_training import export_final_release

    model_config = _load_model_config(args.model_config_json)
    run_dir = Path(args.run_dir)
    release_dir = Path(args.release_dir) if args.release_dir is not None else (run_dir / "final_release")
    readiness_packet_path = args.readiness_packet_path or (run_dir / "readiness" / "p1_readiness_packet.json")
    readiness_history_path = args.readiness_history_path or (run_dir / "readiness" / "p1_readiness_history.jsonl")
    if args.require_readiness_pass:
        if not Path(readiness_packet_path).exists():
            raise SystemExit(f"Readiness packet is required before final export: {readiness_packet_path}")
        readiness_packet = json.loads(Path(readiness_packet_path).read_text(encoding="utf-8-sig"))
        if str(readiness_packet.get("run_gate_status", "FAIL")).upper() != "PASS":
            raise SystemExit("Final export blocked: readiness packet run_gate_status is not PASS.")
        if str(readiness_packet.get("promotion_status", "BLOCKED")).upper() != "PASS":
            raise SystemExit("Final export blocked: readiness packet promotion_status is not PASS.")

    tokenizer_root = args.tokenizer_root or (run_dir / "tokenizer" / "iris_p1_tokenizer")
    streaming_manifest_path = args.streaming_manifest_path or (run_dir / "data" / "p1_streaming_manifest_committed.json")
    summary = export_final_release(
        run_dir=run_dir,
        checkpoint_manifest_path=args.checkpoint_manifest_path,
        release_dir=release_dir,
        model_config=model_config,
        tokenizer_root=tokenizer_root,
        readiness_packet_path=readiness_packet_path,
        readiness_history_path=readiness_history_path,
        streaming_manifest_path=streaming_manifest_path,
    )
    if str(args.hf_final_repo_id).strip():
        sync_final_release(
            repo=HFRepoSpec(repo_id=str(args.hf_final_repo_id).strip()),
            release_dir=release_dir,
            token=args.hf_token,
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _fast_exit(exit_code: int) -> None:
    # TEMPORARY TECHNICAL DEBT: use os._exit to avoid interpreter-finalization crashes
    # observed after HF streaming teardown under Python 3.12. Remove once datasets/pyarrow
    # shutdown is stable in the Kaggle and local WSL training environments.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))


if __name__ == "__main__":
    try:
        code = int(main())
    except SystemExit as error:
        if isinstance(error.code, int):
            code = int(error.code)
        else:
            if error.code not in (None, ""):
                print(str(error.code), file=sys.stderr)
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    _fast_exit(0 if code in (None, 0) else int(code))
