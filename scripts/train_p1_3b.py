from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.train import (
    HFRepoSpec,
    bootstrap_checkpoint_run,
    cycle_memory_profile_candidates,
    iris3b_config_from_mapping,
    select_cycle_iris3b_config,
    sync_checkpoint_run,
    sync_final_release,
    write_latest_pointer,
)
from iris.train.journal import load_journal


@dataclass(frozen=True)
class HostParallelismConfig:
    host_cpu_threads: int
    batch_prefetch: int
    tokenizer_corpus_workers: int
    sentencepiece_threads: int
    hf_parallel_loading_workers: int


def _kaggle_runtime_present() -> bool:
    if any(str(os.environ.get(key, "")).strip() for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")):
        return True
    return any(Path(path).exists() for path in ("/kaggle/working", "/kaggle/temp"))


def _load_model_config(path: Path | None, *, memory_profile: str = "auto", kaggle_runtime: bool | None = None):
    effective_kaggle_runtime = _kaggle_runtime_present() if kaggle_runtime is None else bool(kaggle_runtime)
    if path is None:
        return select_cycle_iris3b_config(
            explicit_config=None,
            memory_profile=memory_profile,
            kaggle_runtime=effective_kaggle_runtime,
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Model config must be a JSON object: {path}")
    explicit_config = iris3b_config_from_mapping(payload)
    return select_cycle_iris3b_config(
        explicit_config=explicit_config,
        memory_profile=memory_profile,
        kaggle_runtime=effective_kaggle_runtime,
    )


def _looks_like_device_oom(error: BaseException) -> bool:
    message = str(error or "").lower()
    return ("resource_exhausted" in message) or ("out of memory" in message) or ("oom" in message)


def _run_has_applied_progress(output_dir: Path) -> bool:
    journal_path = Path(output_dir) / "segment_journal.jsonl"
    if not journal_path.exists():
        return False
    return any(str(event.get("status", "")).upper() == "APPLIED" for event in load_journal(journal_path))


def _cleanup_after_failed_device_oom(error: BaseException) -> None:
    tb = getattr(error, "__traceback__", None)
    if tb is not None:
        try:
            traceback.clear_frames(tb)
        except Exception:
            pass
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def _configure_cache_env(cache_root: Path | None) -> None:
    if cache_root is None:
        return
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hf" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf" / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "hf" / "transformers"))


def _configure_jax_memory_env(*, kaggle_runtime: bool) -> None:
    if not kaggle_runtime:
        return
    # TEMPORARY TECHNICAL DEBT: keep the Kaggle P1 institution run on the
    # default 1x-H100 path from failing due to JAX preallocation and the full
    # 2048-token activation footprint. Remove once the governed 3B training stack
    # reliably fits the default cycle config on Kaggle-class H100 runtimes.
    # Intended replacement: the unmodified default_iris3b_config() cycle path.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")


def _resolve_cache_free_space_floor_gib(*, kaggle_runtime: bool, requested_floor_gib: int) -> int:
    requested = max(int(requested_floor_gib), 0)
    if requested > 0:
        return requested
    if not kaggle_runtime:
        return 0
    # TEMPORARY TECHNICAL DEBT: reserve free space headroom on Kaggle-class
    # ephemeral disks so cache/snapshot growth does not exhaust the runtime
    # volume before the next learned training segment boundary. Remove once the
    # runtime has profile-governed storage budgeting with artifact-aware
    # pressure signals instead of this fixed floor.
    # Intended replacement: policy-governed runtime storage budgeting.
    return 12


def _resolve_snapshot_fallback_root(
    *,
    kaggle_runtime: bool,
    requested_root: Path | None,
) -> Path | None:
    if requested_root is not None:
        return Path(requested_root)
    if not kaggle_runtime:
        return None
    # TEMPORARY TECHNICAL DEBT: spill local snapshot materialization into a
    # dedicated /kaggle/working subtree so reproducible-but-rebuildable
    # snapshot artifacts do not crowd the primary run/checkpoint directory.
    # Remove once runtime storage placement is governed by a profile-aware
    # artifact policy instead of this fixed Kaggle fallback path.
    # Intended replacement: policy-governed artifact placement for training IO.
    return Path("/kaggle/working/iris_p1_spill/snapshots")


def _resolve_checkpoint_retention_limit(*, kaggle_runtime: bool, requested_limit: int) -> int:
    requested = int(requested_limit)
    if requested > 0:
        return requested
    if not kaggle_runtime:
        return 0
    # TEMPORARY TECHNICAL DEBT: cap retained checkpoint payloads on Kaggle so
    # ephemeral storage is not exhausted by fully governed APPLIED payloads
    # accumulating faster than the next cycle can export or sync them. Remove
    # once checkpoint placement/retention is policy-governed by artifact-aware
    # runtime storage management instead of this fixed fallback cap.
    # Intended replacement: policy-governed checkpoint artifact lifecycle.
    return 1


def _available_cpu_count() -> int:
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except Exception:
        return max(int(os.cpu_count() or 1), 1)


def _resolve_host_parallelism(*, kaggle_runtime: bool, args: argparse.Namespace) -> HostParallelismConfig:
    cpu_count = _available_cpu_count()
    auto_host_threads = max(1, cpu_count - (4 if kaggle_runtime else 1))
    host_cpu_threads = int(args.host_cpu_threads) if int(args.host_cpu_threads) > 0 else auto_host_threads
    host_cpu_threads = max(1, min(host_cpu_threads, cpu_count))

    default_batch_prefetch = min(max(host_cpu_threads // 2, 2), 12)
    batch_prefetch = int(args.batch_prefetch) if int(args.batch_prefetch) > 0 else default_batch_prefetch
    batch_prefetch = max(1, min(batch_prefetch, host_cpu_threads))

    default_corpus_workers = min(host_cpu_threads, 8)
    tokenizer_corpus_workers = (
        int(args.tokenizer_corpus_workers) if int(args.tokenizer_corpus_workers) > 0 else default_corpus_workers
    )
    tokenizer_corpus_workers = max(1, min(tokenizer_corpus_workers, host_cpu_threads))

    default_sentencepiece_threads = min(host_cpu_threads, 16)
    sentencepiece_threads = (
        int(args.sentencepiece_threads) if int(args.sentencepiece_threads) > 0 else default_sentencepiece_threads
    )
    sentencepiece_threads = max(1, min(sentencepiece_threads, host_cpu_threads))

    hf_parallel_loading_workers = max(1, min(host_cpu_threads, 8))
    return HostParallelismConfig(
        host_cpu_threads=host_cpu_threads,
        batch_prefetch=batch_prefetch,
        tokenizer_corpus_workers=tokenizer_corpus_workers,
        sentencepiece_threads=sentencepiece_threads,
        hf_parallel_loading_workers=hf_parallel_loading_workers,
    )


def _configure_host_parallelism_env(parallelism: HostParallelismConfig) -> None:
    host_threads = max(int(parallelism.host_cpu_threads), 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(host_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(host_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(host_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(host_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(host_threads))
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", str(parallelism.hf_parallel_loading_workers))
    if not str(os.environ.get("XLA_FLAGS", "")).strip():
        os.environ["XLA_FLAGS"] = (
            f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={host_threads}"
        )


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
    parser.add_argument("--snapshot-fallback-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--dataset-cache-limit-gib", type=int, default=50)
    parser.add_argument("--cache-free-space-floor-gib", type=int, default=0)
    parser.add_argument("--checkpoint-retention-limit", type=int, default=0)
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
    parser.add_argument(
        "--memory-profile",
        type=str,
        default="auto",
        choices=["auto", "default", "kaggle_safe", "kaggle_safer", "kaggle_emergency", "kaggle_survival"],
    )
    parser.add_argument("--host-cpu-threads", type=int, default=0)
    parser.add_argument("--batch-prefetch", type=int, default=0)
    parser.add_argument("--tokenizer-corpus-workers", type=int, default=0)
    parser.add_argument("--sentencepiece-threads", type=int, default=0)
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

        kaggle_runtime = _kaggle_runtime_present()
        cache_free_space_floor_gib = _resolve_cache_free_space_floor_gib(
            kaggle_runtime=kaggle_runtime,
            requested_floor_gib=args.cache_free_space_floor_gib,
        )
        snapshot_fallback_root = _resolve_snapshot_fallback_root(
            kaggle_runtime=kaggle_runtime,
            requested_root=args.snapshot_fallback_root,
        )
        checkpoint_retention_limit = _resolve_checkpoint_retention_limit(
            kaggle_runtime=kaggle_runtime,
            requested_limit=args.checkpoint_retention_limit,
        )
        parallelism = _resolve_host_parallelism(kaggle_runtime=kaggle_runtime, args=args)
        _configure_cache_env(args.cache_root)
        _configure_host_parallelism_env(parallelism)
        _configure_jax_memory_env(kaggle_runtime=kaggle_runtime)
        _maybe_bootstrap_from_hf(args)
        attempted_profiles = []
        candidate_profiles = cycle_memory_profile_candidates(
            memory_profile=args.memory_profile,
            kaggle_runtime=kaggle_runtime,
        )
        summary = None
        last_error: BaseException | None = None
        for attempt_index, candidate_profile in enumerate(candidate_profiles):
            attempted_profiles.append(candidate_profile)
            model_config = _load_model_config(
                args.model_config_json,
                memory_profile=candidate_profile,
                kaggle_runtime=kaggle_runtime,
            )
            try:
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
                        snapshot_fallback_root=snapshot_fallback_root,
                        tokenizer_dir=args.tokenizer_dir,
                        tokenizer_workdir=args.tokenizer_workdir,
                        host_cpu_threads=parallelism.host_cpu_threads,
                        batch_prefetch=parallelism.batch_prefetch,
                        tokenizer_corpus_workers=parallelism.tokenizer_corpus_workers,
                        sentencepiece_threads=parallelism.sentencepiece_threads,
                        max_cycle_minutes=args.max_cycle_minutes,
                        max_segments=args.max_segments,
                        dataset_cache_limit_gib=args.dataset_cache_limit_gib,
                        cache_free_space_floor_gib=cache_free_space_floor_gib,
                        checkpoint_retention_limit=checkpoint_retention_limit,
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
                break
            except BaseException as error:
                remaining_candidates = candidate_profiles[attempt_index + 1 :]
                can_retry = (
                    bool(remaining_candidates)
                    and _looks_like_device_oom(error)
                    and not _run_has_applied_progress(args.output_dir)
                )
                if not can_retry:
                    last_error = error
                    raise
                _cleanup_after_failed_device_oom(error)
        if summary is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Training cycle exited without producing a summary.")
        summary["host_parallelism"] = {
            "host_cpu_threads": parallelism.host_cpu_threads,
            "batch_prefetch": parallelism.batch_prefetch,
            "tokenizer_corpus_workers": parallelism.tokenizer_corpus_workers,
            "sentencepiece_threads": parallelism.sentencepiece_threads,
        }
        summary["requested_memory_profile"] = str(args.memory_profile)
        summary["attempted_memory_profiles"] = [str(profile) for profile in attempted_profiles]
        summary["effective_memory_profile"] = str(attempted_profiles[-1])
        summary["memory_profile_fallback_used"] = bool(len(attempted_profiles) > 1)
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

    model_config = _load_model_config(args.model_config_json, memory_profile="default")
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
