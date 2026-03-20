from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import iris.train.iris3b_training as iris3b_training_module


def test_resolve_checkpoint_payload_roots_prefers_shm_for_optimizer_and_temp_for_params(
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "scripts"))
    train_script = importlib.import_module("train_p1_3b")
    train_script = importlib.reload(train_script)

    free_bytes_by_path = {
        "/dev/shm/iris_p1_spill/checkpoints/payloads": 30,
        "/kaggle/temp/iris_p1_spill/checkpoints/payloads": 25,
        "/kaggle/working/iris_p1_spill/checkpoints/payloads": 10,
    }
    monkeypatch.setattr(
        train_script,
        "_candidate_free_bytes",
        lambda path: free_bytes_by_path.get(str(path), -1),
    )

    resolved = train_script._resolve_checkpoint_payload_roots(
        kaggle_runtime=True,
        requested_root=None,
        requested_params_root=None,
        requested_optimizer_root=None,
        requested_rng_root=None,
    )

    assert resolved == {
        "params": Path("/kaggle/temp/iris_p1_spill/checkpoints/payloads"),
        "optimizer_state": Path("/dev/shm/iris_p1_spill/checkpoints/payloads"),
        "rng_state": Path("/kaggle/temp/iris_p1_spill/checkpoints/payloads"),
    }


def test_maybe_bootstrap_from_hf_relocates_payloads_to_component_roots(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "scripts"))
    train_script = importlib.import_module("train_p1_3b")
    train_script = importlib.reload(train_script)

    output_dir = tmp_path / "bootstrapped_run"
    payload_root = tmp_path / "spill" / "payloads"

    def fake_bootstrap_checkpoint_run(*, repo, local_dir, token):
        del repo, token
        run_dir = Path(local_dir)
        for component in ("params", "optimizer_state", "rng_state"):
            payload_dir = run_dir / "checkpoints" / "payloads" / "segment_000007" / component
            payload_dir.mkdir(parents=True, exist_ok=True)
            (payload_dir / "tensor.bin").write_text(component, encoding="utf-8")
        (run_dir / "checkpoints" / "segment_000007.json").write_text("{}", encoding="utf-8")
        return {"run_id": "p1-run", "segment_id": 7}

    monkeypatch.setattr(train_script, "bootstrap_checkpoint_run", fake_bootstrap_checkpoint_run)
    monkeypatch.setattr(train_script, "_kaggle_runtime_present", lambda: True)

    args = SimpleNamespace(
        output_dir=output_dir,
        download_latest=True,
        hf_checkpoint_repo_id="Danwuoo/IRIS-math",
        hf_token="token",
        checkpoint_payload_root=None,
        checkpoint_params_root=payload_root,
        checkpoint_optimizer_root=tmp_path / "spill_optimizer" / "payloads",
        checkpoint_rng_root=payload_root,
    )

    train_script._maybe_bootstrap_from_hf(args)

    emitted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert emitted["bootstrap_status"] == "restored"
    assert emitted["checkpoint_payload_root"] == ""
    assert emitted["checkpoint_payload_roots"] == {
        "params": str(payload_root),
        "optimizer_state": str(tmp_path / "spill_optimizer" / "payloads"),
        "rng_state": str(payload_root),
    }
    assert emitted["payload_relocation"]["target_roots"] == emitted["checkpoint_payload_roots"]
    assert not (output_dir / "checkpoints" / "payloads").exists()
    assert (payload_root / "segment_000007" / "params" / "tensor.bin").exists()
    assert (tmp_path / "spill_optimizer" / "payloads" / "segment_000007" / "optimizer_state" / "tensor.bin").exists()
    assert (payload_root / "segment_000007" / "rng_state" / "tensor.bin").exists()


def test_cycle_retries_smaller_profile_when_initial_checkpoint_space_is_insufficient(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "scripts"))
    train_script = importlib.import_module("train_p1_3b")
    train_script = importlib.reload(train_script)

    attempts: list[str] = []
    output_dir = tmp_path / "retry_run"

    def fake_run_p1_training_cycle(config: object) -> dict[str, object]:
        profile = str(config.model_config["profile"])
        attempts.append(profile)
        if len(attempts) == 1:
            raise RuntimeError(
                "Insufficient free space on the checkpoint payload volume to commit the initial checkpoint payload: "
                "checkpoint_payload_root=/tmp/payloads, free_bytes=1, required_free_bytes=2."
            )
        return {
            "status": "Done",
            "checkpoint_manifest_path": str(output_dir / "checkpoints" / "retry" / "segment_000000.json"),
            "streaming_manifest_sha256": "manifest-sha",
            "tokenizer_manifest_ref": "tokenizer/manifest.json",
            "requested_streaming_mode": "auto",
            "effective_streaming_mode": "auto",
            "local_snapshot_manifest_ref": "",
            "last_segment_id": 0,
        }

    monkeypatch.setattr(train_script, "_kaggle_runtime_present", lambda: True)
    monkeypatch.setattr(train_script, "_maybe_bootstrap_from_hf", lambda args: None)
    monkeypatch.setattr(train_script, "_cleanup_after_failed_resource_pressure", lambda error: None)
    monkeypatch.setattr(
        train_script,
        "_load_model_config",
        lambda path, memory_profile, kaggle_runtime: {"profile": memory_profile},
    )
    monkeypatch.setattr(
        train_script,
        "cycle_memory_profile_candidates",
        lambda memory_profile, kaggle_runtime: ("kaggle_safe", "kaggle_emergency"),
    )
    monkeypatch.setattr(iris3b_training_module, "run_p1_training_cycle", fake_run_p1_training_cycle)
    monkeypatch.setattr(train_script, "write_latest_pointer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_p1_3b.py",
            "cycle",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "retry",
            "--max-cycle-minutes",
            "1",
            "--no-download-latest",
            "--no-sync-checkpoint",
        ],
    )

    assert train_script.main() == 0

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert attempts == ["kaggle_safe", "kaggle_emergency"]
    assert summary["attempted_memory_profiles"] == ["kaggle_safe", "kaggle_emergency"]
    assert summary["effective_memory_profile"] == "kaggle_emergency"
    assert summary["memory_profile_fallback_used"] is True
