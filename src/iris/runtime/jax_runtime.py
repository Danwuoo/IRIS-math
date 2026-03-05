from __future__ import annotations

from typing import Dict, List


def assert_jax_runtime(device: str = "cpu", require_gpu: bool = False) -> Dict[str, List[str]]:
    try:
        import flax  # noqa: F401
        import jax
        import optax  # noqa: F401
    except Exception as error:  # pragma: no cover - import failure depends on runtime
        raise RuntimeError(
            "Strict JAX runtime required. Install jax/flax/optax before running this command."
        ) from error

    normalized_device = str(device).lower()
    if normalized_device not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported device '{device}'. Expected one of: cpu, gpu.")

    devices = list(jax.devices())
    if not devices:
        raise RuntimeError("JAX runtime is available but no devices were detected.")

    all_devices = [f"{dev.platform}:{dev.id}" for dev in devices]
    gpu_devices = [f"{dev.platform}:{dev.id}" for dev in devices if dev.platform == "gpu"]
    if (normalized_device == "gpu" or require_gpu) and not gpu_devices:
        raise RuntimeError(
            "Strict JAX runtime requested GPU, but no JAX GPU device is available."
        )

    return {
        "all_devices": all_devices,
        "gpu_devices": gpu_devices,
    }
