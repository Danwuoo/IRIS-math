from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from iris.runtime import assert_jax_runtime
from iris.train import evaluate_latest_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate latest toy checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/toy_train"))
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--strict-jax", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    assert_jax_runtime(
        device=args.device,
        require_gpu=bool(args.strict_jax and str(args.device).lower() == "gpu"),
    )

    metrics = evaluate_latest_run(
        output_dir=args.output_dir,
        data_seed=args.data_seed,
        device=args.device,
        strict_jax=args.strict_jax,
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
