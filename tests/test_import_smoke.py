from __future__ import annotations

import iris.arc
import iris.levels
import iris.regression
import iris.train


def test_non_jax_import_smoke_paths_are_available() -> None:
    assert iris.levels.LEVEL_IDS == tuple(f"L{index}" for index in range(7))
    assert hasattr(iris.train, "load_default_policy_bundle")
    assert hasattr(iris.train, "load_policy_bundle_for_profile")
    assert hasattr(iris.regression, "GateContext")
    assert hasattr(iris.arc, "load_conceptarc_tasks")
