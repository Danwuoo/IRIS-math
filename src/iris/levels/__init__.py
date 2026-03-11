from importlib import import_module

from .base import LevelInput, LevelInterface, LevelOutput
from .stubs import L0Level, L1Level, L2Level, L3Level, L4Level, L5Level, L6Level, build_level_stack

LEVEL_IDS = tuple(f"L{index}" for index in range(7))

_MOUNTED_EXPORTS = {
    "L0MountedLevel",
    "L1MountedLevel",
    "L2MountedLevel",
    "L3MountedLevel",
    "L4MountedLevel",
    "L5MountedLevel",
    "L6MountedLevel",
    "apply_level_stack_params",
    "init_level_stack_params",
    "init_level_params",
    "l6_credit_from_params",
    "level_forward",
}

__all__ = [
    "LEVEL_IDS",
    "L0Level",
    "L1Level",
    "L2Level",
    "L3Level",
    "L4Level",
    "L5Level",
    "L6Level",
    "L0MountedLevel",
    "L1MountedLevel",
    "L2MountedLevel",
    "L3MountedLevel",
    "L4MountedLevel",
    "L5MountedLevel",
    "L6MountedLevel",
    "LevelInput",
    "LevelInterface",
    "LevelOutput",
    "apply_level_stack_params",
    "build_level_stack",
    "init_level_stack_params",
    "init_level_params",
    "l6_credit_from_params",
    "level_forward",
]


def __getattr__(name: str):
    if name not in _MOUNTED_EXPORTS:
        raise AttributeError(f"module 'iris.levels' has no attribute {name!r}")

    try:
        mounted = import_module("iris.levels.mounted")
    except Exception as error:  # pragma: no cover - optional dependency failure
        raise RuntimeError(
            "Mounted level exports require the optional JAX runtime."
        ) from error
    value = getattr(mounted, name)
    globals()[name] = value
    return value
