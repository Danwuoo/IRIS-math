from __future__ import annotations

import pkgutil
from pathlib import Path

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "iris"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[attr-defined]

from .schema import STATE_IR_SLOT_ORDER, STATE_IR_TOKEN_ORDER, StateIR, StateIRValidationError

__all__ = [
    "STATE_IR_SLOT_ORDER",
    "STATE_IR_TOKEN_ORDER",
    "SingleTrunk",
    "StateIR",
    "StateIRValidationError",
    "TrunkOutput",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"SingleTrunk", "TrunkOutput"}:
        from .trunk import SingleTrunk, TrunkOutput

        return {"SingleTrunk": SingleTrunk, "TrunkOutput": TrunkOutput}[name]
    raise AttributeError(f"module 'iris' has no attribute {name!r}")
