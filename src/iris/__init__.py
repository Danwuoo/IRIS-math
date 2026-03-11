from .schema import STATE_IR_SLOT_ORDER, STATE_IR_TOKEN_ORDER, StateIR, StateIRValidationError

__all__ = [
    "STATE_IR_SLOT_ORDER",
    "STATE_IR_TOKEN_ORDER",
    "StateIR",
    "StateIRValidationError",
    "SingleTrunk",
    "TrunkOutput",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"SingleTrunk", "TrunkOutput"}:
        from .trunk import SingleTrunk, TrunkOutput

        return {"SingleTrunk": SingleTrunk, "TrunkOutput": TrunkOutput}[name]
    raise AttributeError(f"module 'iris' has no attribute {name!r}")
