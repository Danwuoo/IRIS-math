from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .state_ir import STATE_IR_SLOT_ORDER, StateIR

LEGACY_STATE_IR_TOKEN_ORDER: Tuple[str, ...] = ("T", "G", "O", "R", "X", "M")
V2_STATE_IR_SLOT_ORDER: Tuple[str, ...] = STATE_IR_SLOT_ORDER

_LEGACY_SECTION_NOTES = {
    "T": "legacy baseline token bucket retained only for transitional compatibility",
    "G": "legacy baseline goal/control bucket retained only for transitional compatibility",
    "O": "legacy object bucket does not faithfully recover v2 slot semantics on its own",
    "R": "legacy relation bucket does not faithfully recover v2 slot semantics on its own",
    "X": "legacy event bucket does not faithfully recover v2 verifier/control separation",
    "M": "legacy macro bucket does not faithfully recover v2 memory/abstraction semantics",
}


@dataclass(frozen=True)
class LegacyStateIRAdapterReport:
    schema: str
    implementation_status: str
    lossy_projection: bool
    native_slot_lengths: Dict[str, int]
    legacy_section_lengths: Dict[str, int]
    v2_slot_status: Dict[str, str]
    notes: Tuple[str, ...]


def build_legacy_state_ir_adapter_report(state: StateIR) -> LegacyStateIRAdapterReport:
    lengths = state.section_lengths()
    notes = tuple(
        [
            "TEMPORARY TECHNICAL DEBT: legacy_state_ir_adapter exists only to make transitional usage explicit.",
            "Removal criterion: remove once no active consumer depends on legacy T/G/O/R/X/M sections.",
            "Intended learned replacement: native seven-slot State IR objects and schema-faithful level consumers.",
            "Legacy projection recipe is lossy: PF->T, CS->G, SY->O, CG->R, FR+VS->X, LM->M.",
        ]
    )
    return LegacyStateIRAdapterReport(
        schema="iris.legacy_state_ir_adapter/v1",
        implementation_status="transition_only",
        lossy_projection=True,
        native_slot_lengths={slot: int(lengths[slot]) for slot in V2_STATE_IR_SLOT_ORDER},
        legacy_section_lengths={
            "T": int(lengths["PF"]),
            "G": int(lengths["CS"]),
            "O": int(lengths["SY"]),
            "R": int(lengths["CG"]),
            "X": int(lengths["FR"] + lengths["VS"]),
            "M": int(lengths["LM"]),
        },
        v2_slot_status={
            slot: "native_v2_slot_live" for slot in V2_STATE_IR_SLOT_ORDER
        },
        notes=notes + tuple(_LEGACY_SECTION_NOTES[token_type] for token_type in LEGACY_STATE_IR_TOKEN_ORDER),
    )
