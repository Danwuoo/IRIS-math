from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JOURNAL_SCHEMA = "iris.segment_journal/v1"
PENDING = "PENDING"
APPLIED = "APPLIED"


def load_journal(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def append_journal_event(path: Path, event: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = load_journal(path)
    event_record = dict(event)
    event_record["schema"] = JOURNAL_SCHEMA
    event_record["event_id"] = len(events) + 1
    event_record["event_time"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_record, sort_keys=True) + "\n")
    return event_record


def journal_head_hash(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    payload = json.dumps(events[-1], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def last_applied_event(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for event in reversed(events):
        if event.get("status") == APPLIED:
            return event
    return None


def resolve_resume_pointer(
    events: List[Dict[str, Any]],
) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not events:
        return 0, None, None

    latest_event = events[-1]
    applied_event = last_applied_event(events)
    status = latest_event.get("status")

    if status == PENDING:
        return int(latest_event["segment_id"]), applied_event, latest_event
    if status == APPLIED:
        return int(latest_event["segment_id"]) + 1, latest_event, None
    raise ValueError(f"Unknown journal status '{status}'.")
