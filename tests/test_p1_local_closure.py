from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_p1_local_closure_script_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "closure"
    result = subprocess.run(
        [sys.executable, "scripts/p1_local_closure.py", "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "closure_summary.json").read_text(encoding="utf-8"))
    assert summary["termination"] == "Done"
