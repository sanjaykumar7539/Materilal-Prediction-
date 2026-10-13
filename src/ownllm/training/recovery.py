"""Run-state recovery records for resumable training jobs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunState:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, **state: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        payload = {"updated": time.time(), **state}
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())
