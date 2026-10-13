"""Dataset / checkpoint lineage manifests for auditability."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_manifest(
    path: str | Path,
    model_version: str,
    config: dict,
    datasets: Iterable[str | Path],
    tokenizer: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict:
    manifest: dict[str, Any] = {
        "created": time.time(),
        "model_version": model_version,
        "config": config,
        "datasets": [{"path": str(x), "sha256": sha256_file(x)} for x in datasets],
        "tokenizer": None
        if not tokenizer
        else {"path": str(tokenizer), "sha256": sha256_file(tokenizer)},
        "checkpoint": None
        if not checkpoint
        else {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    }
    if extra:
        manifest["extra"] = extra
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest
