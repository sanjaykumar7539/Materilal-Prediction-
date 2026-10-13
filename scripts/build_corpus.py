#!/usr/bin/env python3
"""Normalize, quality-filter, and deduplicate local text into JSONL."""
from __future__ import annotations

import argparse
from pathlib import Path

from ownllm.data.cleaning import exact_dedup, normalize, quality_ok
from ownllm.data.jsonl import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Directory of .txt files")
    ap.add_argument("output", help="Output JSONL path")
    ap.add_argument("--min-chars", type=int, default=100)
    args = ap.parse_args()

    records = []
    for path in Path(args.input).rglob("*.txt"):
        text = normalize(path.read_text(encoding="utf-8", errors="ignore"))
        if quality_ok(text, min_chars=args.min_chars):
            records.append({"id": path.stem, "text": text, "source": str(path)})

    cleaned = list(exact_dedup(records))
    write_jsonl(cleaned, args.output)
    print({"input": len(records), "output": len(cleaned), "path": args.output})


if __name__ == "__main__":
    main()
