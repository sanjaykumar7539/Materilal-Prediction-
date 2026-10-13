#!/usr/bin/env python3
"""Prepare training data from local TXT, PDF, EPUB, DOCX, and ZIP documents."""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from ownllm.data.documents import iter_documents
from ownllm.data.sharding import shard_encoded_documents
from ownllm.tokenizer_train import train_bpe


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare tokenizer + token shards")
    ap.add_argument("input", help="Document file or directory containing TXT, PDF, EPUB, DOCX, or ZIP files")
    ap.add_argument("output", help="Output directory")
    ap.add_argument("--vocab", type=int, default=32000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Temporary plain-text files keep tokenizer and sharding inputs identical.
    # Archive member names never become filesystem paths.
    with tempfile.TemporaryDirectory(prefix="ownllm-documents-") as temporary:
        files = []
        try:
            for index, (source, text) in enumerate(iter_documents(args.input, exclude=args.output)):
                extracted = Path(temporary) / f"document_{index:06d}.txt"
                extracted.write_text(text, encoding="utf-8")
                files.append(extracted)
                logging.info("Read %s (%s characters)", source, len(text))
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        if not files:
            raise SystemExit("No readable TXT, PDF, EPUB, DOCX, or ZIP documents found. Scanned PDFs need OCR.")
        tok = train_bpe(files, Path(args.output) / "tokenizer", args.vocab)
        eos = tok.token_to_id("<eos>")
        docs = (tok.encode(p.read_text(encoding="utf-8")).ids + [eos] for p in files)
        metas = shard_encoded_documents(docs, Path(args.output) / "shards")
        print({"files": len(files), "shards": len(metas), "metas": metas})


if __name__ == "__main__":
    main()
