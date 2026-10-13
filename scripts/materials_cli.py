#!/usr/bin/env python3
"""Offline Materials Consultant CLI (domain system prompt + OwnLLM).

Usage:
  python scripts/materials_cli.py --checkpoint models/ownllm.pt \\
      --tokenizer models/tokenizer.json --query "Titanium Ti-6Al-4V"

Without a trained checkpoint, prints the formatted prompt only (dry-run).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure domains/ is discoverable when run from repo root
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OWNLLM_DOMAINS_DIR", str(ROOT / "domains"))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="CampusMind Materials domain offline CLI")
    ap.add_argument("--query", "-q", required=True, help="Material / engineering question")
    ap.add_argument("--checkpoint", default=os.getenv("OWNLLM_MODEL_PATH", ""))
    ap.add_argument("--tokenizer", default=os.getenv("TOKENIZER_PATH", ""))
    ap.add_argument("--domain", default="materials")
    ap.add_argument("--context", default="", help="Optional local RAG context text")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=-1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true", help="Only print the formatted prompt")
    ap.add_argument("--offline", action="store_true", help="Enforce in-process offline guard")
    args = ap.parse_args()

    if args.offline:
        from ownllm.offline import enforce_offline

        enforce_offline()
        print("[offline guard active]", file=sys.stderr)

    from ownllm.prompts import load_domain

    profile = load_domain(args.domain)
    prompt = profile.format(args.query, context=args.context or None)

    if args.dry_run or not args.checkpoint or not args.tokenizer:
        print("--- FORMATTED PROMPT (domain=%s) ---" % profile.id)
        print(prompt)
        if not args.checkpoint:
            print(
                "\n[dry-run] No checkpoint provided. Train OwnLLM then pass "
                "--checkpoint and --tokenizer for generation.",
                file=sys.stderr,
            )
        return

    import torch
    from tokenizers import Tokenizer

    from ownllm import ModelConfig, OwnLLM
    from ownllm.runtime import ModelRuntime

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = state.get("config") or {}
    if isinstance(cfg_dict, dict) and "model" in cfg_dict:
        cfg_dict = cfg_dict["model"]
    cfg = ModelConfig.from_dict(cfg_dict) if cfg_dict else ModelConfig()
    model = OwnLLM(cfg)
    model.load_state_dict(state.get("model", state))
    tok = Tokenizer.from_file(args.tokenizer)
    rt = ModelRuntime(
        model,
        tok,
        device=args.device,
        default_domain=args.domain,
        max_concurrency=1,
    )

    kwargs = {
        "domain": args.domain,
        "context": args.context or None,
    }
    if args.max_new_tokens > 0:
        kwargs["max_new_tokens"] = args.max_new_tokens
    if args.temperature >= 0:
        kwargs["temperature"] = args.temperature

    text = asyncio.run(rt.generate_text(args.query, **kwargs))
    print(text)


if __name__ == "__main__":
    main()
