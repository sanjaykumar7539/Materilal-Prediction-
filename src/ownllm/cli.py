#!/usr/bin/env python3
"""Offline local inference CLI against a trained checkpoint + tokenizer."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ownllm import ModelConfig, OwnLLM
from ownllm.runtime import ModelRuntime


def load_runtime(checkpoint: str, tokenizer_path: str, device: str) -> ModelRuntime:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = state.get("config") or {}
    if isinstance(cfg_dict, dict) and "model" in cfg_dict:
        cfg_dict = cfg_dict["model"]
    cfg = ModelConfig.from_dict(cfg_dict) if cfg_dict else ModelConfig()
    model = OwnLLM(cfg)
    model.load_state_dict(state.get("model", state))
    model.eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return ModelRuntime(model, tokenizer, device=device)


async def main_async(args: argparse.Namespace) -> None:
    runtime = load_runtime(args.checkpoint, args.tokenizer, args.device)
    text = await runtime.generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="OwnLLM offline inference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
