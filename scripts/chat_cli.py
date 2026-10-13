#!/usr/bin/env python3
"""Interactive multi-turn chat CLI for OwnLLM (offline, local checkpoint).

Maintains conversation history and renders it with the configured chat
template before every turn, exactly like the API path.

Usage:
  python scripts/chat_cli.py --checkpoint models/ownllm.pt \
      --tokenizer models/tokenizer.json [--domain materials]

Type 'exit' / 'quit' / Ctrl-D to leave. Type 'reset' to clear history.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OWNLLM_DOMAINS_DIR", str(ROOT / "domains"))
sys.path.insert(0, str(ROOT / "src"))

from ownllm.prompts import apply_chat_template  # noqa: E402

EXIT_WORDS = {"exit", "quit", "q"}
RESET_WORD = "reset"


def render_history(
    history: list[tuple[str, str]],
    user_message: str,
    system_prompt: str | None,
    template: str,
) -> str:
    """Render prior turns + new user message into one prompt string."""
    turns: list[str] = []
    for user_text, assistant_text in history:
        turns.append(
            apply_chat_template(user_text, system_prompt=None, template=template)
            + assistant_text.strip()
        )
    prefix = "".join(t for t in turns)
    current = apply_chat_template(user_message, system_prompt=system_prompt, template=template)
    return prefix + current


async def chat_loop(args: argparse.Namespace) -> None:
    import torch
    from tokenizers import Tokenizer

    from ownllm import ModelConfig, OwnLLM
    from ownllm.prompts import load_domain
    from ownllm.runtime import ModelRuntime

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = state.get("config") or {}
    if isinstance(cfg_dict, dict) and "model" in cfg_dict:
        cfg_dict = cfg_dict["model"]
    cfg = ModelConfig.from_dict(cfg_dict) if cfg_dict else ModelConfig()
    model = OwnLLM(cfg)
    model.load_state_dict(state.get("model", state))
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    rt = ModelRuntime(
        model,
        tok,
        device=args.device,
        default_domain=args.domain or None,
        max_concurrency=1,
    )

    system_prompt: str | None = None
    template = args.template
    if args.domain:
        profile = load_domain(args.domain)
        system_prompt = profile.system_prompt
        template = profile.chat_template
        print(f"[domain={profile.id} | template={template}]", file=sys.stderr)

    history: list[tuple[str, str]] = []
    print("OwnLLM chat ready. 'reset' clears history, 'exit' quits.")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in EXIT_WORDS:
            break
        if user_text.lower() == RESET_WORD:
            history.clear()
            print("[history cleared]")
            continue

        prompt = render_history(history, user_text, system_prompt, template)
        reply = await rt.generate_text(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            apply_template=False,  # already fully rendered above
            repetition_penalty=args.repetition_penalty,
        )
        reply = reply.strip()
        print(f"Bot> {reply}")
        history.append((user_text, reply))


def main() -> None:
    ap = argparse.ArgumentParser(description="OwnLLM interactive offline chat")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--domain", default=os.getenv("OWNLLM_DOMAIN", ""))
    ap.add_argument("--template", default="chatml", choices=["chatml", "plain", "instruct"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()
    asyncio.run(chat_loop(args))


if __name__ == "__main__":
    main()
