#!/usr/bin/env python3
"""Train CampusMind OwnLLM from scratch on prepared token shards.

Example:
    python train.py --shards data/prepared/shards/tokens_00000.bin
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from model import ModelConfig, OwnLLM
from ownllm.data.memmap import MemmapTokenDataset
from ownllm.training.checkpoint import save_checkpoint
from ownllm.training.engine import Trainer
from ownllm.training.sampling import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.json")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--out", default="runs/pretrain")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad", type=float, default=1.0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


def load_config(path: str | Path) -> tuple[ModelConfig, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = ModelConfig.from_dict(raw.get("model", raw))
    if cfg.dim % cfg.n_heads:
        raise ValueError("config: dim must be divisible by n_heads")
    if cfg.n_heads % cfg.n_kv_heads:
        raise ValueError("config: n_heads must be divisible by n_kv_heads")
    return cfg, raw


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch < 1 or args.grad_accum < 1:
        raise ValueError("steps, batch, and grad-accum must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg, raw = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OwnLLM(cfg).to(device)
    if raw.get("training", {}).get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    dataset = MemmapTokenDataset(args.shards, cfg.max_seq_len)
    loader, _ = make_loader(dataset, args.batch)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    trainer = Trainer(
        model,
        optimizer,
        grad_accum=args.grad_accum,
        max_grad=args.max_grad,
        amp=args.amp,
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = iter(loader)

    print(
        {
            "device": str(device),
            "parameters": model.num_parameters(),
            "steps": args.steps,
            "output": str(output_dir),
        }
    )
    for step in range(1, args.steps + 1):
        microbatches = []
        for _ in range(args.grad_accum):
            try:
                input_ids, targets = next(batches)
            except StopIteration:
                batches = iter(loader)
                input_ids, targets = next(batches)
            microbatches.append((input_ids.to(device), targets.to(device)))

        loss = trainer.train_batch(microbatches)
        if step == 1 or step % args.log_every == 0:
            print({"step": step, "loss": round(loss, 6)})
        if step % args.save_every == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                step=step,
                config=cfg.to_dict(),
            )

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        step=args.steps,
        config=cfg.to_dict(),
    )
    print({"done": True, "checkpoint": str(final_path)})


if __name__ == "__main__":
    main()
