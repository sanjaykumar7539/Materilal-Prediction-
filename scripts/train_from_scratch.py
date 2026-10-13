#!/usr/bin/env python3
"""Generic from-scratch pretraining entry point. Corpus is user-supplied."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ownllm import ModelConfig, OwnLLM
from ownllm.data.memmap import MemmapTokenDataset
from ownllm.training.checkpoint import save_checkpoint
from ownllm.training.engine import Trainer
from ownllm.training.sampling import make_loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model.json")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    cfg_dict = raw.get("model", raw)
    cfg = ModelConfig.from_dict(cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OwnLLM(cfg).to(device)
    if raw.get("training", {}).get("gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)

    ds = MemmapTokenDataset(args.shards, cfg.max_seq_len)
    loader, _ = make_loader(ds, args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    trainer = Trainer(model, opt, grad_accum=args.grad_accum, amp=args.amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    it = iter(loader)

    for step in range(1, args.steps + 1):
        microbatches = []
        for _ in range(max(1, args.grad_accum)):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            microbatches.append((x.to(device), y.to(device)))
        loss = trainer.train_batch(microbatches)
        if step % 100 == 0:
            print({"step": step, "loss": loss})
        if step % 1000 == 0:
            save_checkpoint(out / "last.pt", model, opt, step=step, config=cfg.to_dict())

    save_checkpoint(out / "final.pt", model, opt, step=args.steps, config=cfg.to_dict())
    print({"done": True, "steps": args.steps, "out": str(out / "final.pt")})


if __name__ == "__main__":
    main()
