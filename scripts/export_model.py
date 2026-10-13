#!/usr/bin/env python3
"""Export OwnLLM checkpoint to native PyTorch or ONNX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ownllm import ModelConfig, OwnLLM


class LogitsOnly(nn.Module):
    def __init__(self, model: OwnLLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(input_ids)
        return logits


def load(config: str, checkpoint: str) -> tuple[OwnLLM, ModelConfig]:
    raw = json.loads(Path(config).read_text())
    cfg = ModelConfig.from_dict(raw.get("model", raw))
    model = OwnLLM(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format", choices=["native", "onnx"], default="native")
    p.add_argument("--output", required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    model, cfg = load(args.config, args.checkpoint)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "native":
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out)
        print(out)
        return

    dummy = torch.ones(1, min(16, cfg.max_seq_len), dtype=torch.long)
    torch.onnx.export(
        LogitsOnly(model),
        dummy,
        str(out),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(out)


if __name__ == "__main__":
    main()
