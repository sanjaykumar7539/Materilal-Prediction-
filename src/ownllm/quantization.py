"""Post-training quantization helpers (CPU INT8 / half precision)."""
from __future__ import annotations

import torch
from torch import nn


def dynamic_int8(model: nn.Module) -> nn.Module:
    """CPU-oriented dynamic INT8 for Linear layers. Validate quality before promotion."""
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def fp16(model: nn.Module) -> nn.Module:
    return model.half()


def bf16(model: nn.Module) -> nn.Module:
    return model.to(dtype=torch.bfloat16)
