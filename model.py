"""Public model entry point for the CampusMind OwnLLM project.

The implementation lives in :mod:`ownllm.model`.  Re-exporting it here keeps
the conventional ``from model import OwnLLM`` interface without maintaining a
second, potentially divergent copy of the architecture.
"""

from ownllm.model import Attention, Block, ModelConfig, OwnLLM, RMSNorm, SwiGLU, apply_rope

__all__ = [
    "Attention",
    "Block",
    "ModelConfig",
    "OwnLLM",
    "RMSNorm",
    "SwiGLU",
    "apply_rope",
]
