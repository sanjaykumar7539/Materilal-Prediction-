"""CampusMind OwnLLM — offline-first decoder-only LLM foundation with domain profiles."""
from .model import ModelConfig, OwnLLM

__version__ = "1.1.2"
__all__ = ["OwnLLM", "ModelConfig", "__version__"]
