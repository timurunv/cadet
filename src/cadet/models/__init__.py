"""Model modules for transformer-based hate speech detection."""

from .base_model import MODEL_REGISTRY, BaseModel
from .cadet import CADET, GradientReversalLayer
from .llm_guard import LlamaGuard, LLMGuardModel, PromptGuard, ShieldGemma
from .simple_baselines import SimpleBaselineModel

__all__ = [
    # Base model
    "BaseModel",
    "MODEL_REGISTRY",
    # Baseline models
    "SimpleBaselineModel",
    # CADET models
    "CADET",
    "GradientReversalLayer",
    # LLM Guard models
    "LLMGuardModel",
    "LlamaGuard",
    "PromptGuard",
    "ShieldGemma",
]
