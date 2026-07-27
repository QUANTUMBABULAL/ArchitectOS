"""
Brain package: local decision-making backed by Ollama.

The brain decides WHEN and WHICH external AI systems should be consulted.
It is a coordinator, not the primary knowledge source. OllamaClient is
pure transport; DecisionEngine is routing policy.
"""

from .decision_engine import (
    ComplexityAssessment,
    ConsultationDecision,
    DecisionEngine,
    TaskComplexity,
)
from .ollama_client import ChatMessage, OllamaClient, OllamaClientConfig

__all__ = [
    "ChatMessage",
    "ComplexityAssessment",
    "ConsultationDecision",
    "DecisionEngine",
    "OllamaClient",
    "OllamaClientConfig",
    "TaskComplexity",
]
