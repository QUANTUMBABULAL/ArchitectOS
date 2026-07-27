"""
Orchestrator package: end-to-end research coordination.

ResearchOrchestrator binds Planner, DecisionEngine, WorkerManager,
ConsensusEngine, and MemoryStore into one research control flow.
"""

from .research_orchestrator import (
    ResearchOrchestrator,
    ResearchOutcome,
    StepResult,
)

__all__ = [
    "ResearchOrchestrator",
    "ResearchOutcome",
    "StepResult",
]
