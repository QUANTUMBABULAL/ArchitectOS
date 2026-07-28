"""
Consensus package: agreement analysis across multiple AI opinions.

The engine consumes plain Opinion values so it stays decoupled from the
worker layer; the orchestrator adapts worker responses into opinions.
"""

from .consensus_engine import (
    ConsensusEngine,
    ConsensusResult,
    Contradiction,
    Opinion,
)

__all__ = [
    "ConsensusEngine",
    "ConsensusResult",
    "Contradiction",
    "Opinion",
]
