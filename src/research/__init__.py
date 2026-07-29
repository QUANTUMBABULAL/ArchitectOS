"""
Research package: the operator that investigates rather than fans out.

A request enters as one question and leaves as one executive report. In
between, the planner decomposes it into subtasks, the operator
distributes those subtasks across providers by capability, the extractor
turns each answer into structured evidence, and the report builder
condenses the verified evidence into a decision.
"""

from .evidence import EvidenceExtractor, EvidenceItem, EvidenceSet
from .executive import Alternative, ExecutiveReport, ExecutiveReportBuilder
from .operator import ResearchOperator, ResearchResult, ResearchStage
from .plan import ResearchPlan, ResearchTask, TaskKind
from .planner import ResearchPlanner

__all__ = [
    "Alternative",
    "EvidenceExtractor",
    "EvidenceItem",
    "EvidenceSet",
    "ExecutiveReport",
    "ExecutiveReportBuilder",
    "ResearchOperator",
    "ResearchPlan",
    "ResearchPlanner",
    "ResearchResult",
    "ResearchStage",
    "ResearchTask",
    "TaskKind",
]
