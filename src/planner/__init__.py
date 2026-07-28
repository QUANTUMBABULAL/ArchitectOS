"""
Planner package: converts user goals into executable research plans.

Plan shape follows the DecisionEngine's complexity assessment. The planner
never executes plans; the orchestrator does.
"""

from .planner import PlanStep, PlanStepKind, Planner, ResearchPlan

__all__ = [
    "PlanStep",
    "PlanStepKind",
    "Planner",
    "ResearchPlan",
]
