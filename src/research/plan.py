"""
Research plan model.

A research request is not one question — it is a set of questions whose
answers combine into a recommendation. This module holds the plan the
operator executes: the subtasks, what kind of investigation each one is,
and which provider was assigned to it.

Nothing here performs inference or touches a browser. The plan is data,
so it can be produced by a model, by a deterministic template, or by a
test, and executed identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskKind(str, Enum):
    """
    The kind of investigation a subtask performs.

    The kind drives two things: the phrasing of the prompt sent to a
    provider, and which provider is best suited to receive it. A provider
    tagged for current events is a better fit for LANDSCAPE than for
    SYNTHESIS.

    Attributes:
        LANDSCAPE: What exists — options, launches, candidates.
        MEASUREMENT: Hard numbers — benchmarks, endurance, throughput.
        QUALITATIVE: Subjective quality — reviews, comparisons, feel.
        COMMUNITY: Owner and community sentiment, long-term reports.
        PRICING: Cost, availability, and value.
        RISK: Weaknesses, failure modes, and caveats.
        SYNTHESIS: Weigh everything and recommend.
    """

    LANDSCAPE = "landscape"
    MEASUREMENT = "measurement"
    QUALITATIVE = "qualitative"
    COMMUNITY = "community"
    PRICING = "pricing"
    RISK = "risk"
    SYNTHESIS = "synthesis"

    @property
    def preferred_capabilities(self) -> tuple[str, ...]:
        """
        Return provider capability tags suited to this kind of task.

        Returns:
            Capability tags, best first.
        """
        return _PREFERRED_CAPABILITIES.get(self, ("general",))


_PREFERRED_CAPABILITIES: dict[TaskKind, tuple[str, ...]] = {
    TaskKind.LANDSCAPE: ("current_events", "search", "general"),
    TaskKind.MEASUREMENT: ("reasoning", "math", "code", "general"),
    TaskKind.QUALITATIVE: ("writing", "general", "reasoning"),
    TaskKind.COMMUNITY: ("search", "current_events", "general"),
    TaskKind.PRICING: ("search", "current_events", "math", "general"),
    TaskKind.RISK: ("reasoning", "general"),
    TaskKind.SYNTHESIS: ("reasoning", "general"),
}


@dataclass(frozen=True, slots=True)
class ResearchTask:
    """
    One subtask of a research plan.

    Attributes:
        task_id: One-based position in the plan.
        title: Short imperative label, shown on worker cards.
        question: The full question sent to the assigned provider.
        kind: What kind of investigation this is.
        assigned_to: Provider assigned to answer it, once distribution has
            run.
    """

    task_id: int
    title: str
    question: str
    kind: TaskKind = TaskKind.LANDSCAPE
    assigned_to: Optional[str] = None

    def assign(self, provider: str) -> "ResearchTask":
        """
        Return a copy of this task assigned to a provider.

        Args:
            provider: Provider name.

        Returns:
            Assigned task.
        """
        return ResearchTask(
            task_id=self.task_id,
            title=self.title,
            question=self.question,
            kind=self.kind,
            assigned_to=provider,
        )

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream.

        Returns:
            JSON-compatible payload.
        """
        return {
            "taskId": self.task_id,
            "title": self.title,
            "question": self.question,
            "kind": self.kind.value,
            "assignedTo": self.assigned_to,
        }


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    """
    The full decomposition of one research request.

    Attributes:
        question: The user's original request, verbatim.
        objective: One sentence describing what a good answer establishes.
        tasks: Subtasks in execution order.
        generated_by: ``"model"`` when the local model produced the plan,
            ``"template"`` when the deterministic fallback did.
    """

    question: str
    objective: str
    tasks: list[ResearchTask] = field(default_factory=list)
    generated_by: str = "template"

    @property
    def investigation_tasks(self) -> list[ResearchTask]:
        """
        Return the tasks that gather evidence.

        Synthesis is excluded: it is performed by the operator over the
        gathered evidence, not by asking a provider to guess.

        Returns:
            Evidence-gathering tasks.
        """
        return [
            task for task in self.tasks if task.kind is not TaskKind.SYNTHESIS
        ]

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream.

        Returns:
            JSON-compatible payload.
        """
        return {
            "question": self.question,
            "objective": self.objective,
            "generatedBy": self.generated_by,
            "tasks": [task.to_payload() for task in self.tasks],
        }


__all__ = [
    "ResearchPlan",
    "ResearchTask",
    "TaskKind",
]
