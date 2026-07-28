"""
Research planner.

The Planner converts a user goal into an executable research plan. Plan
shape is driven by the DecisionEngine's complexity assessment: simple
goals become one local-answer step; complex goals are decomposed into
focused consultation sub-questions followed by a synthesis step. The
planner never executes anything itself — the orchestrator runs plans.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from src.brain import ComplexityAssessment, DecisionEngine, TaskComplexity
from src.config import Settings, get_settings
from src.constants import PLANNER_MAX_BRANCHES
from src.exceptions import PlannerError
from src.logger import get_logger
from src.prompts import (
    DECOMPOSER_SCHEMA,
    DECOMPOSER_SYSTEM_PROMPT,
    decomposer_prompt,
)


class PlanStepKind(str, Enum):
    """
    Kinds of executable plan steps.

    LOCAL steps are answered by the local model. CONSULT steps are sent
    to external AI workers. SYNTHESIZE steps combine earlier results into
    a final report.
    """

    LOCAL = "local"
    CONSULT = "consult"
    SYNTHESIZE = "synthesize"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """
    One executable step in a research plan.

    Attributes:
        step_id: Stable identifier of the step.
        index: Zero-based execution order.
        kind: Step kind.
        description: Human-readable purpose of the step.
        prompt: Prompt text executed for this step.
        depends_on: Step IDs whose results feed into this step.
    """

    step_id: str
    index: int
    kind: PlanStepKind
    description: str
    prompt: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    """
    Executable research plan for one goal.

    Attributes:
        plan_id: Stable plan identifier.
        goal: Original user goal.
        steps: Ordered executable steps.
        assessment: Complexity assessment that shaped the plan.
        used_fallback: True when decomposition fell back to the
            deterministic single-consultation shape.
        created_at: UTC timestamp when the plan was created.
    """

    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...]
    assessment: Optional[ComplexityAssessment] = None
    used_fallback: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def consult_steps(self) -> tuple[PlanStep, ...]:
        """
        Return the consultation steps of this plan.

        Returns:
            Steps with kind CONSULT in execution order.
        """
        return tuple(
            step for step in self.steps if step.kind == PlanStepKind.CONSULT
        )


class Planner:
    """
    Converts user goals into executable research plans.

    The planner asks the DecisionEngine to classify the goal, then either
    produces a single local step or a model-decomposed consultation plan
    with a final synthesis step. Decomposition failures degrade to a
    deterministic one-consultation plan instead of raising.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        max_sub_questions: int = PLANNER_MAX_BRANCHES,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
            decision_engine: Decision engine used for classification and
                decomposition.
            max_sub_questions: Maximum consultation sub-questions per plan.
            settings: Optional application settings supplying the
                generation cap for decomposition.

        Raises:
            PlannerError: If max_sub_questions is not positive.
        """
        if max_sub_questions < 1:
            raise PlannerError(
                "max_sub_questions must be at least 1",
                code="PLANNER_MAX_BRANCHES_INVALID",
            )
        self._engine = decision_engine
        self._max_sub_questions = max_sub_questions
        self._settings = settings or get_settings()
        self._logger = get_logger(__name__)

    async def create_plan(self, goal: str) -> ResearchPlan:
        """
        Create an executable research plan for a goal.

        Args:
            goal: User research goal.

        Returns:
            Research plan with at least one step.

        Raises:
            PlannerError: If the goal is empty.
        """
        cleaned = self._require_goal(goal)
        assessment = await self._engine.classify_complexity(cleaned)

        if assessment.complexity == TaskComplexity.SIMPLE:
            return self._build_local_plan(cleaned, assessment)

        sub_questions, used_fallback = await self._decompose(cleaned)
        return self._build_consultation_plan(
            goal=cleaned,
            sub_questions=sub_questions,
            assessment=assessment,
            used_fallback=used_fallback,
        )

    async def _decompose(self, goal: str) -> tuple[list[str], bool]:
        """
        Decompose a complex goal into sub-questions.

        Args:
            goal: Cleaned research goal.

        Returns:
            Tuple of (sub-questions, used_fallback).
        """
        try:
            raw = await self._engine.client.generate(
                prompt=decomposer_prompt(goal, self._max_sub_questions),
                system=DECOMPOSER_SYSTEM_PROMPT,
                options={"temperature": 0.2},
                num_predict=self._settings.ollama_decompose_tokens,
                response_format=DECOMPOSER_SCHEMA,
                timeout_seconds=self._settings.ollama_timeout,
                operation="decompose",
            )
            data = self._parse_json_object(raw)
            candidates = data.get("sub_questions", [])
            if not isinstance(candidates, list):
                raise ValueError("sub_questions is not a list")

            sub_questions: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                question = str(candidate).strip()
                key = question.lower()
                if question and key not in seen:
                    seen.add(key)
                    sub_questions.append(question)
                if len(sub_questions) >= self._max_sub_questions:
                    break

            if not sub_questions:
                raise ValueError("Decomposition produced no sub-questions")

            return sub_questions, False
        except Exception as exc:
            self._logger.warning(
                "Goal decomposition failed (%s); using single-consultation "
                "fallback plan",
                exc,
            )
            return [goal], True

    def _build_local_plan(
        self,
        goal: str,
        assessment: ComplexityAssessment,
    ) -> ResearchPlan:
        """
        Build a one-step plan answered by the local model.

        Args:
            goal: Cleaned research goal.
            assessment: Complexity assessment.

        Returns:
            Research plan with a single LOCAL step.
        """
        step = PlanStep(
            step_id=uuid4().hex,
            index=0,
            kind=PlanStepKind.LOCAL,
            description="Answer the goal directly with the local model",
            prompt=goal,
        )
        plan = ResearchPlan(
            plan_id=uuid4().hex,
            goal=goal,
            steps=(step,),
            assessment=assessment,
        )
        self._logger.info(
            "Created local plan %s with 1 step", plan.plan_id
        )
        return plan

    def _build_consultation_plan(
        self,
        goal: str,
        sub_questions: list[str],
        assessment: ComplexityAssessment,
        used_fallback: bool,
    ) -> ResearchPlan:
        """
        Build a consultation plan with a final synthesis step.

        Args:
            goal: Cleaned research goal.
            sub_questions: Consultation sub-questions.
            assessment: Complexity assessment.
            used_fallback: Whether decomposition used the fallback.

        Returns:
            Research plan with CONSULT steps and one SYNTHESIZE step.
        """
        steps: list[PlanStep] = []
        for index, question in enumerate(sub_questions):
            steps.append(
                PlanStep(
                    step_id=uuid4().hex,
                    index=index,
                    kind=PlanStepKind.CONSULT,
                    description=f"Consult external AI systems: {question}",
                    prompt=question,
                )
            )

        consult_ids = tuple(step.step_id for step in steps)
        steps.append(
            PlanStep(
                step_id=uuid4().hex,
                index=len(steps),
                kind=PlanStepKind.SYNTHESIZE,
                description=(
                    "Synthesize consultation results into a final report"
                ),
                prompt=goal,
                depends_on=consult_ids,
            )
        )

        plan = ResearchPlan(
            plan_id=uuid4().hex,
            goal=goal,
            steps=tuple(steps),
            assessment=assessment,
            used_fallback=used_fallback,
        )
        self._logger.info(
            "Created consultation plan %s with %d step(s)",
            plan.plan_id,
            len(plan.steps),
        )
        return plan

    @staticmethod
    def _require_goal(goal: str) -> str:
        """
        Validate and clean a goal string.

        Args:
            goal: Raw goal text.

        Returns:
            Stripped goal text.

        Raises:
            PlannerError: If the goal is empty.
        """
        if not goal or not goal.strip():
            raise PlannerError(
                "Research goal cannot be empty",
                code="PLANNER_GOAL_EMPTY",
            )
        return goal.strip()

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, object]:
        """
        Extract and parse the first JSON object in model output.

        Args:
            raw: Raw model output.

        Returns:
            Parsed JSON object.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise ValueError("No JSON object found in model output")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not an object")
        return data


__all__ = [
    "PlanStep",
    "PlanStepKind",
    "Planner",
    "ResearchPlan",
]
