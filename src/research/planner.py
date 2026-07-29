"""
Research planning: turning one request into an investigation.

A research analyst does not ask five colleagues the same question. They
decide what must be established, break that into subtasks, and give each
subtask to whoever is best placed to answer it. This module is the first
half of that: decomposition.

The local Ollama model writes the plan when it can, because subtask
phrasing is genuinely a language problem. When the model is unreachable,
slow, or returns something unusable, a deterministic template produces a
plan from the request's shape instead. Research never fails because the
planner failed — it degrades to a well-formed default.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from src.brain import OllamaClient
from src.config import Settings, get_settings
from src.logger import get_logger

from .plan import ResearchPlan, ResearchTask, TaskKind

_PLANNER_SYSTEM_PROMPT = (
    "You are the planning stage of a research engine. You decompose one "
    "research request into independent subtasks that different "
    "researchers can investigate in parallel.\n"
    "Respond ONLY with a JSON object of the form:\n"
    '{"objective": "<one sentence>", "tasks": [{"title": "<3-6 words>", '
    '"question": "<the full question to investigate>", "kind": '
    '"<landscape|measurement|qualitative|community|pricing|risk>"}]}\n'
    "Rules: produce between 4 and 7 tasks; every task must be answerable "
    "on its own without the others; no task may restate the original "
    "request; cover different angles, never the same angle twice."
)

_MAX_TASKS = 7
_MIN_TASKS = 3
_PLAN_TIMEOUT_SECONDS = 60.0

# Signals used by the deterministic fallback to choose a template. Order
# matters: the first matching family wins.
_COMPARISON_MARKERS = (
    "best", "vs", "versus", "compare", "comparison", "which", "top ",
    "recommend", "should i buy", "alternative", "under ", "cheapest",
)
_HOWTO_MARKERS = ("how to", "how do i", "how can i", "steps to", "guide to")
_EXPLAIN_MARKERS = ("what is", "explain", "why does", "why is", "meaning of")


class ResearchPlanner:
    """
    Decomposes a research request into a plan of subtasks.

    The planner owns no provider knowledge and performs no dispatch. It
    answers exactly one question: *what needs to be found out?*
    """

    def __init__(
        self,
        ollama: Optional[OllamaClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
            ollama: Local model client used to write plans. When omitted,
                only the deterministic template is used.
            settings: Optional application settings.
        """
        self._ollama = ollama
        self._settings = settings or get_settings()
        self._logger = get_logger(__name__)

    async def plan(self, question: str) -> ResearchPlan:
        """
        Build a research plan for one request.

        Args:
            question: The user's research request.

        Returns:
            A plan with at least three subtasks. Never raises: a failed
            model call falls back to the template plan.
        """
        cleaned = question.strip()
        if not cleaned:
            return ResearchPlan(
                question=question,
                objective="Answer the request.",
                tasks=[],
            )

        if self._ollama is not None and self._settings.research_planning:
            plan = await self._model_plan(cleaned)
            if plan is not None:
                self._logger.info(
                    "Planner produced %d subtask(s) with the local model",
                    len(plan.tasks),
                )
                return plan

        plan = self._template_plan(cleaned)
        self._logger.info(
            "Planner produced %d subtask(s) from the %s template",
            len(plan.tasks),
            plan.generated_by,
        )
        return plan

    async def _model_plan(self, question: str) -> Optional[ResearchPlan]:
        """
        Ask the local model to decompose the request.

        Args:
            question: Research request.

        Returns:
            Parsed plan, or None when the model fails or returns garbage.
        """
        try:
            raw = await self._ollama.generate(  # type: ignore[union-attr]
                prompt=f"Research request: {question}",
                system=_PLANNER_SYSTEM_PROMPT,
                response_format="json",
                num_predict=900,
                timeout_seconds=_PLAN_TIMEOUT_SECONDS,
                operation="research_plan",
            )
        except Exception as exc:
            self._logger.warning(
                "Planner model call failed (%s); using the template plan",
                exc,
            )
            return None

        payload = _parse_json_object(raw)
        if not payload:
            return None

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) < _MIN_TASKS:
            return None

        tasks: list[ResearchTask] = []
        for index, item in enumerate(raw_tasks[:_MAX_TASKS], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            body = str(item.get("question") or "").strip()
            if not body:
                continue
            tasks.append(
                ResearchTask(
                    task_id=len(tasks) + 1,
                    title=title or f"Subtask {index}",
                    question=body,
                    kind=_coerce_kind(item.get("kind")),
                )
            )

        if len(tasks) < _MIN_TASKS:
            return None

        objective = str(payload.get("objective") or "").strip()
        return ResearchPlan(
            question=question,
            objective=objective or f"Answer: {question}",
            tasks=tasks,
            generated_by="model",
        )

    def _template_plan(self, question: str) -> ResearchPlan:
        """
        Build a plan without inference, from the request's shape.

        Three families cover the requests a research operator actually
        receives: choosing between options, learning how to do something,
        and understanding a subject. Each has a fixed, sensible
        decomposition.

        Args:
            question: Research request.

        Returns:
            Template-generated plan.
        """
        lowered = question.lower()

        if any(marker in lowered for marker in _HOWTO_MARKERS):
            family, specs = "howto", _HOWTO_TASKS
        elif any(marker in lowered for marker in _COMPARISON_MARKERS):
            family, specs = "comparison", _COMPARISON_TASKS
        elif any(marker in lowered for marker in _EXPLAIN_MARKERS):
            family, specs = "explain", _EXPLAIN_TASKS
        else:
            family, specs = "general", _GENERAL_TASKS

        tasks = [
            ResearchTask(
                task_id=index,
                title=title,
                question=template.format(question=question),
                kind=kind,
            )
            for index, (title, kind, template) in enumerate(specs, start=1)
        ]
        return ResearchPlan(
            question=question,
            objective=f"Establish a defensible answer to: {question}",
            tasks=tasks,
            generated_by=f"template:{family}",
        )


# (title, kind, question template) per request family.
_COMPARISON_TASKS: tuple[tuple[str, TaskKind, str], ...] = (
    (
        "Find current options",
        TaskKind.LANDSCAPE,
        "For this request: {question}\n\nList the current, actually "
        "available options as of today, including anything released "
        "recently. For each, give the exact name and release timing.",
    ),
    (
        "Collect measured data",
        TaskKind.MEASUREMENT,
        "For this request: {question}\n\nReport measured, numeric "
        "performance data for the leading options — benchmarks, capacity, "
        "endurance, throughput, or whatever is measurable here. Give "
        "numbers with their units and say where each number comes from.",
    ),
    (
        "Compare quality",
        TaskKind.QUALITATIVE,
        "For this request: {question}\n\nCompare the leading options on "
        "the qualities reviewers actually judge them on. Be specific "
        "about where each one is better and where it is worse.",
    ),
    (
        "Gather owner sentiment",
        TaskKind.COMMUNITY,
        "For this request: {question}\n\nSummarize what long-term owners "
        "and community discussion (forums, Reddit, review comments) say "
        "about the leading options — especially complaints that only "
        "appear after weeks of use.",
    ),
    (
        "Compare pricing",
        TaskKind.PRICING,
        "For this request: {question}\n\nGive current real-world prices "
        "for the leading options, note where each is available, and say "
        "which represents the best value and why.",
    ),
    (
        "Identify weaknesses",
        TaskKind.RISK,
        "For this request: {question}\n\nName the strongest reasons NOT "
        "to choose each leading option: known defects, weak areas, "
        "support problems, or anything that disappoints buyers.",
    ),
)

_HOWTO_TASKS: tuple[tuple[str, TaskKind, str], ...] = (
    (
        "Establish the approach",
        TaskKind.LANDSCAPE,
        "For this request: {question}\n\nDescribe the approaches that "
        "actually work today, and which is currently considered standard.",
    ),
    (
        "Detail the steps",
        TaskKind.QUALITATIVE,
        "For this request: {question}\n\nGive the concrete steps for the "
        "standard approach, in order, with the exact commands, settings, "
        "or actions required at each step.",
    ),
    (
        "Find prerequisites and costs",
        TaskKind.PRICING,
        "For this request: {question}\n\nList what must be in place "
        "first — tools, accounts, versions, costs, and time required.",
    ),
    (
        "Collect practitioner experience",
        TaskKind.COMMUNITY,
        "For this request: {question}\n\nSummarize what people who have "
        "actually done this report: where they got stuck, and what they "
        "wish they had known first.",
    ),
    (
        "Identify failure modes",
        TaskKind.RISK,
        "For this request: {question}\n\nName the common mistakes and "
        "failure modes, and how to avoid or recover from each.",
    ),
)

_EXPLAIN_TASKS: tuple[tuple[str, TaskKind, str], ...] = (
    (
        "Establish the definition",
        TaskKind.LANDSCAPE,
        "For this request: {question}\n\nGive a precise, current "
        "definition and explain what distinguishes it from adjacent "
        "concepts it is often confused with.",
    ),
    (
        "Collect evidence and data",
        TaskKind.MEASUREMENT,
        "For this request: {question}\n\nProvide the concrete facts, "
        "figures, and documented examples that support the explanation.",
    ),
    (
        "Explain practical significance",
        TaskKind.QUALITATIVE,
        "For this request: {question}\n\nExplain why this matters in "
        "practice and what changes as a result of understanding it.",
    ),
    (
        "Find disputed points",
        TaskKind.RISK,
        "For this request: {question}\n\nIdentify what is genuinely "
        "disputed or commonly misunderstood here, and what the competing "
        "positions are.",
    ),
)

_GENERAL_TASKS: tuple[tuple[str, TaskKind, str], ...] = (
    (
        "Establish the facts",
        TaskKind.LANDSCAPE,
        "For this request: {question}\n\nEstablish the current facts, "
        "including anything that changed recently.",
    ),
    (
        "Collect supporting data",
        TaskKind.MEASUREMENT,
        "For this request: {question}\n\nProvide concrete data, figures, "
        "and documented sources that support or refute the main claims.",
    ),
    (
        "Gather practitioner views",
        TaskKind.COMMUNITY,
        "For this request: {question}\n\nSummarize what people with "
        "direct experience report, and where their views diverge.",
    ),
    (
        "Identify risks and caveats",
        TaskKind.RISK,
        "For this request: {question}\n\nName the caveats, risks, and "
        "conditions under which the usual answer is wrong.",
    ),
)


def _coerce_kind(raw: Any) -> TaskKind:
    """
    Convert a model-supplied kind into a TaskKind.

    Args:
        raw: Value from the model.

    Returns:
        Matching kind, defaulting to LANDSCAPE.
    """
    try:
        return TaskKind(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return TaskKind.LANDSCAPE


def _parse_json_object(raw: str) -> dict[str, Any]:
    """
    Parse the first JSON object found in model output.

    Models wrap JSON in prose or fences often enough that a bare
    ``json.loads`` is not usable on its own.

    Args:
        raw: Raw model output.

    Returns:
        Parsed object, or an empty dict when nothing parses.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = [
    "ResearchPlanner",
]
