"""
The research operator: plan, distribute, gather, verify, report.

This is what separates a research engine from a prompt fan-out. A fan-out
sends one question to every provider and concatenates the replies. The
operator instead runs an investigation:

1. **Plan** — decompose the request into subtasks that establish
   different things (:mod:`src.research.planner`).
2. **Distribute** — assign each subtask to the provider whose declared
   capabilities suit it, so no two providers do the same work unless the
   task is important enough to deserve a second opinion.
3. **Gather** — dispatch every subtask concurrently and extract
   structured evidence from each answer (:mod:`src.research.evidence`).
4. **Verify** — measure cross-provider agreement over the *claims*, and
   record the conflicts rather than averaging them away.
5. **Report** — produce one executive report and keep the raw answers
   behind it (:mod:`src.research.executive`).

Every stage publishes an event, so Mission Control shows what the engine
is doing rather than a spinner.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from src.config import Settings, get_settings
from src.consensus import ConsensusEngine
from src.consensus.consensus_engine import Opinion
from src.events import EventType, WorkerPhase, get_emitter, research_event
from src.findings import (
    ConsensusAggregate,
    ProviderFindings,
    aggregate_findings,
    extract_findings,
)
from src.logger import get_logger
from src.session import BrowserSessionManager
from src.workers.base_worker import WorkerResponse

from .evidence import EvidenceExtractor, EvidenceSet
from .executive import ExecutiveReport, ExecutiveReportBuilder
from .plan import ResearchPlan, ResearchTask
from .planner import ResearchPlanner


class ResearchStage(str, Enum):
    """
    Coarse stage of a research run, shown in Mission Control.

    Attributes:
        PLANNING: Decomposing the request into subtasks.
        DISTRIBUTING: Assigning subtasks to providers.
        GATHERING: Providers are answering their subtasks.
        EXTRACTING: Converting answers into structured evidence.
        VERIFYING: Cross-checking claims between providers.
        REPORTING: Writing the executive report.
        DONE: Finished.
    """

    PLANNING = "Planning"
    DISTRIBUTING = "Distributing"
    GATHERING = "Gathering"
    EXTRACTING = "Extracting evidence"
    VERIFYING = "Verifying claims"
    REPORTING = "Writing report"
    DONE = "Done"


# Relative cost of each stage, used to report honest progress instead of
# a timer. Gathering dominates because it is the only stage that waits on
# a browser.
_STAGE_WEIGHTS: dict[str, float] = {
    ResearchStage.PLANNING.value: 0.08,
    ResearchStage.DISTRIBUTING.value: 0.02,
    ResearchStage.GATHERING.value: 0.62,
    ResearchStage.EXTRACTING.value: 0.08,
    ResearchStage.VERIFYING.value: 0.12,
    ResearchStage.REPORTING.value: 0.08,
}


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """
    Everything one research run produced.

    Attributes:
        research_id: Identifier shared by every event of this run.
        plan: The executed plan, with assignments.
        evidence: Structured evidence gathered.
        report: The executive report.
        responses: Raw worker responses, per task.
        agreement: Cross-provider agreement over claims, in [0, 1].
        elapsed_seconds: Wall-clock duration.
    """

    research_id: str
    plan: ResearchPlan
    evidence: EvidenceSet
    report: ExecutiveReport
    responses: list[WorkerResponse] = field(default_factory=list)
    agreement: float = 0.0
    elapsed_seconds: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream and the UI.

        Returns:
            JSON-compatible payload.
        """
        return {
            "researchId": self.research_id,
            "plan": self.plan.to_payload(),
            "evidence": self.evidence.to_payload(),
            "report": self.report.to_payload(),
            "agreement": round(self.agreement, 4),
            "elapsedSeconds": round(self.elapsed_seconds, 1),
        }


class ResearchOperator:
    """
    Runs an investigation across providers and reports on it.

    The operator owns orchestration only. Planning, extraction, scoring,
    and reporting each live in their own module, so any one of them can
    be replaced without touching the pipeline.
    """

    def __init__(
        self,
        session: BrowserSessionManager,
        consensus: ConsensusEngine,
        planner: ResearchPlanner,
        report_builder: ExecutiveReportBuilder,
        extractor: Optional[EvidenceExtractor] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the operator.

        Args:
            session: Persistent provider session manager.
            consensus: Engine used to score agreement and find conflicts.
            planner: Produces the research plan.
            report_builder: Produces the executive report.
            extractor: Optional evidence extractor.
            settings: Optional application settings.
        """
        self._session = session
        self._consensus = consensus
        self._planner = planner
        self._reports = report_builder
        self._evidence = extractor or EvidenceExtractor()
        self._settings = settings or get_settings()
        self._events = get_emitter()
        self._logger = get_logger(__name__)
        self._research_id = ""
        self._progress = 0.0

    async def run(self, question: str) -> ResearchResult:
        """
        Execute one research request end to end.

        Args:
            question: The user's research request.

        Returns:
            The completed research result.

        Raises:
            WorkerError: If no provider is available to do the work.
        """
        started = asyncio.get_running_loop().time()
        self._research_id = uuid4().hex
        self._progress = 0.0

        providers = self._session.ready_providers()
        self._events.emit(
            research_event(
                EventType.RESEARCH_STARTED,
                self._research_id,
                question=question,
                providers=providers,
                stage=ResearchStage.PLANNING.value,
            )
        )

        # 1. Plan
        self._stage(ResearchStage.PLANNING, "Decomposing the request")
        plan = await self._planner.plan(question)
        self._events.emit(
            research_event(
                EventType.RESEARCH_PLANNED,
                self._research_id,
                plan=plan.to_payload(),
            )
        )
        self._advance(ResearchStage.PLANNING)

        # 2. Distribute
        self._stage(ResearchStage.DISTRIBUTING, "Assigning subtasks")
        assignments = self._distribute(plan, providers)
        plan = ResearchPlan(
            question=plan.question,
            objective=plan.objective,
            tasks=assignments,
            generated_by=plan.generated_by,
        )
        for task in assignments:
            self._events.emit(
                research_event(
                    EventType.TASK_ASSIGNED,
                    self._research_id,
                    task=task.to_payload(),
                )
            )
        self._advance(ResearchStage.DISTRIBUTING)

        # 3. Gather
        self._stage(
            ResearchStage.GATHERING,
            f"Investigating {len(assignments)} subtask(s)",
        )
        responses = await self._gather(assignments)
        self._advance(ResearchStage.GATHERING)

        # 4. Extract + verify
        self._stage(ResearchStage.EXTRACTING, "Extracting evidence")
        evidence = self._extract(assignments, responses)
        self._events.emit(
            research_event(
                EventType.EVIDENCE_EXTRACTED,
                self._research_id,
                itemCount=len(evidence.items),
                providers=evidence.providers,
            )
        )
        self._advance(ResearchStage.EXTRACTING)

        self._stage(ResearchStage.VERIFYING, "Cross-checking claims")
        agreement, conflicts = await self._verify(question, responses)
        aggregate = self._aggregate(question, responses)
        self._advance(ResearchStage.VERIFYING)

        # 5. Report
        self._stage(ResearchStage.REPORTING, "Writing the executive report")
        report = await self._reports.build(
            plan=plan,
            evidence=evidence,
            aggregate=aggregate,
            agreement=agreement,
            contradictions=conflicts,
        )
        self._advance(ResearchStage.REPORTING)

        elapsed = asyncio.get_running_loop().time() - started
        result = ResearchResult(
            research_id=self._research_id,
            plan=plan,
            evidence=evidence,
            report=report,
            responses=responses,
            agreement=agreement,
            elapsed_seconds=elapsed,
        )

        self._events.emit(
            research_event(
                EventType.RESEARCH_FINISHED,
                self._research_id,
                stage=ResearchStage.DONE.value,
                progress=1.0,
                confidence=round(report.confidence, 4),
                headline=report.headline,
                elapsedSeconds=round(elapsed, 1),
                supporting=report.supporting_providers,
                opposing=report.dissenting_providers,
            )
        )
        self._logger.info(
            "Research complete in %.1fs: %s (confidence %.0f%%, "
            "%d evidence item(s))",
            elapsed,
            report.headline,
            report.confidence * 100,
            len(evidence.items),
        )
        return result

    def _distribute(
        self,
        plan: ResearchPlan,
        providers: list[str],
    ) -> list[ResearchTask]:
        """
        Assign subtasks to providers by capability, then by load.

        Each task goes to the ready provider whose declared capabilities
        best match the task's kind, preferring providers with the fewest
        tasks so far. When there are more providers than tasks, the
        highest-value tasks are duplicated to a second provider so the
        important findings get independent corroboration.

        Args:
            plan: The plan to distribute.
            providers: Ready provider names.

        Returns:
            Assigned tasks. Empty when no provider is ready.
        """
        tasks = plan.investigation_tasks
        if not providers or not tasks:
            return []

        capabilities = self._provider_capabilities(providers)
        load: dict[str, int] = {name: 0 for name in providers}
        assigned: list[ResearchTask] = []

        for task in tasks:
            best = self._best_provider(task, providers, capabilities, load)
            load[best] += 1
            assigned.append(task.assign(best))

        # Spare capacity is spent on corroboration, never on repeating the
        # user's question to everybody.
        idle = [name for name in providers if load[name] == 0]
        for index, name in enumerate(idle):
            source = tasks[index % len(tasks)]
            load[name] += 1
            assigned.append(source.assign(name))

        return [
            ResearchTask(
                task_id=index,
                title=task.title,
                question=task.question,
                kind=task.kind,
                assigned_to=task.assigned_to,
            )
            for index, task in enumerate(assigned, start=1)
        ]

    def _provider_capabilities(
        self,
        providers: list[str],
    ) -> dict[str, frozenset[str]]:
        """
        Read declared capabilities for each ready provider.

        Args:
            providers: Provider names.

        Returns:
            Capability tags per provider; empty when unavailable.
        """
        declared = self._session.provider_capabilities()
        return {
            name: declared.get(name, frozenset({"general"}))
            for name in providers
        }

    @staticmethod
    def _best_provider(
        task: ResearchTask,
        providers: list[str],
        capabilities: dict[str, frozenset[str]],
        load: dict[str, int],
    ) -> str:
        """
        Choose the provider best suited to one task.

        Suitability is normalized to [0, 1] and then subtracted from the
        provider's current load. That ordering matters: capability
        decides between equally-loaded providers, but load decides
        between differently-loaded ones, so a single well-tagged provider
        can never absorb the whole plan while others sit idle.

        Args:
            task: Task to assign.
            providers: Ready provider names.
            capabilities: Capability tags per provider.
            load: Tasks already assigned per provider.

        Returns:
            Chosen provider name.
        """
        preferred = task.kind.preferred_capabilities
        # Earlier preferences are worth more; the maximum is the score of
        # a provider declaring every preferred capability.
        weights = {
            capability: len(preferred) - rank
            for rank, capability in enumerate(preferred)
        }
        best_possible = sum(weights.values()) or 1

        def score(name: str) -> float:
            tags = capabilities.get(name, frozenset())
            match = sum(
                weight
                for capability, weight in weights.items()
                if capability in tags
            )
            return load.get(name, 0) - (match / best_possible)

        return min(providers, key=score)

    async def _gather(
        self,
        tasks: list[ResearchTask],
    ) -> list[WorkerResponse]:
        """
        Dispatch every subtask concurrently to its assigned provider.

        Args:
            tasks: Assigned tasks.

        Returns:
            Responses in task order; failures are recorded, not raised.
        """
        instruction = self._evidence.instruction_block()

        async def ask(task: ResearchTask) -> Optional[WorkerResponse]:
            self._events.provider(
                EventType.PROVIDER_STARTED,
                task.assigned_to or "unknown",
                phase=WorkerPhase.THINKING,
                stage=task.title,
                taskId=task.task_id,
                taskTitle=task.title,
                taskKind=task.kind.value,
            )
            try:
                results = await self._session.dispatch(
                    prompt=task.question + instruction,
                    providers=[task.assigned_to],
                    new_conversation=False,
                )
            except Exception as exc:
                self._logger.warning(
                    "Task %d (%s) failed: %s", task.task_id, task.title, exc
                )
                return None

            response = results[0] if results else None
            self._events.emit(
                research_event(
                    EventType.TASK_FINISHED,
                    self._research_id,
                    taskId=task.task_id,
                    taskTitle=task.title,
                    provider=task.assigned_to,
                    success=bool(response and response.success),
                )
            )
            return response

        gathered = await asyncio.gather(
            *(ask(task) for task in tasks), return_exceptions=True
        )
        return [
            item
            for item in gathered
            if isinstance(item, WorkerResponse)
        ]

    def _extract(
        self,
        tasks: list[ResearchTask],
        responses: list[WorkerResponse],
    ) -> EvidenceSet:
        """
        Convert answers into structured evidence.

        Args:
            tasks: Assigned tasks, used to label each answer.
            responses: Worker responses.

        Returns:
            Evidence set including the raw answers.
        """
        by_provider: dict[str, list[ResearchTask]] = {}
        for task in tasks:
            by_provider.setdefault(task.assigned_to or "", []).append(task)

        items = []
        raw: dict[str, str] = {}

        for response in responses:
            if not response.success or not response.answer.strip():
                continue
            queue = by_provider.get(response.worker_name) or []
            task = queue.pop(0) if queue else None
            title = task.title if task else "Research"
            task_id = task.task_id if task else 0

            label = f"{response.worker_name} · {title}"
            raw[label] = response.answer
            items.extend(
                self._evidence.extract(
                    provider=response.worker_name,
                    task_id=task_id,
                    task_title=title,
                    answer=response.answer,
                )
            )

        return EvidenceSet(items=items, raw_answers=raw)

    async def _verify(
        self,
        question: str,
        responses: list[WorkerResponse],
    ) -> tuple[float, list[str]]:
        """
        Score agreement and collect conflicts between answers.

        Args:
            question: The original request.
            responses: Worker responses.

        Returns:
            Tuple of (agreement score, human-readable conflicts).
        """
        opinions = [
            Opinion(source=response.worker_name, text=response.answer)
            for response in responses
            if response.success and response.answer.strip()
        ]
        if len(opinions) < 2:
            return (0.0, [])

        try:
            result = await self._consensus.evaluate(
                question=question, opinions=opinions
            )
        except Exception as exc:
            self._logger.warning("Consensus evaluation failed: %s", exc)
            return (0.0, [])

        conflicts = [
            f"{item.source_a} vs {item.source_b}: {item.description}"
            for item in result.contradictions[:5]
        ]
        for conflict in conflicts:
            self._events.emit(
                research_event(
                    EventType.CONTRADICTION_DETECTED,
                    self._research_id,
                    description=conflict,
                )
            )
        self._events.emit(
            research_event(
                EventType.CONSENSUS_UPDATED,
                self._research_id,
                agreement=round(result.agreement_score, 4),
                confidence=round(result.confidence, 4),
                opinionCount=result.opinion_count,
            )
        )
        return (result.agreement_score, conflicts)

    def _aggregate(
        self,
        question: str,
        responses: list[WorkerResponse],
    ) -> Optional[ConsensusAggregate]:
        """
        Merge recommendation-level findings across answers.

        Args:
            question: The original request.
            responses: Worker responses.

        Returns:
            Aggregate when anything was extractable, otherwise None.
        """
        findings: list[ProviderFindings] = []
        for response in responses:
            if not response.success or not response.answer.strip():
                continue
            try:
                findings.append(
                    extract_findings(response.worker_name, response.answer)
                )
            except Exception:
                continue

        if not findings:
            return None
        try:
            return aggregate_findings(question, findings)
        except Exception as exc:
            self._logger.debug("Finding aggregation skipped: %s", exc)
            return None

    def _stage(self, stage: ResearchStage, detail: str) -> None:
        """
        Publish the start of a pipeline stage.

        Args:
            stage: Stage being entered.
            detail: One-line description of the work.
        """
        self._logger.info("Research stage: %s — %s", stage.value, detail)
        self._events.emit(
            research_event(
                EventType.RESEARCH_PROGRESS,
                self._research_id,
                stage=stage.value,
                detail=detail,
                progress=round(self._progress, 3),
            )
        )

    def _advance(self, stage: ResearchStage) -> None:
        """
        Record a completed stage and republish progress.

        Progress is the sum of completed stage weights, not a timer, so
        it never runs ahead of the work.

        Args:
            stage: Stage that completed.
        """
        self._progress = min(
            1.0, self._progress + _STAGE_WEIGHTS.get(stage.value, 0.0)
        )
        self._events.emit(
            research_event(
                EventType.RESEARCH_PROGRESS,
                self._research_id,
                stage=stage.value,
                progress=round(self._progress, 3),
                completed=True,
            )
        )


__all__ = [
    "ResearchOperator",
    "ResearchResult",
    "ResearchStage",
]
