"""
Research orchestrator.

ResearchOrchestrator runs research sessions end to end. It coordinates the
Planner (goal decomposition), the DecisionEngine (local answers and
consultation routing), the WorkerManager (external AI consultation), the
ConsensusEngine (agreement analysis), and the MemoryStore (persistence).
It contains no site-specific, model-specific, or storage-specific logic —
only the control flow that binds the specialists together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.brain import DecisionEngine
from src.consensus import ConsensusEngine, ConsensusResult, Opinion
from src.exceptions import OrchestratorError
from src.logger import get_logger
from src.memory import (
    MemoryStore,
    ReportRecord,
    ResearchRecord,
    ResponseRecord,
    new_id,
)
from src.planner import Planner, PlanStep, PlanStepKind, ResearchPlan
from src.workers import WorkerManager, WorkerQuery, WorkerResponse


_SYNTHESIS_SYSTEM_PROMPT = (
    "You are the synthesis component of a research operating system. "
    "You are given a research goal, sub-questions, and answers gathered "
    "from external AI systems, possibly with consensus notes about "
    "agreement and contradictions. Write a clear, well-structured final "
    "report in Markdown that answers the goal. Attribute claims to their "
    "sources when they disagree, and state open questions explicitly. "
    "Do not invent information that is not in the gathered material."
)


@dataclass(frozen=True, slots=True)
class StepResult:
    """
    Outcome of executing one plan step.

    Attributes:
        step: The executed plan step.
        responses: Responses gathered for the step.
        consensus: Consensus analysis for consultation steps.
    """

    step: PlanStep
    responses: list[WorkerResponse] = field(default_factory=list)
    consensus: Optional[ConsensusResult] = None


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    """
    Final outcome of one research session.

    Attributes:
        research_id: Persisted research session identifier.
        goal: Original user goal.
        report: Final synthesized report text.
        plan: Executed research plan.
        step_results: Per-step execution results.
        consensus_summary: Aggregated consensus metadata.
    """

    research_id: str
    goal: str
    report: str
    plan: ResearchPlan
    step_results: list[StepResult] = field(default_factory=list)
    consensus_summary: dict[str, Any] = field(default_factory=dict)


class ResearchOrchestrator:
    """
    Coordinates planner, decision engine, workers, consensus, and memory.

    One orchestrator instance can run many research sessions sequentially.
    Each session is persisted from the moment planning succeeds, so
    failures always leave an auditable record.
    """

    def __init__(
        self,
        planner: Planner,
        decision_engine: DecisionEngine,
        worker_manager: WorkerManager,
        consensus_engine: ConsensusEngine,
        memory: MemoryStore,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            planner: Planner producing executable plans.
            decision_engine: Local decision engine.
            worker_manager: Registry and dispatcher for workers.
            consensus_engine: Agreement analyzer.
            memory: Persistent research memory.
        """
        self._planner = planner
        self._engine = decision_engine
        self._workers = worker_manager
        self._consensus = consensus_engine
        self._memory = memory
        self._logger = get_logger(__name__)

    async def run_research(self, goal: str) -> ResearchOutcome:
        """
        Execute one research session end to end.

        Args:
            goal: User research goal.

        Returns:
            Research outcome with the final report.

        Raises:
            OrchestratorError: If the session fails.
        """
        plan = await self._plan(goal)
        research_id = new_id()

        await self._memory.save_research(
            ResearchRecord(
                research_id=research_id,
                goal=plan.goal,
                status="running",
                plan=self._serialize_plan(plan),
            )
        )
        self._logger.info(
            "Research %s started: %s", research_id, plan.goal
        )

        try:
            step_results = await self._execute_steps(research_id, plan)
            report, consensus_summary = await self._synthesize(
                research_id, plan, step_results
            )

            await self._memory.save_report(
                ReportRecord(
                    report_id=new_id(),
                    research_id=research_id,
                    content=report,
                    consensus=consensus_summary,
                )
            )
            await self._memory.update_research_status(
                research_id, "completed"
            )
            self._logger.info("Research %s completed", research_id)

            return ResearchOutcome(
                research_id=research_id,
                goal=plan.goal,
                report=report,
                plan=plan,
                step_results=step_results,
                consensus_summary=consensus_summary,
            )
        except Exception as exc:
            await self._mark_failed(research_id)
            if isinstance(exc, OrchestratorError):
                raise
            raise OrchestratorError(
                f"Research {research_id} failed: {exc}",
                code="ORCHESTRATOR_RESEARCH_FAILED",
            ) from exc

    async def run_direct_consultation(
        self,
        goal: str,
        synthesize: bool = False,
    ) -> ResearchOutcome:
        """
        Consult every ready worker directly, bypassing planning.

        This is the fast path for an explicit research command, where the
        user has already declared intent. It skips complexity
        classification and goal decomposition — both of which require
        local inference — and sends the goal verbatim to all ready
        providers concurrently. Consensus, persistence, and reporting are
        unchanged, so the audit trail matches a planned run.

        Providers that fail are recorded and skipped; the run succeeds as
        long as one provider answers.

        Args:
            goal: Research goal, used verbatim as the prompt.
            synthesize: Whether to additionally run local-model synthesis
                over the gathered answers. Off by default because the
                point of this path is to avoid inference latency.

        Returns:
            Research outcome with the unified report.

        Raises:
            OrchestratorError: If no provider produced a usable answer.
        """
        cleaned = goal.strip()
        if not cleaned:
            raise OrchestratorError(
                "Research goal cannot be empty",
                code="ORCHESTRATOR_GOAL_EMPTY",
            )

        workers = [worker.name for worker in self._workers.ready_workers()]
        if not workers:
            raise OrchestratorError(
                "No providers are ready; cannot run a direct consultation",
                code="ORCHESTRATOR_NO_WORKERS",
            )

        step = PlanStep(
            step_id=new_id(),
            index=0,
            kind=PlanStepKind.CONSULT,
            description="Direct multi-provider consultation",
            prompt=cleaned,
        )
        plan = ResearchPlan(
            plan_id=new_id(),
            goal=cleaned,
            steps=(step,),
        )

        research_id = new_id()
        await self._memory.save_research(
            ResearchRecord(
                research_id=research_id,
                goal=cleaned,
                status="running",
                plan=self._serialize_plan(plan),
            )
        )
        self._logger.info(
            "Direct consultation %s started across %d provider(s): %s",
            research_id,
            len(workers),
            ", ".join(workers),
        )

        try:
            responses = await self._workers.dispatch_many(
                workers, WorkerQuery(prompt=cleaned)
            )

            for response in responses:
                await self._persist_response(research_id, step, response)
                if response.success:
                    self._logger.info(
                        "Response received from %s in %.1fs",
                        response.worker_name,
                        response.elapsed_seconds,
                    )
                else:
                    self._logger.warning(
                        "Provider %s failed: %s",
                        response.worker_name,
                        response.error,
                    )

            opinions = [
                Opinion(source=r.worker_name, text=r.answer)
                for r in responses
                if r.success and r.answer.strip()
            ]
            if not opinions:
                raise OrchestratorError(
                    "Every provider failed; no answers to combine. See the "
                    "log for per-provider errors.",
                    code="ORCHESTRATOR_ALL_PROVIDERS_FAILED",
                )

            consensus = await self._consensus.evaluate(
                question=cleaned,
                opinions=opinions,
            )
            step_results = [
                StepResult(
                    step=step,
                    responses=responses,
                    consensus=consensus,
                )
            ]
            consensus_summary = self._summarize_consensus(step_results)

            if synthesize:
                report, consensus_summary = await self._synthesize(
                    research_id, plan, step_results
                )
            else:
                report = self._assemble_report(plan, step_results)

            await self._memory.save_report(
                ReportRecord(
                    report_id=new_id(),
                    research_id=research_id,
                    content=report,
                    consensus=consensus_summary,
                )
            )
            await self._memory.update_research_status(
                research_id, "completed"
            )
            self._logger.info(
                "Direct consultation %s completed (%d/%d providers "
                "answered, agreement=%.2f)",
                research_id,
                len(opinions),
                len(responses),
                consensus.agreement_score,
            )

            return ResearchOutcome(
                research_id=research_id,
                goal=cleaned,
                report=report,
                plan=plan,
                step_results=step_results,
                consensus_summary=consensus_summary,
            )
        except Exception as exc:
            await self._mark_failed(research_id)
            if isinstance(exc, OrchestratorError):
                raise
            raise OrchestratorError(
                f"Direct consultation {research_id} failed: {exc}",
                code="ORCHESTRATOR_DIRECT_FAILED",
            ) from exc

    async def _plan(self, goal: str) -> ResearchPlan:
        """
        Create the research plan.

        Args:
            goal: User research goal.

        Returns:
            Executable research plan.

        Raises:
            OrchestratorError: If planning fails.
        """
        try:
            return await self._planner.create_plan(goal)
        except Exception as exc:
            raise OrchestratorError(
                f"Planning failed: {exc}",
                code="ORCHESTRATOR_PLANNING_FAILED",
            ) from exc

    async def _execute_steps(
        self,
        research_id: str,
        plan: ResearchPlan,
    ) -> list[StepResult]:
        """
        Execute all non-synthesis plan steps in order.

        Args:
            research_id: Persisted research identifier.
            plan: Research plan.

        Returns:
            Step results for LOCAL and CONSULT steps.
        """
        results: list[StepResult] = []
        for step in plan.steps:
            if step.kind == PlanStepKind.LOCAL:
                results.append(
                    await self._execute_local_step(research_id, step)
                )
            elif step.kind == PlanStepKind.CONSULT:
                results.append(
                    await self._execute_consult_step(research_id, step)
                )
        return results

    async def _execute_local_step(
        self,
        research_id: str,
        step: PlanStep,
    ) -> StepResult:
        """
        Execute a local-answer step with the decision engine.

        Args:
            research_id: Persisted research identifier.
            step: LOCAL plan step.

        Returns:
            Step result with one local response.
        """
        answer = await self._engine.answer_locally(step.prompt)

        response = WorkerResponse(
            query_id=new_id(),
            worker_name="local",
            prompt=step.prompt,
            answer=answer,
            success=bool(answer.strip()),
        )
        await self._persist_response(research_id, step, response)
        return StepResult(step=step, responses=[response])

    async def _execute_consult_step(
        self,
        research_id: str,
        step: PlanStep,
    ) -> StepResult:
        """
        Execute a consultation step through external workers.

        The decision engine chooses which ready workers to consult. When
        no worker is ready or routing declines consultation, the step
        degrades to a local answer so research always progresses.

        Args:
            research_id: Persisted research identifier.
            step: CONSULT plan step.

        Returns:
            Step result with worker responses and consensus analysis.
        """
        # Every ready provider is consulted. Selecting a subset saved a
        # little browser time but cost the thing that makes consensus
        # meaningful: a provider that was never asked cannot corroborate
        # or contradict anything, and its silence is indistinguishable
        # from disagreement. Routing is therefore no longer used here, and
        # the local model is not consulted to decide who answers.
        providers = [worker.name for worker in self._workers.ready_workers()]

        if not providers:
            self._logger.info(
                "Step %s has no ready providers; answering locally",
                step.step_id,
            )
            return await self._execute_local_step(research_id, step)

        self._logger.info(
            "Step %s consulting all %d ready provider(s) in parallel: %s",
            step.step_id,
            len(providers),
            ", ".join(providers),
        )

        query = WorkerQuery(prompt=step.prompt)
        responses = await self._workers.dispatch_many(providers, query)

        for response in responses:
            await self._persist_response(research_id, step, response)

        opinions = [
            Opinion(source=response.worker_name, text=response.answer)
            for response in responses
            if response.success and response.answer.strip()
        ]

        consensus: Optional[ConsensusResult] = None
        if opinions:
            consensus = await self._consensus.evaluate(
                question=step.prompt,
                opinions=opinions,
            )

        return StepResult(
            step=step,
            responses=responses,
            consensus=consensus,
        )

    async def _synthesize(
        self,
        research_id: str,
        plan: ResearchPlan,
        step_results: list[StepResult],
    ) -> tuple[str, dict[str, Any]]:
        """
        Produce the final report and aggregated consensus metadata.

        Args:
            research_id: Persisted research identifier.
            plan: Research plan.
            step_results: Executed step results.

        Returns:
            Tuple of (report text, consensus summary).

        Raises:
            OrchestratorError: If no usable material was gathered.
        """
        material = self._gather_material(step_results)
        if not material.strip():
            raise OrchestratorError(
                "No successful answers were gathered; cannot synthesize "
                "a report",
                code="ORCHESTRATOR_NO_MATERIAL",
            )

        consensus_summary = self._summarize_consensus(step_results)

        has_synthesis_step = any(
            step.kind == PlanStepKind.SYNTHESIZE for step in plan.steps
        )
        if not has_synthesis_step:
            # Single local-step plans: the local answer is the report.
            return material, consensus_summary

        try:
            report = await self._engine.client.generate(
                prompt=(
                    f"Research goal: {plan.goal}\n\n"
                    f"Gathered material:\n{material}"
                ),
                system=_SYNTHESIS_SYSTEM_PROMPT,
                options={"temperature": 0.3},
                operation="synthesize",
            )
            if not report.strip():
                raise ValueError("Synthesis produced empty output")
        except Exception as exc:
            self._logger.warning(
                "Model synthesis failed (%s); assembling deterministic "
                "report",
                exc,
            )
            report = self._assemble_report(plan, step_results)

        # Persist the synthesis as a response for full auditability.
        synthesis_step = next(
            step
            for step in plan.steps
            if step.kind == PlanStepKind.SYNTHESIZE
        )
        await self._persist_response(
            research_id,
            synthesis_step,
            WorkerResponse(
                query_id=new_id(),
                worker_name="local",
                prompt=synthesis_step.prompt,
                answer=report,
                success=True,
            ),
        )
        return report.strip(), consensus_summary

    def _gather_material(self, step_results: list[StepResult]) -> str:
        """
        Assemble successful answers into synthesis input material.

        Args:
            step_results: Executed step results.

        Returns:
            Formatted material text.
        """
        sections: list[str] = []
        for result in step_results:
            answers = [
                f"[{response.worker_name}]\n{response.answer.strip()}"
                for response in result.responses
                if response.success and response.answer.strip()
            ]
            if not answers:
                continue

            section = (
                f"### Sub-question: {result.step.prompt}\n\n"
                + "\n\n".join(answers)
            )
            if result.consensus is not None:
                notes = (
                    f"\n\nConsensus: agreement="
                    f"{result.consensus.agreement_score}, confidence="
                    f"{result.consensus.confidence}"
                )
                if result.consensus.contradictions:
                    conflicts = "; ".join(
                        f"{c.source_a} vs {c.source_b}: {c.description}"
                        for c in result.consensus.contradictions
                    )
                    notes += f"\nContradictions: {conflicts}"
                section += notes
            sections.append(section)

        return "\n\n".join(sections)

    def _assemble_report(
        self,
        plan: ResearchPlan,
        step_results: list[StepResult],
    ) -> str:
        """
        Assemble a deterministic report when model synthesis fails.

        Args:
            plan: Research plan.
            step_results: Executed step results.

        Returns:
            Markdown report text.
        """
        lines = [f"# Research Report: {plan.goal}", ""]
        for result in step_results:
            lines.append(f"## {result.step.prompt}")
            lines.append("")
            for response in result.responses:
                if response.success and response.answer.strip():
                    lines.append(f"**Source: {response.worker_name}**")
                    lines.append("")
                    lines.append(response.answer.strip())
                    lines.append("")
            if result.consensus is not None:
                lines.append(
                    f"*Agreement: {result.consensus.agreement_score}, "
                    f"confidence: {result.consensus.confidence}*"
                )
                for contradiction in result.consensus.contradictions:
                    lines.append(
                        f"*Contradiction ({contradiction.source_a} vs "
                        f"{contradiction.source_b}): "
                        f"{contradiction.description}*"
                    )
                lines.append("")
        return "\n".join(lines).strip()

    def _summarize_consensus(
        self,
        step_results: list[StepResult],
    ) -> dict[str, Any]:
        """
        Aggregate per-step consensus into report-level metadata.

        Args:
            step_results: Executed step results.

        Returns:
            Consensus summary dictionary.
        """
        analyses = [
            result.consensus
            for result in step_results
            if result.consensus is not None
        ]
        if not analyses:
            return {
                "consultations": 0,
                "average_confidence": None,
                "contradictions": [],
                "follow_up_questions": [],
            }

        contradictions = [
            {
                "step": result.step.prompt,
                "source_a": contradiction.source_a,
                "source_b": contradiction.source_b,
                "description": contradiction.description,
            }
            for result in step_results
            if result.consensus is not None
            for contradiction in result.consensus.contradictions
        ]
        follow_ups = [
            question
            for analysis in analyses
            for question in analysis.follow_up_questions
        ]

        return {
            "consultations": len(analyses),
            "average_confidence": round(
                sum(analysis.confidence for analysis in analyses)
                / len(analyses),
                4,
            ),
            "average_agreement": round(
                sum(analysis.agreement_score for analysis in analyses)
                / len(analyses),
                4,
            ),
            "contradictions": contradictions,
            "follow_up_questions": follow_ups,
        }

    async def _persist_response(
        self,
        research_id: str,
        step: PlanStep,
        response: WorkerResponse,
    ) -> None:
        """
        Persist one response, never letting storage failures kill a run.

        Args:
            research_id: Persisted research identifier.
            step: Plan step that produced the response.
            response: Response to persist.
        """
        try:
            await self._memory.save_response(
                ResponseRecord(
                    response_id=new_id(),
                    research_id=research_id,
                    step_id=step.step_id,
                    source=response.worker_name,
                    prompt=response.prompt,
                    answer=response.answer,
                    success=response.success,
                    error=response.error,
                    attempts=response.attempts,
                    elapsed_seconds=response.elapsed_seconds,
                )
            )
        except Exception as exc:
            self._logger.error(
                "Failed to persist response for research %s: %s",
                research_id,
                exc,
            )

    async def _mark_failed(self, research_id: str) -> None:
        """
        Mark a research session as failed, tolerating storage errors.

        Args:
            research_id: Persisted research identifier.
        """
        try:
            await self._memory.update_research_status(
                research_id, "failed"
            )
        except Exception as exc:
            self._logger.error(
                "Failed to mark research %s as failed: %s",
                research_id,
                exc,
            )

    def _serialize_plan(self, plan: ResearchPlan) -> dict[str, Any]:
        """
        Serialize a plan for persistence.

        Args:
            plan: Research plan.

        Returns:
            JSON-compatible plan structure.
        """
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "used_fallback": plan.used_fallback,
            "complexity": (
                plan.assessment.complexity.value
                if plan.assessment is not None
                else None
            ),
            "steps": [
                {
                    "step_id": step.step_id,
                    "index": step.index,
                    "kind": step.kind.value,
                    "description": step.description,
                    "prompt": step.prompt,
                    "depends_on": list(step.depends_on),
                }
                for step in plan.steps
            ],
        }


__all__ = [
    "ResearchOrchestrator",
    "ResearchOutcome",
    "StepResult",
]
