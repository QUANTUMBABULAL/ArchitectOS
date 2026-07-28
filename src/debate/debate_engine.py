"""
Iterative multi-provider debate.

A single round of parallel consultation produces a set of independent
opinions. When those opinions disagree, the useful next step is not to
average them — it is to make the disagreeing providers address each
other. DebateEngine does that.

Round structure
---------------
Round 1 sends the question to every ready provider and measures agreement.
Each later round targets only the providers implicated in a contradiction,
asking each to respond to the specific opposing claim. Because providers
keep their conversations across rounds, a follow-up can say "you concluded
X, another system concluded Y" and the provider still remembers concluding
X.

Debate stops at the first of: confidence above the configured threshold,
no significant contradictions remaining, or the round limit. Stopping is
always recorded with a reason so a report never implies more agreement
than was actually reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from src.config import Settings, get_settings
from src.consensus import ConsensusEngine, ConsensusResult, Contradiction, Opinion
from src.events import EventType, get_emitter, research_event
from src.findings import (
    ConsensusAggregate,
    aggregate_findings,
    extract_findings,
)
from src.logger import get_logger
from src.session import BrowserSessionManager
from src.workers import WorkerResponse

from .contradiction_engine import ContradictionEngine, TargetedQuestion


class StopReason(str, Enum):
    """
    Why a debate ended.

    Attributes:
        CONFIDENCE_REACHED: Consensus confidence met the threshold.
        NO_CONTRADICTIONS: No significant disagreement remained.
        MAX_ROUNDS: The round limit was exhausted while disagreement
            persisted.
        INSUFFICIENT_PROVIDERS: Fewer than two providers answered, so
            there was nothing to debate.
        NO_ANSWERS: No provider produced a usable answer.
    """

    CONFIDENCE_REACHED = "confidence_reached"
    NO_CONTRADICTIONS = "no_contradictions"
    MAX_ROUNDS = "max_rounds"
    INSUFFICIENT_PROVIDERS = "insufficient_providers"
    NO_ANSWERS = "no_answers"

    @property
    def is_converged(self) -> bool:
        """
        Return whether the debate ended in agreement.

        Returns:
            True when the debate stopped because it converged rather than
            because it ran out of rounds or participants.
        """
        return self in {
            StopReason.CONFIDENCE_REACHED,
            StopReason.NO_CONTRADICTIONS,
        }


@dataclass(frozen=True, slots=True)
class DebateRound:
    """
    One completed round of debate.

    Attributes:
        number: One-based round number.
        prompts: Prompt sent to each participating provider.
        responses: Responses gathered this round.
        consensus: Consensus analysis over this round's answers.
        participants: Providers asked this round.
        aggregate: Recommendation-level consensus over this round's
            structured findings.
        questions: Targeted follow-ups this round produced for the next
            round.
    """

    number: int
    prompts: dict[str, str]
    responses: list[WorkerResponse]
    consensus: Optional[ConsensusResult]
    participants: list[str] = field(default_factory=list)
    aggregate: Optional[ConsensusAggregate] = None
    questions: tuple[TargetedQuestion, ...] = ()

    @property
    def answered(self) -> list[WorkerResponse]:
        """
        Return responses that carry usable content.

        Returns:
            Successful, non-empty responses.
        """
        return [
            response
            for response in self.responses
            if response.success and response.answer.strip()
        ]


@dataclass(frozen=True, slots=True)
class DebateOutcome:
    """
    Result of a complete debate.

    Attributes:
        question: Original research question.
        rounds: Completed rounds in order.
        stop_reason: Why the debate ended.
        final_consensus: Consensus from the last round that produced one.
        supporting: Providers whose final answers align with the majority.
        opposing: Providers that remained in disagreement.
    """

    question: str
    rounds: list[DebateRound]
    stop_reason: StopReason
    final_consensus: Optional[ConsensusResult] = None
    supporting: list[str] = field(default_factory=list)
    opposing: list[str] = field(default_factory=list)

    @property
    def round_count(self) -> int:
        """
        Return how many rounds were completed.

        Returns:
            Number of rounds.
        """
        return len(self.rounds)

    @property
    def confidence(self) -> float:
        """
        Return the final consensus confidence.

        Returns:
            Confidence, or 0.0 when no consensus was computed.
        """
        return (
            self.final_consensus.confidence
            if self.final_consensus is not None
            else 0.0
        )

    def latest_answers(self) -> dict[str, str]:
        """
        Return each provider's most recent answer across all rounds.

        Returns:
            Mapping of provider name to answer text.
        """
        answers: dict[str, str] = {}
        for round_ in self.rounds:
            for response in round_.answered:
                answers[response.worker_name] = response.answer
        return answers


class DebateEngine:
    """
    Runs iterative multi-provider debate over a research question.

    The engine owns round strategy and stop conditions. Dispatch and
    conversation continuity belong to the session manager; agreement
    measurement belongs to the consensus engine. This class coordinates
    them and contributes no site-specific or model-specific logic.
    """

    def __init__(
        self,
        session: BrowserSessionManager,
        consensus: ConsensusEngine,
        settings: Optional[Settings] = None,
        contradictions: Optional[ContradictionEngine] = None,
    ) -> None:
        """
        Initialize the debate engine.

        Args:
            session: Persistent session used to reach providers.
            consensus: Agreement analyzer.
            settings: Optional application settings supplying the round
                limit and confidence threshold.
            contradictions: Optional contradiction engine producing
                recommendation-level follow-ups.
        """
        self._session = session
        self._consensus = consensus
        self._settings = settings or get_settings()
        self._contradictions = contradictions or ContradictionEngine()
        self._events = get_emitter()
        self._research_id = ""
        self._logger = get_logger(__name__)

    async def run(
        self,
        question: str,
        max_rounds: Optional[int] = None,
        fresh_conversation: bool = True,
    ) -> DebateOutcome:
        """
        Debate a question across providers until it converges.

        Args:
            question: Research question.
            max_rounds: Round limit. Defaults to the configured maximum.
            fresh_conversation: Whether round one should start new
                conversations. True avoids contamination from unrelated
                earlier questions; later rounds always continue.

        Returns:
            Debate outcome with every round recorded.
        """
        limit = max_rounds or self._settings.debate_max_rounds
        threshold = self._settings.debate_confidence_threshold
        rounds: list[DebateRound] = []

        self._research_id = uuid4().hex
        self._events.emit(
            research_event(
                EventType.RESEARCH_STARTED,
                self._research_id,
                question=question,
                providers=self._session.ready_providers(),
                maxRounds=limit,
                confidenceThreshold=threshold,
            )
        )

        first = await self._run_first_round(question, fresh_conversation)
        rounds.append(first)
        self._publish_round(first, limit)

        if not first.answered:
            self._logger.warning("No provider answered; debate cannot start")
            return self._finish(question, rounds, StopReason.NO_ANSWERS)

        if len(first.answered) < 2:
            self._logger.info(
                "Only one provider answered; nothing to debate"
            )
            return self._finish(
                question, rounds, StopReason.INSUFFICIENT_PROVIDERS
            )

        stop = self._evaluate_stop(first.consensus, threshold)
        round_number = 1
        asked: set[tuple[str, str]] = set()

        while stop is None and round_number < limit:
            round_number += 1
            previous = rounds[-1]

            # Recommendation-level disagreement is preferred: it names a
            # specific product and a specific provider, so the follow-up
            # can be precise. Lexical contradictions are the fallback for
            # answers with no extractable recommendations.
            follow_ups = self._structured_follow_ups(previous, asked)
            if not follow_ups:
                contradictions = (
                    previous.consensus.contradictions
                    if previous.consensus is not None
                    else []
                )
                follow_ups = self._build_follow_ups(
                    question, previous, contradictions
                )

            if not follow_ups:
                stop = StopReason.NO_CONTRADICTIONS
                break

            self._logger.info(
                "Debate Round %d: %d provider(s) asked to address "
                "disagreement",
                round_number,
                len(follow_ups),
            )
            self._events.emit(
                research_event(
                    EventType.RESEARCH_ROUND_STARTED,
                    self._research_id,
                    round=round_number,
                    providers=list(follow_ups),
                )
            )
            next_round = await self._run_follow_up_round(
                round_number, question, follow_ups
            )
            rounds.append(next_round)
            self._publish_round(next_round, limit)

            if not next_round.answered:
                self._logger.warning(
                    "Round %d produced no answers; stopping", round_number
                )
                stop = StopReason.MAX_ROUNDS
                break

            self._logger.info(
                "Consensus updated after round %d: agreement=%.2f "
                "confidence=%.2f",
                round_number,
                next_round.consensus.agreement_score
                if next_round.consensus
                else 0.0,
                next_round.consensus.confidence
                if next_round.consensus
                else 0.0,
            )
            stop = self._evaluate_stop(next_round.consensus, threshold)

        return self._finish(question, rounds, stop or StopReason.MAX_ROUNDS)

    async def _run_first_round(
        self,
        question: str,
        fresh_conversation: bool,
    ) -> DebateRound:
        """
        Ask every ready provider the original question.

        Args:
            question: Research question.
            fresh_conversation: Whether to start new conversations.

        Returns:
            Completed round one.
        """
        providers = self._session.ready_providers()
        self._logger.info(
            "Debate Round 1: asking %d provider(s)", len(providers)
        )

        responses = await self._session.dispatch(
            prompt=question,
            providers=providers,
            new_conversation=fresh_conversation,
        )
        consensus = await self._consensus_for(question, responses)

        return DebateRound(
            number=1,
            prompts={name: question for name in providers},
            responses=responses,
            consensus=consensus,
            participants=providers,
            aggregate=self._aggregate_for(question, responses),
        )

    async def _run_follow_up_round(
        self,
        number: int,
        question: str,
        follow_ups: dict[str, str],
    ) -> DebateRound:
        """
        Send targeted follow-ups to disagreeing providers.

        Each provider receives its own prompt, so the round is not a
        broadcast. Prompts are still submitted concurrently.

        Args:
            number: Round number.
            question: Original research question.
            follow_ups: Prompt per provider.

        Returns:
            Completed round.
        """
        import asyncio

        names = list(follow_ups)
        results = await asyncio.gather(
            *(
                self._session.dispatch(
                    prompt=follow_ups[name],
                    providers=[name],
                    new_conversation=False,
                )
                for name in names
            ),
            return_exceptions=True,
        )

        responses: list[WorkerResponse] = []
        for name, result in zip(names, results):
            if isinstance(result, list):
                responses.extend(result)
            else:
                self._logger.warning(
                    "Follow-up to %s failed: %s", name, result
                )

        consensus = await self._consensus_for(question, responses)
        return DebateRound(
            number=number,
            prompts=dict(follow_ups),
            responses=responses,
            consensus=consensus,
            participants=names,
            aggregate=self._aggregate_for(question, responses),
        )

    def _structured_follow_ups(
        self,
        previous: DebateRound,
        asked: set[tuple[str, str]],
    ) -> dict[str, str]:
        """
        Build follow-ups from recommendation-level disagreement.

        Args:
            previous: Round whose findings are being challenged.
            asked: Provider and subject pairs already questioned, updated
                in place so a debate does not repeat a point.

        Returns:
            Mapping of provider name to follow-up prompt. Empty when no
            structured disagreement was found.
        """
        if previous.aggregate is None:
            return {}

        questions = self._contradictions.build(previous.aggregate, asked)
        if not questions:
            return {}

        follow_ups: dict[str, str] = {}
        for item in questions:
            asked.add((item.provider, item.subject))
            # One prompt per provider per round: several disagreements
            # aimed at one provider are better handled across rounds than
            # bundled into a single unfocused message.
            if item.provider not in follow_ups:
                follow_ups[item.provider] = item.question

        return follow_ups

    def _aggregate_for(
        self,
        question: str,
        responses: list[WorkerResponse],
    ) -> Optional[ConsensusAggregate]:
        """
        Extract structured findings and merge them across providers.

        Args:
            question: Question the answers address.
            responses: Round responses.

        Returns:
            Consensus aggregate, or None when nothing usable was returned.
        """
        findings = [
            extract_findings(response.worker_name, response.answer)
            for response in responses
            if response.success and response.answer.strip()
        ]
        if not findings:
            return None

        aggregate = aggregate_findings(question, findings)
        self._logger.info("Findings merged: %s", aggregate.describe())
        for finding in findings:
            self._logger.debug("Extracted %s", finding.describe())
        return aggregate

    async def _consensus_for(
        self,
        question: str,
        responses: list[WorkerResponse],
    ) -> Optional[ConsensusResult]:
        """
        Evaluate consensus across a round's answers.

        Args:
            question: Question the answers address.
            responses: Round responses.

        Returns:
            Consensus result, or None when nothing usable was returned.
        """
        opinions = [
            Opinion(source=response.worker_name, text=response.answer)
            for response in responses
            if response.success and response.answer.strip()
        ]
        if not opinions:
            return None
        return await self._consensus.evaluate(
            question=question, opinions=opinions
        )

    def _build_follow_ups(
        self,
        question: str,
        previous: DebateRound,
        contradictions: list[Contradiction],
    ) -> dict[str, str]:
        """
        Build a targeted follow-up prompt for each disagreeing provider.

        Only providers named in a contradiction are asked again. The
        prompt quotes the opposing position so the provider can address it
        specifically rather than restating its original answer.

        Args:
            question: Original research question.
            previous: Round whose answers are being challenged.
            contradictions: Detected contradictions.

        Returns:
            Mapping of provider name to follow-up prompt.
        """
        if not contradictions:
            return {}

        answers = {
            response.worker_name: response.answer
            for response in previous.answered
        }
        follow_ups: dict[str, str] = {}

        for contradiction in contradictions:
            for target, other in (
                (contradiction.source_a, contradiction.source_b),
                (contradiction.source_b, contradiction.source_a),
            ):
                if target in follow_ups or target not in answers:
                    continue

                excerpt = self._excerpt(answers.get(other, ""))
                follow_ups[target] = (
                    f"Another AI system reached a different conclusion on "
                    f"this question: \"{question}\"\n\n"
                    f"Their position:\n{excerpt}\n\n"
                    f"The specific disagreement is: "
                    f"{contradiction.description}\n\n"
                    f"Do you still hold your earlier conclusion? Explain "
                    f"precisely why they are right or wrong, and state "
                    f"what evidence would settle it. Be brief."
                )

        return follow_ups

    @staticmethod
    def _excerpt(text: str, limit: int = 900) -> str:
        """
        Trim an opposing answer to a quotable excerpt.

        Args:
            text: Full answer text.
            limit: Maximum characters to include.

        Returns:
            Trimmed excerpt.
        """
        cleaned = text.strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + " [...]"

    def _evaluate_stop(
        self,
        consensus: Optional[ConsensusResult],
        threshold: float,
    ) -> Optional[StopReason]:
        """
        Decide whether the debate should stop.

        Args:
            consensus: Consensus from the round just completed.
            threshold: Confidence required to stop early.

        Returns:
            Stop reason, or None to continue debating.
        """
        if consensus is None:
            return StopReason.NO_ANSWERS

        if consensus.confidence >= threshold:
            self._logger.info(
                "Confidence %.2f met the %.2f threshold; debate complete",
                consensus.confidence,
                threshold,
            )
            return StopReason.CONFIDENCE_REACHED

        if not consensus.contradictions:
            self._logger.info(
                "No contradictions remain; debate complete"
            )
            return StopReason.NO_CONTRADICTIONS

        return None

    def _publish_round(self, round_: DebateRound, limit: int) -> None:
        """
        Publish progress and consensus state after a round.

        Args:
            round_: Completed round.
            limit: Maximum rounds, used to compute progress.
        """
        if not self._events.enabled:
            return

        progress = min(1.0, round_.number / max(1, limit))
        self._events.emit(
            research_event(
                EventType.RESEARCH_PROGRESS,
                self._research_id,
                round=round_.number,
                progress=round(progress, 3),
                answered=[r.worker_name for r in round_.answered],
                failed=[
                    r.worker_name
                    for r in round_.responses
                    if not r.success
                ],
            )
        )

        consensus = round_.consensus
        if consensus is None:
            return

        self._events.emit(
            research_event(
                EventType.CONSENSUS_UPDATED,
                self._research_id,
                round=round_.number,
                agreement=round(consensus.agreement_score, 4),
                confidence=round(consensus.confidence, 4),
                opinionCount=consensus.opinion_count,
                contradictions=[
                    {
                        "sourceA": item.source_a,
                        "sourceB": item.source_b,
                        "description": item.description,
                    }
                    for item in consensus.contradictions
                ],
                products=[
                    {
                        "name": product.display_name,
                        "supporters": list(product.supporters),
                        "dissenters": list(product.dissenters),
                        "confidence": product.confidence,
                    }
                    for product in (
                        round_.aggregate.top(8)
                        if round_.aggregate is not None
                        else []
                    )
                ],
            )
        )

        for item in consensus.contradictions:
            self._events.emit(
                research_event(
                    EventType.CONTRADICTION_DETECTED,
                    self._research_id,
                    sourceA=item.source_a,
                    sourceB=item.source_b,
                    description=item.description,
                )
            )

    def _finish(
        self,
        question: str,
        rounds: list[DebateRound],
        stop: StopReason,
    ) -> DebateOutcome:
        """
        Assemble the final outcome and log the result.

        Args:
            question: Original research question.
            rounds: Completed rounds.
            stop: Stop reason.

        Returns:
            Debate outcome.
        """
        final = next(
            (
                round_.consensus
                for round_ in reversed(rounds)
                if round_.consensus is not None
            ),
            None,
        )

        opposing: list[str] = []
        if final is not None:
            for contradiction in final.contradictions:
                for source in (contradiction.source_a, contradiction.source_b):
                    if source not in opposing:
                        opposing.append(source)

        answered = set(
            response.worker_name
            for round_ in rounds
            for response in round_.answered
        )
        supporting = sorted(answered - set(opposing))

        self._logger.info(
            "Debate finished after %d round(s): %s "
            "(confidence=%.2f, supporting=%d, opposing=%d)",
            len(rounds),
            stop.value,
            final.confidence if final else 0.0,
            len(supporting),
            len(opposing),
        )

        self._events.emit(
            research_event(
                EventType.RESEARCH_FINISHED,
                self._research_id,
                stopReason=stop.value,
                converged=stop.is_converged,
                rounds=len(rounds),
                confidence=round(final.confidence, 4) if final else 0.0,
                supporting=supporting,
                opposing=sorted(opposing),
            )
        )

        return DebateOutcome(
            question=question,
            rounds=rounds,
            stop_reason=stop,
            final_consensus=final,
            supporting=supporting,
            opposing=sorted(opposing),
        )


__all__ = [
    "DebateEngine",
    "DebateOutcome",
    "DebateRound",
    "StopReason",
]
