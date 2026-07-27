"""
Tests for multi-round debate.

The properties that matter: round one asks everyone, later rounds ask only
the providers implicated in a contradiction, follow-ups continue existing
conversations rather than starting new ones, and the debate stops for a
recorded reason. A debate that ends without converging must never be
reported as agreement.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from src.consensus import ConsensusResult, Contradiction
from src.debate.debate_engine import DebateEngine, StopReason
from src.debate.report import render_report
from src.workers import WorkerResponse


class FakeSession:
    """
    Session double recording every dispatch.

    Substitutes for BrowserSessionManager so debate logic can be tested
    without a browser.
    """

    def __init__(
        self,
        providers: list[str],
        answers: Optional[dict[str, str]] = None,
        failing: Optional[set[str]] = None,
    ) -> None:
        """
        Initialize the double.

        Args:
            providers: Providers reported as ready.
            answers: Canned answer per provider.
            failing: Providers that always fail.
        """
        self._providers = providers
        self._answers = answers or {p: f"answer from {p}" for p in providers}
        self._failing = failing or set()
        self.dispatches: list[dict[str, object]] = []

    def ready_providers(self) -> list[str]:
        """Return ready providers."""
        return list(self._providers)

    async def dispatch(
        self,
        prompt: str,
        providers: Optional[list[str]] = None,
        new_conversation: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> list[WorkerResponse]:
        """Record the dispatch and return canned responses."""
        targets = providers if providers is not None else self._providers
        self.dispatches.append(
            {
                "prompt": prompt,
                "providers": list(targets),
                "new_conversation": new_conversation,
            }
        )
        return [
            WorkerResponse(
                query_id="q",
                worker_name=name,
                prompt=prompt,
                answer=""
                if name in self._failing
                else self._answers.get(name, "answer"),
                success=name not in self._failing,
                error="failed" if name in self._failing else None,
            )
            for name in targets
        ]

    async def maybe_reset_for_context(self) -> list[str]:
        """No-op for tests."""
        return []


class ScriptedConsensus:
    """Consensus double returning a scripted sequence of results."""

    def __init__(self, results: list[ConsensusResult]) -> None:
        """
        Initialize the double.

        Args:
            results: Results returned in order; the last repeats.
        """
        self._results = results
        self.calls = 0

    async def evaluate(self, question: str, opinions: list) -> ConsensusResult:
        """Return the next scripted result."""
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]


def consensus(
    confidence: float,
    contradictions: Optional[list[Contradiction]] = None,
    opinions: int = 2,
) -> ConsensusResult:
    """
    Build a consensus result for tests.

    Args:
        confidence: Confidence value.
        contradictions: Optional contradictions.
        opinions: Opinion count.

    Returns:
        Consensus result.
    """
    return ConsensusResult(
        agreement_score=confidence,
        confidence=confidence,
        consensus_reached=confidence >= 0.5,
        contradictions=contradictions or [],
        opinion_count=opinions,
    )


class Settings:
    """Minimal settings stub for the debate engine."""

    debate_max_rounds = 3
    debate_confidence_threshold = 0.8


def run_debate(session: FakeSession, results: list[ConsensusResult]):
    """
    Run a debate with scripted consensus.

    Args:
        session: Session double.
        results: Scripted consensus results.

    Returns:
        Tuple of outcome and consensus double.
    """
    scripted = ScriptedConsensus(results)
    engine = DebateEngine(
        session=session,  # type: ignore[arg-type]
        consensus=scripted,  # type: ignore[arg-type]
        settings=Settings(),  # type: ignore[arg-type]
    )
    outcome = asyncio.run(engine.run("is X true?"))
    return outcome, scripted


class TestRoundOne:
    """The opening round."""

    def test_asks_every_provider(self) -> None:
        """Round one broadcasts to all ready providers."""
        session = FakeSession(["chatgpt", "claude", "gemini"])
        outcome, _ = run_debate(session, [consensus(0.95)])

        assert outcome.round_count == 1
        assert session.dispatches[0]["providers"] == [
            "chatgpt",
            "claude",
            "gemini",
        ]

    def test_starts_fresh_conversation_by_default(self) -> None:
        """
        Round one starts clean so an unrelated earlier question cannot
        contaminate the answer.
        """
        session = FakeSession(["chatgpt", "claude"])
        run_debate(session, [consensus(0.95)])
        assert session.dispatches[0]["new_conversation"] is True

    def test_stops_immediately_on_high_confidence(self) -> None:
        """Meeting the threshold in round one ends the debate."""
        session = FakeSession(["chatgpt", "claude"])
        outcome, _ = run_debate(session, [consensus(0.95)])

        assert outcome.stop_reason is StopReason.CONFIDENCE_REACHED
        assert outcome.stop_reason.is_converged is True
        assert outcome.round_count == 1

    def test_stops_when_no_contradictions(self) -> None:
        """Low confidence but no disagreement is still a stop condition."""
        session = FakeSession(["chatgpt", "claude"])
        outcome, _ = run_debate(session, [consensus(0.4, [])])
        assert outcome.stop_reason is StopReason.NO_CONTRADICTIONS


class TestFollowUpRounds:
    """Targeted later rounds."""

    def test_only_disagreeing_providers_are_asked(self) -> None:
        """
        Round two must not broadcast. Only the two providers named in the
        contradiction are re-asked; the third is left alone.
        """
        session = FakeSession(["chatgpt", "claude", "gemini"])
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="disagree on X",
            similarity=0.1,
        )
        outcome, _ = run_debate(
            session,
            [consensus(0.3, [contradiction]), consensus(0.95)],
        )

        assert outcome.round_count == 2
        later = [
            d for d in session.dispatches[1:]
        ]
        asked = {p for d in later for p in d["providers"]}  # type: ignore[index]
        assert asked == {"chatgpt", "claude"}
        assert "gemini" not in asked

    def test_follow_ups_continue_the_conversation(self) -> None:
        """
        Follow-ups must not start a new conversation, or the provider will
        have forgotten the claim it is being asked to defend.
        """
        session = FakeSession(["chatgpt", "claude"])
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="disagree",
            similarity=0.1,
        )
        run_debate(
            session, [consensus(0.3, [contradiction]), consensus(0.95)]
        )

        for dispatch in session.dispatches[1:]:
            assert dispatch["new_conversation"] is False

    def test_follow_up_quotes_the_opposing_position(self) -> None:
        """The prompt includes the other provider's answer and the clash."""
        session = FakeSession(
            ["chatgpt", "claude"],
            answers={"chatgpt": "X is true", "claude": "X is false"},
        )
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="opposite conclusions on X",
            similarity=0.1,
        )
        run_debate(
            session, [consensus(0.3, [contradiction]), consensus(0.95)]
        )

        prompts = [str(d["prompt"]) for d in session.dispatches[1:]]
        joined = "\n".join(prompts)
        assert "opposite conclusions on X" in joined
        assert "X is false" in joined or "X is true" in joined

    def test_stops_at_max_rounds(self) -> None:
        """Persistent disagreement stops at the configured limit."""
        session = FakeSession(["chatgpt", "claude"])
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="disagree",
            similarity=0.1,
        )
        outcome, _ = run_debate(session, [consensus(0.3, [contradiction])])

        assert outcome.round_count == Settings.debate_max_rounds
        assert outcome.stop_reason is StopReason.MAX_ROUNDS
        assert outcome.stop_reason.is_converged is False


class TestDegradedCases:
    """Too few or no answers."""

    def test_single_provider_is_not_debated(self) -> None:
        """One opinion cannot disagree with itself."""
        session = FakeSession(["chatgpt"])
        outcome, _ = run_debate(session, [consensus(0.3, [], opinions=1)])

        assert outcome.stop_reason is StopReason.INSUFFICIENT_PROVIDERS
        assert outcome.round_count == 1

    def test_all_providers_failing_stops_cleanly(self) -> None:
        """Total failure is reported, not raised."""
        session = FakeSession(
            ["chatgpt", "claude"], failing={"chatgpt", "claude"}
        )
        outcome, _ = run_debate(session, [consensus(0.0, [], opinions=0)])

        assert outcome.stop_reason is StopReason.NO_ANSWERS
        assert outcome.confidence == 0.0

    def test_one_failing_provider_does_not_stop_debate(self) -> None:
        """A failing provider is excluded, others continue."""
        session = FakeSession(
            ["chatgpt", "claude", "gemini"], failing={"gemini"}
        )
        outcome, _ = run_debate(session, [consensus(0.95)])

        assert outcome.stop_reason is StopReason.CONFIDENCE_REACHED
        assert "gemini" not in outcome.latest_answers()


class TestOutcomeAndReport:
    """Result assembly and rendering."""

    def test_tracks_supporting_and_opposing(self) -> None:
        """Providers named in contradictions are recorded as opposing."""
        session = FakeSession(["chatgpt", "claude", "gemini"])
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="clash",
            similarity=0.1,
        )
        outcome, _ = run_debate(session, [consensus(0.3, [contradiction])])

        assert set(outcome.opposing) == {"chatgpt", "claude"}
        assert "gemini" in outcome.supporting

    def test_latest_answers_prefers_later_rounds(self) -> None:
        """A provider's most recent answer wins."""
        session = FakeSession(["chatgpt", "claude"])
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="clash",
            similarity=0.1,
        )
        outcome, _ = run_debate(
            session, [consensus(0.3, [contradiction]), consensus(0.95)]
        )
        assert set(outcome.latest_answers()) == {"chatgpt", "claude"}

    def test_report_marks_non_convergence(self) -> None:
        """
        A debate that ran out of rounds must be labelled unresolved, not
        presented as consensus.
        """
        session = FakeSession(["chatgpt", "claude"])
        contradiction = Contradiction(
            source_a="chatgpt",
            source_b="claude",
            description="clash",
            similarity=0.1,
        )
        outcome, _ = run_debate(session, [consensus(0.3, [contradiction])])
        report = render_report(outcome)

        assert "Not converged" in report
        assert "without convergence" in report
        assert "clash" in report

    def test_report_includes_rounds_and_positions(self) -> None:
        """The transcript and final positions are both present."""
        session = FakeSession(["chatgpt", "claude"])
        outcome, _ = run_debate(session, [consensus(0.95)])
        report = render_report(outcome)

        assert "## Consensus" in report
        assert "Converged" in report
        assert "## Final positions" in report
        assert "## Debate transcript" in report
        assert "Round 1" in report

    def test_report_handles_no_consensus(self) -> None:
        """A report is still produced when nothing was gathered."""
        session = FakeSession(["chatgpt"], failing={"chatgpt"})
        outcome, _ = run_debate(session, [consensus(0.0, [], opinions=0)])
        report = render_report(outcome)

        assert "No consensus could be computed" in report
