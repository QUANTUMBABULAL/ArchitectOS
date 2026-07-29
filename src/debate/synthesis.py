"""
Single concise final answer for a completed debate.

The whole purpose of ArchitectOS is reducing research effort, so the user
must never be handed five long provider transcripts as "the answer". This
module condenses a :class:`DebateOutcome` into one short recommendation
with confidence, supporters, the key disagreement, and sources. The raw
provider responses travel alongside it for an expandable section in the
UI, never as the primary content.

Two summary paths exist:

* A model-written summary via a caller-supplied async ``synthesize``
  function (the local Ollama model). Preferred when available.
* A deterministic summary assembled from consensus data. Always
  available, used as the fallback so a dead model never blocks a result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from src.logger import get_logger

from .debate_engine import DebateOutcome

_SYNTHESIS_TIMEOUT_SECONDS = 90.0
_MAX_ANSWER_CHARS_PER_PROVIDER = 2500

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """
    One concise research result.

    Attributes:
        summary: 2–5 paragraph final recommendation, ready to display.
        confidence: Final consensus confidence in [0, 1].
        supporting: Providers aligned with the recommendation.
        opposing: Providers that remained in disagreement.
        disagreements: Key unresolved disagreements, human-readable.
        sources: Citations extracted from provider answers
            (``{"title": ..., "url": ...}``).
        raw_answers: Each provider's final full answer, for the
            expandable raw section.
        rounds: Debate rounds completed.
        converged: Whether the debate converged.
        stop_reason: Why the debate ended.
        synthesized: True when the summary was model-written.
    """

    summary: str
    confidence: float
    supporting: list[str] = field(default_factory=list)
    opposing: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    raw_answers: dict[str, str] = field(default_factory=dict)
    rounds: int = 0
    converged: bool = False
    stop_reason: str = ""
    synthesized: bool = False

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the WebSocket event stream.

        Returns:
            JSON-compatible payload.
        """
        return {
            "summary": self.summary,
            "confidence": round(self.confidence, 4),
            "supporting": list(self.supporting),
            "opposing": list(self.opposing),
            "disagreements": list(self.disagreements),
            "sources": list(self.sources),
            "rawAnswers": dict(self.raw_answers),
            "rounds": self.rounds,
            "converged": self.converged,
            "stopReason": self.stop_reason,
            "synthesized": self.synthesized,
        }


def _collect_sources(outcome: DebateOutcome) -> list[dict[str, str]]:
    """
    Collect unique citations from every round's structured findings.

    Args:
        outcome: Completed debate.

    Returns:
        Citation dictionaries, capped at 15.
    """
    seen: dict[str, dict[str, str]] = {}
    for round_ in outcome.rounds:
        aggregate = round_.aggregate
        if aggregate is None:
            continue
        for finding in aggregate.findings:
            for citation in finding.citations:
                if citation.url and citation.url not in seen:
                    seen[citation.url] = {
                        "title": citation.title
                        or citation.domain
                        or citation.url,
                        "url": citation.url,
                    }
    return list(seen.values())[:15]


def _collect_disagreements(outcome: DebateOutcome) -> list[str]:
    """
    Render the key unresolved disagreements.

    Args:
        outcome: Completed debate.

    Returns:
        Human-readable disagreement lines, capped at 5.
    """
    consensus = outcome.final_consensus
    if consensus is None:
        return []
    return [
        f"{c.source_a} vs {c.source_b}: {c.description}"
        for c in consensus.contradictions[:5]
    ]


def deterministic_summary(outcome: DebateOutcome) -> str:
    """
    Build a concise summary without any model inference.

    Assembled purely from consensus data, so it is always available and
    never hallucinates: every sentence is backed by a recorded fact.

    Args:
        outcome: Completed debate.

    Returns:
        2–4 short paragraphs of plain text.
    """
    consensus = outcome.final_consensus
    if consensus is None:
        return (
            "No provider returned a usable answer, so no recommendation "
            "can be made. Check provider sign-ins on the Browser page "
            "and try again."
        )

    paragraphs: list[str] = []

    verdict = (
        "The providers converged"
        if outcome.stop_reason.is_converged
        else "The providers did not fully converge"
    )
    top_line = ""
    aggregate = next(
        (
            r.aggregate
            for r in reversed(outcome.rounds)
            if r.aggregate is not None and r.aggregate.products
        ),
        None,
    )
    if aggregate is not None and aggregate.top(1):
        product = aggregate.top(1)[0]
        top_line = (
            f" The leading recommendation is {product.display_name}, "
            f"supported by {', '.join(product.supporters)}."
        )
    paragraphs.append(
        f"{verdict} after {outcome.round_count} round(s) with "
        f"{consensus.confidence:.0%} confidence "
        f"({consensus.opinion_count} providers answering).{top_line}"
    )

    if outcome.supporting:
        aligned = ", ".join(outcome.supporting)
        line = f"Aligned providers: {aligned}."
        if outcome.opposing:
            line += f" Still disagreeing: {', '.join(outcome.opposing)}."
        paragraphs.append(line)

    disagreements = _collect_disagreements(outcome)
    if disagreements:
        paragraphs.append(
            "Key disagreement: " + disagreements[0]
        )

    if not outcome.stop_reason.is_converged:
        paragraphs.append(
            "Treat the positions as unresolved rather than settled; "
            "expand the raw responses below to judge the disagreement "
            "yourself."
        )

    return "\n\n".join(paragraphs)


def synthesis_prompt(outcome: DebateOutcome) -> str:
    """
    Build the local-model prompt for the concise final answer.

    Args:
        outcome: Completed debate.

    Returns:
        Prompt text.
    """
    answers = outcome.latest_answers()
    sections = []
    for provider, answer in sorted(answers.items()):
        trimmed = answer.strip()[:_MAX_ANSWER_CHARS_PER_PROVIDER]
        sections.append(f"--- {provider} ---\n{trimmed}")

    disagreements = _collect_disagreements(outcome)
    disagreement_text = (
        "\n".join(f"- {line}" for line in disagreements)
        if disagreements
        else "None recorded."
    )

    return (
        "You are ArchitectOS, a research engine that consulted several "
        "AI providers about one question and must now write ONE concise "
        "final answer.\n\n"
        f"Question: {outcome.question}\n\n"
        f"Provider answers:\n\n{chr(10).join(sections)}\n\n"
        f"Recorded disagreements:\n{disagreement_text}\n\n"
        "Write the final answer now. Requirements:\n"
        "- 2 to 4 short paragraphs, plain prose, no headings and no "
        "bullet lists.\n"
        "- Open with the single clearest recommendation.\n"
        "- Mention which providers agree and name any key disagreement "
        "in one sentence.\n"
        "- Do not repeat the providers' answers; condense them.\n"
        "- Do not mention that you are a language model."
    )


async def build_final_answer(
    outcome: DebateOutcome,
    synthesize: Optional[Callable[[str], Awaitable[str]]] = None,
) -> FinalAnswer:
    """
    Condense a debate outcome into one concise final answer.

    Args:
        outcome: Completed debate.
        synthesize: Optional async function (prompt -> text) that writes
            the summary with the local model. On absence, failure, or
            timeout the deterministic summary is used — the user always
            gets a result.

    Returns:
        Final answer ready for presentation.
    """
    summary = ""
    synthesized = False

    if synthesize is not None and outcome.latest_answers():
        try:
            summary = (
                await asyncio.wait_for(
                    synthesize(synthesis_prompt(outcome)),
                    timeout=_SYNTHESIS_TIMEOUT_SECONDS,
                )
            ).strip()
            synthesized = bool(summary)
        except Exception as exc:
            logger.warning(
                "Final-answer synthesis failed (%s); using the "
                "deterministic summary",
                exc,
            )

    if not summary:
        summary = deterministic_summary(outcome)

    return FinalAnswer(
        summary=summary,
        confidence=outcome.confidence,
        supporting=list(outcome.supporting),
        opposing=list(outcome.opposing),
        disagreements=_collect_disagreements(outcome),
        sources=_collect_sources(outcome),
        raw_answers=outcome.latest_answers(),
        rounds=outcome.round_count,
        converged=outcome.stop_reason.is_converged,
        stop_reason=outcome.stop_reason.value,
        synthesized=synthesized,
    )


__all__ = [
    "FinalAnswer",
    "build_final_answer",
    "deterministic_summary",
    "synthesis_prompt",
]
