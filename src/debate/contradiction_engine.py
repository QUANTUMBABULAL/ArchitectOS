"""
Targeted follow-up generation from recommendation-level disagreement.

Text-similarity contradiction detection can tell you that two answers
differ. It cannot tell you *what* to ask about. Structured findings can:
if one provider recommended a product nobody else did, the useful question
is addressed to that provider; if most providers recommended a product and
one omitted it, the useful question is addressed to the one that omitted
it.

This engine produces those questions and nothing else. It asks only the
providers implicated in a specific disagreement, so a round never
broadcasts, and it never asks a provider about a claim it did not make.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.findings import ConsensusAggregate, ProductConsensus
from src.logger import get_logger


class QuestionKind(str, Enum):
    """
    Category of disagreement a follow-up addresses.

    Attributes:
        DEFEND_UNIQUE: One provider recommended something no other did,
            and is asked to justify it.
        EXPLAIN_OMISSION: A provider omitted something most others
            recommended, and is asked why.
        PRICE_CONFLICT: Providers quoted materially different prices for
            the same product.
    """

    DEFEND_UNIQUE = "defend_unique"
    EXPLAIN_OMISSION = "explain_omission"
    PRICE_CONFLICT = "price_conflict"


@dataclass(frozen=True, slots=True)
class TargetedQuestion:
    """
    One follow-up aimed at a single provider.

    Attributes:
        provider: Provider being questioned.
        kind: Category of disagreement.
        subject: Product or claim under discussion.
        question: Prompt text to send.
        counterparties: Other providers whose position motivated the
            question.
    """

    provider: str
    kind: QuestionKind
    subject: str
    question: str
    counterparties: tuple[str, ...] = ()

    def describe(self) -> str:
        """
        Render a one-line summary for logs.

        Returns:
            Human-readable description.
        """
        others = ", ".join(self.counterparties) or "none"
        return (
            f"{self.provider} <- {self.kind.value} on '{self.subject}' "
            f"(vs {others})"
        )


class ContradictionEngine:
    """
    Builds targeted follow-up questions from a consensus aggregate.

    Question selection is prioritized so a limited round budget is spent
    on the most informative disagreements first: contested products where
    several providers agreed and someone dissented, then unique
    recommendations, then price conflicts.
    """

    def __init__(
        self,
        max_questions: int = 6,
        price_conflict_ratio: float = 0.25,
        min_support_for_omission: int = 2,
    ) -> None:
        """
        Initialize the engine.

        Args:
            max_questions: Maximum questions produced per round, bounding
                how many providers a single round can involve.
            price_conflict_ratio: Relative price spread above which a
                price disagreement is worth questioning.
            min_support_for_omission: How many providers must recommend a
                product before omitting it is treated as a disagreement
                rather than ordinary variation.
        """
        self._max_questions = max(1, max_questions)
        self._price_ratio = max(0.0, price_conflict_ratio)
        self._min_support = max(2, min_support_for_omission)
        self._logger = get_logger(__name__)

    def build(
        self,
        aggregate: ConsensusAggregate,
        already_asked: Optional[set[tuple[str, str]]] = None,
    ) -> list[TargetedQuestion]:
        """
        Generate follow-up questions for the detected disagreements.

        Args:
            aggregate: Cross-provider consensus for the current round.
            already_asked: Provider and subject pairs asked in earlier
                rounds, so a debate does not loop on the same point.

        Returns:
            Questions ordered by informativeness, at most one per provider
            per subject.
        """
        asked = set(already_asked or set())
        questions: list[TargetedQuestion] = []

        for builder in (
            self._omission_questions,
            self._unique_questions,
            self._price_questions,
        ):
            for question in builder(aggregate):
                marker = (question.provider, question.subject)
                if marker in asked:
                    continue
                asked.add(marker)
                questions.append(question)
                if len(questions) >= self._max_questions:
                    self._log(questions)
                    return questions

        self._log(questions)
        return questions

    def _log(self, questions: list[TargetedQuestion]) -> None:
        """
        Log the generated question set.

        Args:
            questions: Questions produced this round.
        """
        if not questions:
            self._logger.info(
                "No recommendation-level contradictions to question"
            )
            return

        self._logger.info(
            "Generated %d targeted follow-up(s): %s",
            len(questions),
            "; ".join(question.describe() for question in questions),
        )

    def _omission_questions(
        self,
        aggregate: ConsensusAggregate,
    ) -> list[TargetedQuestion]:
        """
        Ask providers why they omitted a widely recommended product.

        Args:
            aggregate: Consensus aggregate.

        Returns:
            Omission questions, most-supported product first.
        """
        questions: list[TargetedQuestion] = []

        for product in aggregate.contested:
            if product.support_count < self._min_support:
                continue

            supporters = ", ".join(product.supporters)
            for dissenter in product.dissenters:
                questions.append(
                    TargetedQuestion(
                        provider=dissenter,
                        kind=QuestionKind.EXPLAIN_OMISSION,
                        subject=product.display_name,
                        counterparties=product.supporters,
                        question=(
                            f"{product.support_count} other AI systems "
                            f"({supporters}) recommended "
                            f"\"{product.display_name}\" for this question, "
                            f"and you did not include it.\n\n"
                            f"Why did you exclude it? State plainly whether "
                            f"you consider it a poor fit, whether you were "
                            f"unaware of it, or whether you simply ranked "
                            f"other options higher. If you now think it "
                            f"belongs, say so. Be brief and specific."
                        ),
                    )
                )
        return questions

    def _unique_questions(
        self,
        aggregate: ConsensusAggregate,
    ) -> list[TargetedQuestion]:
        """
        Ask providers to justify recommendations nobody else made.

        Args:
            aggregate: Consensus aggregate.

        Returns:
            Defence questions, highest-confidence outlier first.
        """
        if len(aggregate.answering_providers) < 2:
            return []

        questions: list[TargetedQuestion] = []
        ordered = sorted(
            aggregate.unique,
            key=lambda product: -product.confidence,
        )

        for product in ordered:
            provider = product.supporters[0]
            others = tuple(
                name
                for name in aggregate.answering_providers
                if name != provider
            )
            others_text = ", ".join(others) or "the other systems"

            questions.append(
                TargetedQuestion(
                    provider=provider,
                    kind=QuestionKind.DEFEND_UNIQUE,
                    subject=product.display_name,
                    counterparties=others,
                    question=(
                        f"You recommended \"{product.display_name}\" for "
                        f"this question, and no other AI system did "
                        f"({others_text} all omitted it).\n\n"
                        f"Explain your reasoning. What specifically makes it "
                        f"a good choice here, and what evidence supports "
                        f"that? If on reflection it was a weak "
                        f"recommendation, say so directly. Be brief."
                    ),
                )
            )
        return questions

    def _price_questions(
        self,
        aggregate: ConsensusAggregate,
    ) -> list[TargetedQuestion]:
        """
        Ask about materially inconsistent prices for one product.

        Args:
            aggregate: Consensus aggregate.

        Returns:
            Price conflict questions.
        """
        questions: list[TargetedQuestion] = []

        for product in aggregate.products:
            ratio = product.price_disagreement_ratio
            if ratio is None or ratio < self._price_ratio:
                continue

            quotes = ", ".join(
                f"{provider} said {price.render()}"
                for provider, price in product.prices
            )
            for provider, price in product.prices:
                others = tuple(
                    name for name, _ in product.prices if name != provider
                )
                questions.append(
                    TargetedQuestion(
                        provider=provider,
                        kind=QuestionKind.PRICE_CONFLICT,
                        subject=f"{product.display_name} price",
                        counterparties=others,
                        question=(
                            f"AI systems gave inconsistent prices for "
                            f"\"{product.display_name}\": {quotes}.\n\n"
                            f"You said {price.render()}. What is that figure "
                            f"based on, how current is it, and which "
                            f"configuration or region does it apply to? If "
                            f"you are not confident in it, say so."
                        ),
                    )
                )
        return questions


__all__ = [
    "ContradictionEngine",
    "QuestionKind",
    "TargetedQuestion",
]
