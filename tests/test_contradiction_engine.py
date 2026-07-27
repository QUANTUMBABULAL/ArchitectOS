"""
Tests for targeted follow-up generation.

The engine's purpose is precision: it must ask only the providers actually
involved in a disagreement, and only about the specific point of conflict.
A regression that broadcasts questions, or asks a provider about a claim
it never made, defeats the whole mechanism.
"""

from __future__ import annotations

from src.debate import ContradictionEngine, QuestionKind
from src.findings import aggregate_findings, extract_findings


def build_aggregate(answers: dict[str, str]):
    """
    Build an aggregate from raw provider answers.

    Args:
        answers: Mapping of provider name to answer text.

    Returns:
        Consensus aggregate.
    """
    findings = [
        extract_findings(provider, text) for provider, text in answers.items()
    ]
    return aggregate_findings("best walking shoes?", findings)


THREE_WAY = {
    "chatgpt": "1. **Brooks Ghost 15** — great cushion\n2. **Hoka Bondi 8** — soft",
    "claude": (
        "1. **Brooks Ghost 15** — reliable\n"
        "2. **Hoka Bondi 8** — plush\n"
        "3. **Skechers Go Walk 6** — cheap and comfortable"
    ),
    "gemini": "1. **Brooks Ghost 15** — solid\n2. **Hoka Bondi 8** — cushioned",
}


class TestUniqueRecommendations:
    """Questioning outliers."""

    def test_asks_the_outlier_to_defend(self) -> None:
        """Only the provider that named the outlier is questioned."""
        engine = ContradictionEngine()
        questions = engine.build(build_aggregate(THREE_WAY))

        defences = [
            q for q in questions if q.kind is QuestionKind.DEFEND_UNIQUE
        ]
        assert defences
        skechers = next(
            q for q in defences if "skechers" in q.subject.lower()
        )
        assert skechers.provider == "claude"

    def test_question_names_the_subject(self) -> None:
        """The prompt states which recommendation is in question."""
        engine = ContradictionEngine()
        questions = engine.build(build_aggregate(THREE_WAY))
        skechers = next(
            q
            for q in questions
            if "skechers" in q.subject.lower()
            and q.kind is QuestionKind.DEFEND_UNIQUE
        )
        assert "Skechers" in skechers.question
        assert "no other AI system did" in skechers.question

    def test_no_unique_questions_with_one_provider(self) -> None:
        """A single provider cannot be an outlier against itself."""
        engine = ContradictionEngine()
        aggregate = build_aggregate(
            {"chatgpt": "1. **Brooks Ghost 15** — good"}
        )
        assert engine.build(aggregate) == []


class TestOmissions:
    """Questioning providers that left something out."""

    def test_asks_dissenters_why_excluded(self) -> None:
        """
        Providers that omitted a widely recommended product are asked
        directly, and the supporters are named for them.
        """
        answers = dict(THREE_WAY)
        answers["chatgpt"] = (
            "1. **Brooks Ghost 15** — good\n2. **Skechers Go Walk 6** — cheap"
        )
        engine = ContradictionEngine()
        questions = engine.build(build_aggregate(answers))

        omissions = [
            q for q in questions if q.kind is QuestionKind.EXPLAIN_OMISSION
        ]
        assert omissions
        target = next(
            q for q in omissions if "skechers" in q.subject.lower()
        )
        assert target.provider == "gemini"
        assert "Why did you exclude it?" in target.question

    def test_omission_requires_minimum_support(self) -> None:
        """
        A product named by only one provider is an outlier, not an
        omission, so no one is asked to justify excluding it.
        """
        engine = ContradictionEngine(min_support_for_omission=3)
        questions = engine.build(build_aggregate(THREE_WAY))
        assert not [
            q for q in questions if q.kind is QuestionKind.EXPLAIN_OMISSION
        ]


class TestPriceConflicts:
    """Questioning inconsistent prices."""

    def test_flags_large_price_gap(self) -> None:
        """A materially different price is questioned on both sides."""
        engine = ContradictionEngine(price_conflict_ratio=0.2)
        aggregate = build_aggregate(
            {
                "chatgpt": "1. **Widget Pro** — costs $100",
                "claude": "1. **Widget Pro** — costs $200",
            }
        )
        questions = engine.build(aggregate)
        price_questions = [
            q for q in questions if q.kind is QuestionKind.PRICE_CONFLICT
        ]

        assert {q.provider for q in price_questions} == {"chatgpt", "claude"}
        assert "$100" in price_questions[0].question

    def test_ignores_small_price_gap(self) -> None:
        """Minor variation is not treated as disagreement."""
        engine = ContradictionEngine(price_conflict_ratio=0.5)
        aggregate = build_aggregate(
            {
                "chatgpt": "1. **Widget Pro** — costs $100",
                "claude": "1. **Widget Pro** — costs $105",
            }
        )
        assert not [
            q
            for q in engine.build(aggregate)
            if q.kind is QuestionKind.PRICE_CONFLICT
        ]


class TestBudgetAndRepetition:
    """Bounding a round and avoiding loops."""

    def test_respects_max_questions(self) -> None:
        """The round budget is never exceeded."""
        engine = ContradictionEngine(max_questions=2)
        assert len(engine.build(build_aggregate(THREE_WAY))) <= 2

    def test_skips_already_asked_pairs(self) -> None:
        """
        A provider is not asked the same question twice, which is what
        stops a debate looping on one point.
        """
        engine = ContradictionEngine()
        aggregate = build_aggregate(THREE_WAY)

        first = engine.build(aggregate)
        asked = {(q.provider, q.subject) for q in first}
        second = engine.build(aggregate, already_asked=asked)

        assert not {(q.provider, q.subject) for q in second} & asked

    def test_no_questions_when_unanimous(self) -> None:
        """Full agreement produces nothing to ask."""
        engine = ContradictionEngine()
        aggregate = build_aggregate(
            {
                "chatgpt": "1. **Brooks Ghost 15** — good",
                "claude": "1. **Brooks Ghost 15** — good",
            }
        )
        assert engine.build(aggregate) == []

    def test_questions_only_target_involved_providers(self) -> None:
        """No question is addressed to an uninvolved provider."""
        engine = ContradictionEngine()
        aggregate = build_aggregate(THREE_WAY)
        answering = set(aggregate.answering_providers)

        for question in engine.build(aggregate):
            assert question.provider in answering
