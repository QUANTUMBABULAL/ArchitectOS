"""
Consensus engine for comparing multiple AI opinions.

The engine measures pairwise agreement between answers, detects likely
contradictions, produces a confidence score, and generates follow-up
questions aimed at resolving disagreement. Its deterministic core (lexical
similarity plus negation and numeric-divergence heuristics) always works;
an optional DecisionEngine refines contradiction analysis and follow-up
generation using the local model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from src.brain import DecisionEngine
from src.config import Settings, get_settings
from src.exceptions import ConsensusError
from src.logger import get_logger


@dataclass(frozen=True, slots=True)
class Opinion:
    """
    One answer from one AI source.

    Attributes:
        source: Name of the AI system that produced the answer.
        text: Answer text.
    """

    source: str
    text: str


@dataclass(frozen=True, slots=True)
class Contradiction:
    """
    A detected contradiction between two opinions.

    Attributes:
        source_a: First source name.
        source_b: Second source name.
        description: Explanation of the suspected contradiction.
        similarity: Lexical similarity between the two answers.
    """

    source_a: str
    source_b: str
    description: str
    similarity: float


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """
    Outcome of comparing multiple AI opinions.

    Attributes:
        agreement_score: Mean pairwise similarity between 0.0 and 1.0.
        confidence: Overall confidence in the combined answer.
        consensus_reached: Whether agreement meets the configured threshold.
        contradictions: Detected contradictions.
        follow_up_questions: Questions that would resolve disagreement.
        opinion_count: Number of opinions compared.
        pairwise_similarities: Similarity per source pair.
    """

    agreement_score: float
    confidence: float
    consensus_reached: bool
    contradictions: list[Contradiction] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    opinion_count: int = 0
    pairwise_similarities: dict[str, float] = field(default_factory=dict)


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
        "for", "from", "has", "have", "if", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "their", "there", "these",
        "they", "this", "to", "was", "were", "which", "will", "with",
        "you", "your",
    }
)

_NEGATION_MARKERS: frozenset[str] = frozenset(
    {
        "not", "no", "never", "cannot", "can't", "don't", "doesn't",
        "won't", "isn't", "aren't", "shouldn't", "couldn't", "wouldn't",
        "false", "incorrect", "wrong", "impossible",
    }
)

_CONTRADICTION_SYSTEM_PROMPT = (
    "You compare two AI answers to the same question and judge whether "
    "they contradict each other on any substantive claim. Respond ONLY "
    'with a JSON object of the form {"contradicts": true|false, '
    '"description": "<one sentence naming the conflicting claims>"}.'
)

_FOLLOW_UP_SYSTEM_PROMPT = (
    "You generate follow-up research questions that would resolve "
    "disagreements between AI answers. Respond ONLY with a JSON object "
    'of the form {"questions": ["question", ...]} containing between '
    "1 and %(max_questions)d questions."
)


class ConsensusEngine:
    """
    Compares AI opinions and quantifies their agreement.

    Similarity combines token-set overlap with sequence similarity so both
    wording and content contribute. Contradiction detection first applies
    cheap deterministic heuristics; when a DecisionEngine is available it
    is used to confirm or refine suspected contradictions.
    """

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        settings: Optional[Settings] = None,
        max_follow_up_questions: int = 3,
    ) -> None:
        """
        Initialize the consensus engine.

        Args:
            decision_engine: Optional local model access for refined
                analysis.
            settings: Optional application settings.
            max_follow_up_questions: Maximum generated follow-up questions.

        Raises:
            ConsensusError: If max_follow_up_questions is not positive.
        """
        if max_follow_up_questions < 1:
            raise ConsensusError(
                "max_follow_up_questions must be at least 1",
                code="CONSENSUS_MAX_QUESTIONS_INVALID",
            )
        self._engine = decision_engine
        self._settings = settings or get_settings()
        self._max_questions = max_follow_up_questions
        self._logger = get_logger(__name__)

    async def evaluate(
        self,
        question: str,
        opinions: list[Opinion],
    ) -> ConsensusResult:
        """
        Evaluate consensus across multiple opinions.

        Args:
            question: Question the opinions answer.
            opinions: Opinions to compare.

        Returns:
            Consensus result. With fewer than the configured minimum
            number of opinions, agreement defaults to full and confidence
            is reduced to reflect the missing corroboration.

        Raises:
            ConsensusError: If no opinions are provided.
        """
        valid = [
            opinion for opinion in opinions if opinion.text.strip()
        ]
        if not valid:
            raise ConsensusError(
                "Consensus evaluation requires at least one non-empty "
                "opinion",
                code="CONSENSUS_NO_OPINIONS",
            )

        if len(valid) < self._settings.consensus_min_agents:
            return ConsensusResult(
                agreement_score=1.0,
                confidence=0.5,
                consensus_reached=False,
                contradictions=[],
                follow_up_questions=[],
                opinion_count=len(valid),
                pairwise_similarities={},
            )

        similarities: dict[str, float] = {}
        suspected: list[tuple[Opinion, Opinion, float, str]] = []

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i], valid[j]
                similarity = self._similarity(a.text, b.text)
                similarities[f"{a.source}|{b.source}"] = round(
                    similarity, 4
                )

                reason = self._heuristic_contradiction(a.text, b.text)
                if reason is not None:
                    suspected.append((a, b, similarity, reason))

        agreement = (
            sum(similarities.values()) / len(similarities)
            if similarities
            else 1.0
        )

        contradictions = await self._confirm_contradictions(
            question, suspected
        )

        confidence = self._confidence(
            agreement=agreement,
            opinion_count=len(valid),
            contradiction_count=len(contradictions),
        )
        consensus_reached = (
            agreement >= self._settings.consensus_threshold
            and not contradictions
        )

        follow_ups: list[str] = []
        if contradictions or not consensus_reached:
            follow_ups = await self._follow_up_questions(
                question, contradictions
            )

        return ConsensusResult(
            agreement_score=round(agreement, 4),
            confidence=round(confidence, 4),
            consensus_reached=consensus_reached,
            contradictions=contradictions,
            follow_up_questions=follow_ups,
            opinion_count=len(valid),
            pairwise_similarities=similarities,
        )

    def _similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute combined lexical similarity of two answers.

        Args:
            text_a: First answer.
            text_b: Second answer.

        Returns:
            Similarity between 0.0 and 1.0.
        """
        tokens_a = self._content_tokens(text_a)
        tokens_b = self._content_tokens(text_b)

        if tokens_a and tokens_b:
            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        else:
            jaccard = 0.0

        sequence = SequenceMatcher(
            None,
            text_a.lower()[:4000],
            text_b.lower()[:4000],
        ).ratio()

        return 0.6 * jaccard + 0.4 * sequence

    def _heuristic_contradiction(
        self,
        text_a: str,
        text_b: str,
    ) -> Optional[str]:
        """
        Detect a suspected contradiction with deterministic heuristics.

        Two signals are used: strongly asymmetric negation density over a
        shared topic, and disagreeing numeric values.

        Args:
            text_a: First answer.
            text_b: Second answer.

        Returns:
            Reason string when a contradiction is suspected, else None.
        """
        tokens_a = self._content_tokens(text_a)
        tokens_b = self._content_tokens(text_b)
        if not tokens_a or not tokens_b:
            return None

        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if overlap < 0.05:
            return None

        density_a = self._negation_density(text_a)
        density_b = self._negation_density(text_b)
        if abs(density_a - density_b) > 0.02:
            return (
                "One answer negates claims the other asserts "
                f"(negation densities {density_a:.3f} vs {density_b:.3f})"
            )

        numbers_a = self._numbers(text_a)
        numbers_b = self._numbers(text_b)
        if numbers_a and numbers_b and not (numbers_a & numbers_b):
            return (
                "The answers cite entirely different numeric values "
                "for a shared topic"
            )

        return None

    async def _confirm_contradictions(
        self,
        question: str,
        suspected: list[tuple[Opinion, Opinion, float, str]],
    ) -> list[Contradiction]:
        """
        Confirm suspected contradictions, using the local model if present.

        Args:
            question: Original question.
            suspected: Suspected contradiction tuples.

        Returns:
            Confirmed contradictions.
        """
        confirmed: list[Contradiction] = []

        for a, b, similarity, reason in suspected:
            description = reason

            if self._engine is not None:
                try:
                    raw = await self._engine.client.generate(
                        prompt=(
                            f"Question: {question}\n\n"
                            f"Answer from {a.source}:\n{a.text[:2000]}\n\n"
                            f"Answer from {b.source}:\n{b.text[:2000]}"
                        ),
                        system=_CONTRADICTION_SYSTEM_PROMPT,
                        options={"temperature": 0.0},
                    )
                    data = self._parse_json_object(raw)
                    if not bool(data.get("contradicts", True)):
                        continue
                    model_description = str(
                        data.get("description", "")
                    ).strip()
                    if model_description:
                        description = model_description
                except Exception as exc:
                    self._logger.warning(
                        "Model contradiction check failed (%s); keeping "
                        "heuristic verdict",
                        exc,
                    )

            confirmed.append(
                Contradiction(
                    source_a=a.source,
                    source_b=b.source,
                    description=description,
                    similarity=round(similarity, 4),
                )
            )

        return confirmed

    async def _follow_up_questions(
        self,
        question: str,
        contradictions: list[Contradiction],
    ) -> list[str]:
        """
        Generate follow-up questions to resolve disagreement.

        Args:
            question: Original question.
            contradictions: Confirmed contradictions.

        Returns:
            Follow-up questions, at most the configured maximum.
        """
        if self._engine is not None:
            try:
                conflict_lines = "\n".join(
                    f"- {c.source_a} vs {c.source_b}: {c.description}"
                    for c in contradictions
                ) or "- The answers diverge without a named contradiction"

                raw = await self._engine.client.generate(
                    prompt=(
                        f"Question: {question}\n\n"
                        f"Disagreements:\n{conflict_lines}"
                    ),
                    system=_FOLLOW_UP_SYSTEM_PROMPT
                    % {"max_questions": self._max_questions},
                    options={"temperature": 0.3},
                )
                data = self._parse_json_object(raw)
                candidates = data.get("questions", [])
                if isinstance(candidates, list):
                    questions = [
                        str(candidate).strip()
                        for candidate in candidates
                        if str(candidate).strip()
                    ]
                    if questions:
                        return questions[: self._max_questions]
            except Exception as exc:
                self._logger.warning(
                    "Model follow-up generation failed (%s); using "
                    "template questions",
                    exc,
                )

        questions = [
            (
                f"Sources {c.source_a} and {c.source_b} disagree: "
                f"{c.description}. Which position is correct, and what "
                "evidence supports it?"
            )
            for c in contradictions
        ]
        if not questions:
            questions = [
                f"What additional evidence would confirm the answer to: "
                f"{question}"
            ]
        return questions[: self._max_questions]

    def _confidence(
        self,
        agreement: float,
        opinion_count: int,
        contradiction_count: int,
    ) -> float:
        """
        Compute overall confidence.

        Confidence starts from agreement, grows with corroborating
        sources, and shrinks with each confirmed contradiction.

        Args:
            agreement: Mean pairwise agreement.
            opinion_count: Number of opinions.
            contradiction_count: Number of confirmed contradictions.

        Returns:
            Confidence between 0.0 and 1.0.
        """
        source_bonus = min(0.15, 0.05 * (opinion_count - 1))
        contradiction_penalty = 0.2 * contradiction_count
        return max(
            0.0,
            min(1.0, agreement + source_bonus - contradiction_penalty),
        )

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        """
        Tokenize text into lowercase content words.

        Args:
            text: Input text.

        Returns:
            Set of tokens with stopwords removed.
        """
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        return {
            token
            for token in tokens
            if len(token) > 2 and token not in _STOPWORDS
        }

    @staticmethod
    def _negation_density(text: str) -> float:
        """
        Compute the fraction of words that are negation markers.

        Args:
            text: Input text.

        Returns:
            Negation marker density.
        """
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return 0.0
        negations = sum(1 for word in words if word in _NEGATION_MARKERS)
        return negations / len(words)

    @staticmethod
    def _numbers(text: str) -> set[str]:
        """
        Extract normalized numeric literals from text.

        Args:
            text: Input text.

        Returns:
            Set of numeric strings.
        """
        return set(re.findall(r"\d+(?:\.\d+)?", text))

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
    "ConsensusEngine",
    "ConsensusResult",
    "Contradiction",
    "Opinion",
]
