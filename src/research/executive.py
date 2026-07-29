"""
Executive report assembly.

The output of a research operator is a decision, not a transcript. This
module turns gathered evidence into one report a person can act on in
under two minutes: the recommendation, how confident it is and why, what
would argue against it, what to pick instead under different priorities,
and the sources.

Two paths produce the prose, exactly as in the rest of the engine: the
local model writes it when available, and a deterministic assembler
writes it from the same structured evidence when the model is not. Both
paths are bounded to the same length budget, and neither ever emits a
provider transcript — raw answers travel separately, behind the UI's
Evidence section.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from src.findings.aggregator import ConsensusAggregate
from src.logger import get_logger

from .evidence import EvidenceItem, EvidenceSet
from .plan import ResearchPlan

logger = get_logger(__name__)

_MIN_WORDS = 500
_MAX_WORDS = 800
_SYNTHESIS_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class Alternative:
    """
    A runner-up worth choosing under a specific priority.

    Attributes:
        priority: What this option is best for.
        choice: The option itself.
        rationale: One line on why it wins for that priority.
    """

    priority: str
    choice: str
    rationale: str = ""

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream.

        Returns:
            JSON-compatible payload.
        """
        return {
            "priority": self.priority,
            "choice": self.choice,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ExecutiveReport:
    """
    The deliverable of one research run.

    Attributes:
        question: The original request.
        headline: The recommendation in a few words.
        headline_label: What the headline answers, e.g. "Best phone".
        summary: The report prose, 500–800 words.
        confidence: Confidence in the recommendation, in [0, 1].
        evidence_points: The strongest supporting claims.
        weaknesses: Reasons the recommendation might be wrong.
        alternatives: Better choices under different priorities.
        sources: Supporting URLs.
        supporting_providers: Providers whose evidence backs the headline.
        dissenting_providers: Providers that disagreed.
        contradictions: Unresolved conflicts between providers.
        word_count: Word count of the summary.
        synthesized: True when the local model wrote the prose.
    """

    question: str
    headline: str
    headline_label: str
    summary: str
    confidence: float
    evidence_points: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    supporting_providers: list[str] = field(default_factory=list)
    dissenting_providers: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    word_count: int = 0
    synthesized: bool = False

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream and the UI.

        Returns:
            JSON-compatible payload.
        """
        return {
            "question": self.question,
            "headline": self.headline,
            "headlineLabel": self.headline_label,
            "summary": self.summary,
            "confidence": round(self.confidence, 4),
            "evidencePoints": list(self.evidence_points),
            "weaknesses": list(self.weaknesses),
            "alternatives": [alt.to_payload() for alt in self.alternatives],
            "sources": list(self.sources),
            "supportingProviders": list(self.supporting_providers),
            "dissentingProviders": list(self.dissenting_providers),
            "contradictions": list(self.contradictions),
            "wordCount": self.word_count,
            "synthesized": self.synthesized,
        }


class ExecutiveReportBuilder:
    """
    Builds an executive report from evidence.

    The builder owns presentation only. Everything it states must already
    exist in the evidence it was given, which is what keeps the report
    auditable against the expandable raw answers.
    """

    def __init__(
        self,
        synthesize: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ) -> None:
        """
        Initialize the builder.

        Args:
            synthesize: Optional async ``(system, prompt) -> text``
                function backed by the local model. Omitted or failing,
                the deterministic assembler is used instead.
        """
        self._synthesize = synthesize
        self._logger = get_logger(__name__)

    async def build(
        self,
        plan: ResearchPlan,
        evidence: EvidenceSet,
        aggregate: Optional[ConsensusAggregate] = None,
        agreement: float = 0.0,
        contradictions: Optional[list[str]] = None,
    ) -> ExecutiveReport:
        """
        Assemble the report.

        Args:
            plan: The executed research plan.
            evidence: Evidence gathered across providers.
            aggregate: Optional recommendation-level consensus, used to
                pick the headline when the request compared options.
            agreement: Cross-provider agreement score in [0, 1].
            contradictions: Unresolved conflicts, human-readable.

        Returns:
            Report ready for presentation.
        """
        conflicts = list(contradictions or [])
        headline, label, supporters = self._pick_headline(evidence, aggregate)
        confidence = self._score_confidence(
            evidence, aggregate, agreement, supporters, conflicts
        )
        points = self._evidence_points(evidence)
        weaknesses = self._weaknesses(evidence, conflicts)
        alternatives = self._alternatives(evidence, aggregate, headline)
        sources = self._sources(evidence)
        dissenters = self._dissenters(evidence, supporters)

        summary = ""
        synthesized = False
        if self._synthesize is not None:
            summary = await self._model_summary(
                plan, headline, label, confidence, points, weaknesses,
                alternatives, conflicts,
            )
            synthesized = bool(summary)

        if not summary:
            summary = self._assembled_summary(
                plan, headline, label, confidence, points, weaknesses,
                alternatives, supporters, dissenters, conflicts, evidence,
            )

        summary = _enforce_word_budget(summary)

        return ExecutiveReport(
            question=plan.question,
            headline=headline,
            headline_label=label,
            summary=summary,
            confidence=confidence,
            evidence_points=points,
            weaknesses=weaknesses,
            alternatives=alternatives,
            sources=sources,
            supporting_providers=supporters,
            dissenting_providers=dissenters,
            contradictions=conflicts,
            word_count=len(summary.split()),
            synthesized=synthesized,
        )

    def _pick_headline(
        self,
        evidence: EvidenceSet,
        aggregate: Optional[ConsensusAggregate],
    ) -> tuple[str, str, list[str]]:
        """
        Choose the recommendation and who supports it.

        A named product recommended by several providers is the strongest
        possible headline, so structured findings win when they exist.
        Otherwise the most corroborated claim becomes the headline.

        Args:
            evidence: Gathered evidence.
            aggregate: Optional recommendation-level consensus.

        Returns:
            Tuple of (headline, label, supporting providers).
        """
        if aggregate is not None and aggregate.products:
            best = aggregate.top(1)[0]
            return (
                best.display_name,
                "Recommendation",
                list(best.supporters),
            )

        groups = evidence.grouped()
        if groups:
            best_group = groups[0]
            supporters = sorted({item.provider for item in best_group})
            return (
                _headline_from_fact(best_group[0].fact),
                "Finding",
                supporters,
            )

        return ("No usable answer", "Result", [])

    def _score_confidence(
        self,
        evidence: EvidenceSet,
        aggregate: Optional[ConsensusAggregate],
        agreement: float,
        supporters: list[str],
        conflicts: list[str],
    ) -> float:
        """
        Score confidence from corroboration, agreement, and conflict.

        The score is explainable by construction: it is the weighted sum
        of how many independent providers back the headline, how much
        their answers agreed overall, and how confident they said they
        were — reduced by unresolved contradictions.

        Args:
            evidence: Gathered evidence.
            aggregate: Optional recommendation-level consensus.
            agreement: Cross-provider agreement score.
            supporters: Providers backing the headline.
            conflicts: Unresolved contradictions.

        Returns:
            Confidence in [0, 1].
        """
        providers = evidence.providers
        if not providers:
            return 0.0

        support_ratio = len(supporters) / max(len(providers), 1)
        stated = [item.confidence for item in evidence.items]
        stated_mean = sum(stated) / len(stated) if stated else 0.5

        if aggregate is not None and aggregate.products:
            best = aggregate.top(1)[0]
            support_ratio = max(support_ratio, best.confidence)

        score = 0.45 * support_ratio + 0.30 * agreement + 0.25 * stated_mean
        score -= 0.06 * len(conflicts)

        # A single provider can never produce high confidence: with
        # nothing to corroborate against, agreement is not measurable.
        if len(providers) < 2:
            score = min(score, 0.55)

        return max(0.0, min(1.0, score))

    def _evidence_points(self, evidence: EvidenceSet) -> list[str]:
        """
        Select the strongest supporting claims.

        Args:
            evidence: Gathered evidence.

        Returns:
            Up to six claim lines, corroboration first.
        """
        points: list[str] = []
        for group in evidence.grouped():
            providers = sorted({item.provider for item in group})
            fact = max(group, key=lambda item: item.confidence).fact
            suffix = (
                f" ({len(providers)} providers agree)"
                if len(providers) > 1
                else f" ({providers[0]})"
            )
            points.append(f"{fact}{suffix}")
            if len(points) >= 6:
                break
        return points

    def _weaknesses(
        self,
        evidence: EvidenceSet,
        conflicts: list[str],
    ) -> list[str]:
        """
        Collect what argues against the recommendation.

        Args:
            evidence: Gathered evidence.
            conflicts: Unresolved contradictions.

        Returns:
            Up to five weakness lines.
        """
        weaknesses: list[str] = []
        for item in evidence.items:
            for caveat in item.caveats:
                if caveat and caveat not in weaknesses:
                    weaknesses.append(caveat)

        for item in evidence.items:
            lowered = item.fact.lower()
            if any(
                marker in lowered
                for marker in (
                    "however", "but ", "weak", "lacks", "worse",
                    "downside", "drawback", "complaint", "issue",
                )
            ):
                if item.fact not in weaknesses:
                    weaknesses.append(item.fact)

        weaknesses.extend(
            conflict for conflict in conflicts if conflict not in weaknesses
        )
        return weaknesses[:5]

    def _alternatives(
        self,
        evidence: EvidenceSet,
        aggregate: Optional[ConsensusAggregate],
        headline: str,
    ) -> list[Alternative]:
        """
        Derive runner-up choices under different priorities.

        Args:
            evidence: Gathered evidence.
            aggregate: Optional recommendation-level consensus.
            headline: The chosen recommendation.

        Returns:
            Up to four alternatives.
        """
        alternatives: list[Alternative] = []
        if aggregate is None or not aggregate.products:
            return alternatives

        for product in aggregate.top(5):
            if product.display_name == headline:
                continue
            priority = _priority_for(product.display_name, evidence)
            alternatives.append(
                Alternative(
                    priority=priority,
                    choice=product.display_name,
                    rationale=(
                        f"backed by {', '.join(product.supporters)}"
                        if product.supporters
                        else ""
                    ),
                )
            )
            if len(alternatives) >= 4:
                break
        return alternatives

    def _sources(self, evidence: EvidenceSet) -> list[dict[str, str]]:
        """
        Collect the supporting links.

        Args:
            evidence: Gathered evidence.

        Returns:
            Up to fifteen sources with a display title.
        """
        sources: list[dict[str, str]] = []
        for url in evidence.all_links[:15]:
            host = url.split("//", 1)[-1].split("/", 1)[0]
            sources.append(
                {"title": host[4:] if host.startswith("www.") else host,
                 "url": url}
            )
        return sources

    def _dissenters(
        self,
        evidence: EvidenceSet,
        supporters: list[str],
    ) -> list[str]:
        """
        Return providers that contributed but did not back the headline.

        Args:
            evidence: Gathered evidence.
            supporters: Providers backing the headline.

        Returns:
            Dissenting provider names.
        """
        return [
            provider
            for provider in evidence.providers
            if provider not in supporters
        ]

    async def _model_summary(
        self,
        plan: ResearchPlan,
        headline: str,
        label: str,
        confidence: float,
        points: list[str],
        weaknesses: list[str],
        alternatives: list[Alternative],
        conflicts: list[str],
    ) -> str:
        """
        Ask the local model to write the report prose.

        The model is given only the structured evidence — never the raw
        provider answers — so it cannot smuggle an unverified claim into
        the report.

        Args:
            plan: The executed plan.
            headline: Chosen recommendation.
            label: What the headline answers.
            confidence: Computed confidence.
            points: Supporting claims.
            weaknesses: Counter-arguments.
            alternatives: Runner-ups.
            conflicts: Unresolved contradictions.

        Returns:
            Report prose, or an empty string on failure.
        """
        system = (
            "You write the executive summary of a research report for a "
            "decision-maker. You are given the verified findings only. "
            f"Write between {_MIN_WORDS} and {_MAX_WORDS} words in plain "
            "prose paragraphs. State the recommendation in the first "
            "sentence. Explain what supports it, then what argues "
            "against it, then when a different choice would be better. "
            "Use ONLY the findings supplied; invent nothing; do not "
            "mention providers by name, AI models, or that this came "
            "from multiple sources. No headings, no bullet lists."
        )
        prompt = (
            f"Research request: {plan.question}\n"
            f"Objective: {plan.objective}\n\n"
            f"{label}: {headline}\n"
            f"Confidence: {confidence:.0%}\n\n"
            "Supporting findings:\n"
            + "\n".join(f"- {point}" for point in points)
            + "\n\nCounter-evidence:\n"
            + ("\n".join(f"- {item}" for item in weaknesses) or "- none recorded")
            + "\n\nAlternatives:\n"
            + (
                "\n".join(
                    f"- best for {alt.priority}: {alt.choice}"
                    for alt in alternatives
                )
                or "- none identified"
            )
            + "\n\nUnresolved conflicts:\n"
            + ("\n".join(f"- {item}" for item in conflicts) or "- none")
        )

        try:
            text = await asyncio.wait_for(
                self._synthesize(system, prompt),  # type: ignore[misc]
                timeout=_SYNTHESIS_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            self._logger.warning(
                "Executive summary synthesis failed (%s); assembling it "
                "deterministically instead",
                exc,
            )
            return ""

        cleaned = _strip_headings(text)
        # A summary far below budget means the model produced a stub;
        # the deterministic assembler is better than a stub.
        return cleaned if len(cleaned.split()) >= 120 else ""

    def _assembled_summary(
        self,
        plan: ResearchPlan,
        headline: str,
        label: str,
        confidence: float,
        points: list[str],
        weaknesses: list[str],
        alternatives: list[Alternative],
        supporters: list[str],
        dissenters: list[str],
        conflicts: list[str],
        evidence: EvidenceSet,
    ) -> str:
        """
        Write the report from evidence without inference.

        Args:
            plan: The executed plan.
            headline: Chosen recommendation.
            label: What the headline answers.
            confidence: Computed confidence.
            points: Supporting claims.
            weaknesses: Counter-arguments.
            alternatives: Runner-ups.
            supporters: Providers backing the headline.
            dissenters: Providers that did not.
            conflicts: Unresolved contradictions.
            evidence: Gathered evidence.

        Returns:
            Report prose. Every sentence is backed by supplied evidence.
        """
        paragraphs: list[str] = []

        opener = (
            f"{label}: {headline}. Confidence is {confidence:.0%}, based on "
            f"{len(evidence.items)} finding(s) gathered across "
            f"{len(evidence.providers)} source(s) over "
            f"{len(plan.investigation_tasks)} investigation task(s)."
        )
        if supporters:
            opener += f" Corroborated by {', '.join(supporters)}."
        paragraphs.append(opener)

        if points:
            paragraphs.append(
                "The evidence behind this is direct. "
                + " ".join(_as_sentence(point) for point in points[:4])
            )

        if weaknesses:
            paragraphs.append(
                "There are real reasons for caution. "
                + " ".join(_as_sentence(item) for item in weaknesses[:3])
            )
        else:
            paragraphs.append(
                "No provider raised a substantive objection to this "
                "conclusion, which is itself weak evidence: absence of "
                "recorded objection is not the same as verified absence "
                "of a problem."
            )

        if alternatives:
            paragraphs.append(
                "Under different priorities the answer changes. "
                + " ".join(
                    f"For {alt.priority}, {alt.choice} is the better "
                    f"choice{(' — ' + alt.rationale) if alt.rationale else ''}."
                    for alt in alternatives
                )
            )

        if conflicts:
            paragraphs.append(
                "The sources did not fully agree. "
                + " ".join(_as_sentence(conflict) for conflict in conflicts[:3])
                + " Treat those specific points as unsettled."
            )
        elif dissenters:
            paragraphs.append(
                f"{', '.join(dissenters)} contributed evidence that did "
                "not bear on the headline recommendation, so their input "
                "neither supports nor contradicts it."
            )

        paragraphs.append(
            "The raw provider answers are available in the Evidence "
            "section below; every claim above can be traced back to them."
        )
        return "\n\n".join(paragraphs)


def _priority_for(product: str, evidence: EvidenceSet) -> str:
    """
    Infer what an alternative is best for from the evidence about it.

    Args:
        product: Alternative's name.
        evidence: Gathered evidence.

    Returns:
        A short priority label.
    """
    lowered_name = product.lower()
    mentions = " ".join(
        item.fact.lower()
        for item in evidence.items
        if lowered_name in item.fact.lower()
    )
    for keyword, label in _PRIORITY_KEYWORDS:
        if keyword in mentions:
            return label
    return "an alternative view"


_PRIORITY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("camera", "photography"),
    ("photo", "photography"),
    ("game", "gaming"),
    ("fps", "gaming"),
    ("battery", "battery life"),
    ("endurance", "battery life"),
    ("compact", "a smaller size"),
    ("price", "value"),
    ("cheap", "value"),
    ("budget", "value"),
    ("performance", "raw performance"),
    ("benchmark", "raw performance"),
    ("display", "screen quality"),
    ("support", "long-term support"),
)


def _headline_from_fact(fact: str) -> str:
    """
    Compress a claim into a headline-length phrase.

    Args:
        fact: Claim text.

    Returns:
        Headline phrase.
    """
    first = re.split(r"(?<=[.!?])\s", fact.strip())[0]
    words = first.split()
    if len(words) <= 14:
        return first.rstrip(".")
    return " ".join(words[:14]) + "…"


def _as_sentence(text: str) -> str:
    """
    Ensure a fragment reads as a sentence.

    Args:
        text: Fragment.

    Returns:
        Capitalized, terminated sentence.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned if cleaned[-1] in ".!?" else cleaned + "."


def _strip_headings(text: str) -> str:
    """
    Remove markdown headings and bullets from model prose.

    Args:
        text: Model output.

    Returns:
        Prose without structural markup.
    """
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        stripped = re.sub(r"^[-*•]\s+", "", stripped)
        lines.append(stripped)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _enforce_word_budget(summary: str) -> str:
    """
    Trim a summary that exceeds the maximum length.

    Trimming happens at a sentence boundary so the report never ends
    mid-clause. Short summaries are left alone: padding prose to hit a
    floor adds words, not information.

    Args:
        summary: Report prose.

    Returns:
        Prose within the word budget.
    """
    words = summary.split()
    if len(words) <= _MAX_WORDS:
        return summary

    truncated = " ".join(words[:_MAX_WORDS])
    boundary = max(
        truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?")
    )
    if boundary > 0:
        return truncated[: boundary + 1]
    return truncated + "…"


__all__ = [
    "Alternative",
    "ExecutiveReport",
    "ExecutiveReportBuilder",
]
