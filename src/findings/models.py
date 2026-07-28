"""
Structured representation of what a provider actually said.

A provider answer arrives as prose. Comparing prose across providers only
supports lexical similarity, which cannot answer the questions that
matter: did two providers recommend the same product, at what price, and
on what evidence. These models are the structured target that makes
recommendation-level consensus possible.

Extraction is deliberately lossy and honest about it. Every recommendation
retains the source text it came from, so a downstream reader can always
check the parse rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class Citation:
    """
    A source referenced by a provider.

    Attributes:
        url: Referenced URL.
        title: Link text when the citation was a markdown link.
    """

    url: str
    title: Optional[str] = None

    @property
    def domain(self) -> str:
        """
        Return the citation's host, used for deduplication and grouping.

        Returns:
            Lower-cased host without a leading ``www.``, or an empty
            string when the URL cannot be parsed.
        """
        remainder = self.url.split("//", 1)[-1]
        host = remainder.split("/", 1)[0].split("?", 1)[0].lower()
        return host[4:] if host.startswith("www.") else host


@dataclass(frozen=True, slots=True)
class Price:
    """
    A monetary amount extracted from an answer.

    Attributes:
        amount: Numeric value.
        currency: ISO-like currency code or symbol as written.
        raw: Original text the price was parsed from.
    """

    amount: float
    currency: str
    raw: str

    def render(self) -> str:
        """
        Render the price for display.

        Returns:
            Formatted price string.
        """
        if self.amount.is_integer():
            return f"{self.currency}{self.amount:,.0f}"
        return f"{self.currency}{self.amount:,.2f}"


@dataclass(frozen=True, slots=True)
class Recommendation:
    """
    One concrete thing a provider recommended.

    Attributes:
        name: Recommended product, tool, or option.
        provider: Provider that made the recommendation.
        reasoning: Supporting explanation, when one was given.
        price: Extracted price, when one was given.
        citations: Sources attached to this recommendation.
        confidence: Confidence in the recommendation between 0.0 and 1.0,
            derived from hedging language rather than asserted by the
            provider.
        rank: One-based position in the provider's own ordering, when the
            answer was an ordered list.
        source_text: Original text this recommendation was parsed from.
    """

    name: str
    provider: str
    reasoning: Optional[str] = None
    price: Optional[Price] = None
    citations: tuple[Citation, ...] = ()
    confidence: float = 0.5
    rank: Optional[int] = None
    source_text: str = ""

    @property
    def key(self) -> str:
        """
        Return the normalized identity used to merge across providers.

        Casing, punctuation, and common filler words are removed so
        "Skechers Go Walk" and "skechers go-walk" collapse to one entry.

        Returns:
            Normalized comparison key.
        """
        return normalize_name(self.name)


@dataclass(frozen=True, slots=True)
class ProviderFindings:
    """
    Everything extracted from one provider's answer.

    Attributes:
        provider: Provider name.
        recommendations: Recommendations in the provider's own order.
        reasoning: Overall reasoning or summary text.
        evidence: Statements that carried supporting detail.
        citations: All citations found anywhere in the answer.
        confidence: Overall confidence derived from the answer's language.
        answer_chars: Length of the source answer.
        parse_failed: True when nothing structured could be extracted, so
            callers can distinguish "provider said nothing useful" from
            "extraction did not understand the format".
    """

    provider: str
    recommendations: tuple[Recommendation, ...] = ()
    reasoning: str = ""
    evidence: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    confidence: float = 0.5
    answer_chars: int = 0
    parse_failed: bool = False

    @property
    def recommended_keys(self) -> frozenset[str]:
        """
        Return normalized keys of everything this provider recommended.

        Returns:
            Set of normalized recommendation keys.
        """
        return frozenset(item.key for item in self.recommendations)

    @property
    def products(self) -> tuple[str, ...]:
        """
        Return recommendation names as written by the provider.

        Returns:
            Recommendation names in order.
        """
        return tuple(item.name for item in self.recommendations)

    def describe(self) -> str:
        """
        Render a one-line summary for logs.

        Returns:
            Human-readable description.
        """
        return (
            f"{self.provider}: {len(self.recommendations)} recommendation(s), "
            f"{len(self.citations)} citation(s), "
            f"confidence {self.confidence:.2f}"
        )


# Filler tokens dropped when normalizing a product name. Kept small on
# purpose: aggressive stripping merges genuinely different products.
_NAME_NOISE: frozenset[str] = frozenset(
    {"the", "a", "an", "series", "model", "edition", "version"}
)


def normalize_name(name: str) -> str:
    """
    Normalize a recommendation name for cross-provider comparison.

    Args:
        name: Name as written by a provider.

    Returns:
        Lower-cased key with punctuation and filler words removed.
    """
    cleaned = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in name.lower()
    )
    tokens = [
        token
        for token in cleaned.split()
        if token and token not in _NAME_NOISE
    ]
    return " ".join(tokens)


__all__ = [
    "Citation",
    "Price",
    "ProviderFindings",
    "Recommendation",
    "normalize_name",
]
