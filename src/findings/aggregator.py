"""
Cross-provider consensus over structured recommendations.

The existing consensus engine compares answers as text, which measures
whether providers *sound* alike. This module measures something stricter
and more useful: whether providers recommended the same concrete things.

Duplicate products are merged on a normalized key, supporters are counted,
per-product confidence combines agreement with the providers' own
assertiveness, and the two interesting asymmetries are surfaced
explicitly — products only one provider named, and products most
providers named but some omitted. Those asymmetries are exactly what the
contradiction engine turns into follow-up questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Optional

from .models import Citation, Price, ProviderFindings, Recommendation


@dataclass(frozen=True, slots=True)
class ProductConsensus:
    """
    Merged view of one recommendation across all providers.

    Attributes:
        key: Normalized identity used for merging.
        display_name: Most common spelling among supporters.
        supporters: Providers that recommended it, in first-seen order.
        dissenters: Providers that answered but did not recommend it.
        confidence: Combined confidence in this recommendation.
        mean_rank: Average position among providers that ranked it.
        prices: Prices reported for it, by provider.
        citations: Citations attached to it by any provider.
        reasons: Reasoning given for it, by provider.
    """

    key: str
    display_name: str
    supporters: tuple[str, ...]
    dissenters: tuple[str, ...]
    confidence: float
    mean_rank: Optional[float] = None
    prices: tuple[tuple[str, Price], ...] = ()
    citations: tuple[Citation, ...] = ()
    reasons: tuple[tuple[str, str], ...] = ()

    @property
    def support_count(self) -> int:
        """
        Return how many providers recommended this product.

        Returns:
            Number of supporting providers.
        """
        return len(self.supporters)

    @property
    def is_unique(self) -> bool:
        """
        Return whether exactly one provider recommended this product.

        Returns:
            True when only one provider named it.
        """
        return self.support_count == 1

    @property
    def is_unanimous(self) -> bool:
        """
        Return whether every answering provider recommended it.

        Returns:
            True when no provider omitted it.
        """
        return not self.dissenters

    @property
    def price_spread(self) -> Optional[float]:
        """
        Return the difference between the highest and lowest price quoted.

        Returns:
            Absolute spread, or None when fewer than two prices exist.
        """
        amounts = [price.amount for _, price in self.prices]
        if len(amounts) < 2:
            return None
        return max(amounts) - min(amounts)

    @property
    def price_disagreement_ratio(self) -> Optional[float]:
        """
        Return the price spread as a fraction of the lowest price.

        Expressed as a ratio so a threshold is meaningful regardless of
        the product's absolute cost.

        Returns:
            Ratio, or None when fewer than two prices exist or the lowest
            price is zero.
        """
        amounts = [price.amount for _, price in self.prices]
        if len(amounts) < 2:
            return None
        low = min(amounts)
        if low <= 0:
            return None
        return (max(amounts) - low) / low


@dataclass(frozen=True, slots=True)
class ConsensusAggregate:
    """
    Complete cross-provider picture for one research question.

    Attributes:
        question: Question the findings answer.
        findings: Per-provider structured findings.
        products: Merged products, most-supported first.
        answering_providers: Providers that returned a usable answer.
        silent_providers: Providers that answered but recommended nothing.
        confidence: Overall confidence across the merged picture.
    """

    question: str
    findings: tuple[ProviderFindings, ...]
    products: tuple[ProductConsensus, ...]
    answering_providers: tuple[str, ...]
    silent_providers: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def agreed(self) -> tuple[ProductConsensus, ...]:
        """
        Return products recommended by more than one provider.

        Returns:
            Products with at least two supporters.
        """
        return tuple(p for p in self.products if p.support_count > 1)

    @property
    def unique(self) -> tuple[ProductConsensus, ...]:
        """
        Return products only one provider recommended.

        Returns:
            Products with exactly one supporter.
        """
        return tuple(p for p in self.products if p.is_unique)

    @property
    def unanimous(self) -> tuple[ProductConsensus, ...]:
        """
        Return products every answering provider recommended.

        Returns:
            Unanimously recommended products.
        """
        return tuple(
            p
            for p in self.products
            if p.is_unanimous and len(self.answering_providers) > 1
        )

    @property
    def contested(self) -> tuple[ProductConsensus, ...]:
        """
        Return products with meaningful support and meaningful omission.

        A product supported by several providers and omitted by others is
        the clearest signal of genuine disagreement.

        Returns:
            Contested products, most-supported first.
        """
        return tuple(
            p
            for p in self.products
            if p.support_count >= 2 and p.dissenters
        )

    @property
    def has_disagreement(self) -> bool:
        """
        Return whether any disagreement was detected.

        Returns:
            True when unique or contested products exist.
        """
        return bool(self.unique or self.contested)

    def top(self, limit: int = 5) -> tuple[ProductConsensus, ...]:
        """
        Return the most-supported products.

        Args:
            limit: Maximum products to return.

        Returns:
            Highest-ranked products.
        """
        return self.products[:limit]

    def describe(self) -> str:
        """
        Render a one-line summary for logs.

        Returns:
            Human-readable description.
        """
        return (
            f"{len(self.products)} product(s) across "
            f"{len(self.answering_providers)} provider(s); "
            f"{len(self.agreed)} agreed, {len(self.unique)} unique, "
            f"{len(self.contested)} contested; "
            f"confidence {self.confidence:.2f}"
        )


def _most_common_spelling(names: list[str]) -> str:
    """
    Choose the display spelling for a merged product.

    Ties break toward the longer name, which is usually the more specific
    one ("Skechers Go Walk" over "Skechers").

    Args:
        names: Spellings used by supporters.

    Returns:
        Preferred display name.
    """
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=lambda name: (counts[name], len(name)))


def aggregate_findings(
    question: str,
    findings: list[ProviderFindings],
) -> ConsensusAggregate:
    """
    Merge per-provider findings into a cross-provider consensus.

    Args:
        question: Question the findings answer.
        findings: Structured findings, one per answering provider.

    Returns:
        Consensus aggregate. Providers that answered but recommended
        nothing are recorded separately rather than being treated as
        dissenters, since silence is not the same as disagreement.
    """
    answering = tuple(finding.provider for finding in findings)
    silent = tuple(
        finding.provider for finding in findings if not finding.recommendations
    )
    recommenders = [
        finding for finding in findings if finding.recommendations
    ]

    grouped: dict[str, list[Recommendation]] = {}
    for finding in recommenders:
        for recommendation in finding.recommendations:
            grouped.setdefault(recommendation.key, []).append(recommendation)

    recommender_names = {finding.provider for finding in recommenders}
    products: list[ProductConsensus] = []

    for key, group in grouped.items():
        supporters = tuple(
            dict.fromkeys(item.provider for item in group)
        )
        dissenters = tuple(
            sorted(recommender_names - set(supporters))
        )
        ranks = [item.rank for item in group if item.rank is not None]
        prices = tuple(
            (item.provider, item.price)
            for item in group
            if item.price is not None
        )
        citations: list[Citation] = []
        seen_urls: set[str] = set()
        for item in group:
            for citation in item.citations:
                if citation.url not in seen_urls:
                    seen_urls.add(citation.url)
                    citations.append(citation)

        reasons = tuple(
            (item.provider, item.reasoning)
            for item in group
            if item.reasoning
        )

        # Confidence combines how many providers agreed with how
        # confidently each of them phrased it. Agreement dominates:
        # three hedged agreements beat one assertive outlier.
        support_ratio = (
            len(supporters) / len(recommender_names)
            if recommender_names
            else 0.0
        )
        stated = fmean(item.confidence for item in group)
        confidence = round(
            min(0.99, 0.65 * support_ratio + 0.35 * stated), 4
        )

        products.append(
            ProductConsensus(
                key=key,
                display_name=_most_common_spelling(
                    [item.name for item in group]
                ),
                supporters=supporters,
                dissenters=dissenters,
                confidence=confidence,
                mean_rank=round(fmean(ranks), 2) if ranks else None,
                prices=prices,
                citations=tuple(citations),
                reasons=reasons,
            )
        )

    products.sort(
        key=lambda product: (
            -product.support_count,
            -product.confidence,
            product.mean_rank if product.mean_rank is not None else 99.0,
        )
    )

    overall = (
        round(fmean(product.confidence for product in products), 4)
        if products
        else 0.0
    )

    return ConsensusAggregate(
        question=question,
        findings=tuple(findings),
        products=tuple(products),
        answering_providers=answering,
        silent_providers=silent,
        confidence=overall,
    )


__all__ = [
    "ConsensusAggregate",
    "ProductConsensus",
    "aggregate_findings",
]
