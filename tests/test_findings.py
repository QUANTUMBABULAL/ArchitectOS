"""
Tests for structured finding extraction and cross-provider merging.

Extraction runs on untrusted provider prose and must be conservative: it
is better to extract nothing than to invent a recommendation a provider
never made. The merging tests pin the two asymmetries the contradiction
engine depends on — unique recommendations and contested omissions.
"""

from __future__ import annotations

import pytest

from src.findings import (
    aggregate_findings,
    extract_citations,
    extract_findings,
    extract_price,
    normalize_name,
    score_confidence,
)

NUMBERED_ANSWER = """
Here are my top picks for walking shoes:

1. **Brooks Ghost 15** — excellent cushioning, $140 on the Brooks site.
2. **Hoka Bondi 8**: maximum cushion, roughly $165.
3. **New Balance 990v6** — durable, according to a Runner's World review.

Overall I'd start with the Brooks.
"""

BULLETED_ANSWER = """
- **Hoka Bondi 8**: very soft ride, $160
- **Brooks Ghost 15** — reliable daily trainer
- **Skechers Go Walk 6** — budget option at $75
"""

PROSE_ANSWER = "I recommend the Hoka Bondi 8 because it has the most cushion."


class TestCitations:
    """Citation extraction."""

    def test_extracts_markdown_links(self) -> None:
        """Markdown links contribute a title and a URL."""
        citations = extract_citations(
            "See [Runner's World](https://runnersworld.com/review) for detail."
        )
        assert len(citations) == 1
        assert citations[0].url == "https://runnersworld.com/review"
        assert citations[0].title == "Runner's World"

    def test_extracts_bare_urls(self) -> None:
        """Bare URLs are captured without a title."""
        citations = extract_citations("Source: https://example.com/page")
        assert citations[0].url == "https://example.com/page"
        assert citations[0].title is None

    def test_deduplicates(self) -> None:
        """A URL repeated in both forms yields one citation."""
        text = (
            "[A](https://example.com/x) and again https://example.com/x"
        )
        assert len(extract_citations(text)) == 1

    def test_domain_strips_www(self) -> None:
        """Domains normalize for grouping."""
        citations = extract_citations("https://www.Example.com/a/b?c=1")
        assert citations[0].domain == "example.com"

    def test_trailing_punctuation_trimmed(self) -> None:
        """A sentence-final URL does not keep its full stop."""
        citations = extract_citations("See https://example.com/page.")
        assert citations[0].url == "https://example.com/page"


class TestPrice:
    """Price extraction."""

    @pytest.mark.parametrize(
        ("text", "amount", "currency"),
        [
            ("costs $140", 140.0, "$"),
            ("about $1,299.99 total", 1299.99, "$"),
            ("priced at £89", 89.0, "£"),
            ("€250 in the EU", 250.0, "€"),
            ("1500 USD", 1500.0, "USD"),
            # Four or more digits without a separator: regression guard,
            # an earlier pattern silently missed these.
            ("$1500", 1500.0, "$"),
            ("costs $85.50", 85.5, "$"),
        ],
    )
    def test_extracts(
        self,
        text: str,
        amount: float,
        currency: str,
    ) -> None:
        """Symbols and currency codes are both understood."""
        price = extract_price(text)
        assert price is not None
        assert price.amount == amount
        assert price.currency == currency

    def test_returns_none_without_price(self) -> None:
        """Absence of a price is not an error."""
        assert extract_price("no numbers here") is None

    def test_render_drops_trailing_zeros(self) -> None:
        """Whole amounts render without decimals."""
        price = extract_price("$140")
        assert price is not None
        assert price.render() == "$140"


class TestConfidenceScoring:
    """Linguistic confidence heuristic."""

    def test_hedging_lowers_confidence(self) -> None:
        """Hedged language scores below neutral."""
        hedged = score_confidence("It might possibly work, I'm not sure")
        assert hedged < 0.5

    def test_assertive_raises_confidence(self) -> None:
        """Assertive language scores above neutral."""
        assert score_confidence("This is definitely the best choice") > 0.5

    def test_stays_in_range(self) -> None:
        """The score is always a valid probability-like value."""
        extreme = score_confidence("might maybe perhaps unclear " * 20)
        assert 0.05 <= extreme <= 0.95


class TestRecommendationExtraction:
    """Parsing recommendations out of answers."""

    def test_numbered_list(self) -> None:
        """Numbered items become ranked recommendations."""
        findings = extract_findings("chatgpt", NUMBERED_ANSWER)
        names = [r.name for r in findings.recommendations]

        assert "Brooks Ghost 15" in names
        assert "Hoka Bondi 8" in names
        assert findings.recommendations[0].rank == 1
        assert findings.parse_failed is False

    def test_bulleted_list(self) -> None:
        """Bulleted items are extracted with their prices."""
        findings = extract_findings("claude", BULLETED_ANSWER)
        by_name = {r.name: r for r in findings.recommendations}

        assert "Skechers Go Walk 6" in by_name
        price = by_name["Skechers Go Walk 6"].price
        assert price is not None
        assert price.amount == 75.0

    def test_prose_fallback(self) -> None:
        """An answer with no list still yields a recommendation."""
        findings = extract_findings("grok", PROSE_ANSWER)
        assert findings.recommendations
        assert "Hoka" in findings.recommendations[0].name

    def test_preamble_is_not_a_recommendation(self) -> None:
        """Framing lines are excluded from products."""
        findings = extract_findings("chatgpt", NUMBERED_ANSWER)
        names = [r.name.lower() for r in findings.recommendations]
        assert not any(name.startswith("here are") for name in names)

    def test_reasoning_captured(self) -> None:
        """Explanatory text is retained alongside the name."""
        findings = extract_findings("chatgpt", NUMBERED_ANSWER)
        first = findings.recommendations[0]
        assert first.reasoning
        assert "cushioning" in first.reasoning.lower()

    def test_evidence_lines_collected(self) -> None:
        """Lines citing sources are recorded as evidence."""
        findings = extract_findings("chatgpt", NUMBERED_ANSWER)
        assert any("according to" in item.lower() for item in findings.evidence)

    def test_empty_answer_marks_parse_failure(self) -> None:
        """An empty answer is reported, not guessed at."""
        findings = extract_findings("gemini", "")
        assert findings.parse_failed is True
        assert findings.recommendations == ()

    def test_unstructured_answer_marks_parse_failure(self) -> None:
        """Prose with no recommendation yields no invented products."""
        findings = extract_findings(
            "gemini", "That depends on many personal factors."
        )
        assert findings.parse_failed is True

    def test_source_text_retained(self) -> None:
        """Each recommendation keeps the line it was parsed from."""
        findings = extract_findings("chatgpt", NUMBERED_ANSWER)
        assert all(r.source_text for r in findings.recommendations)


class TestNormalization:
    """Name normalization used for merging."""

    def test_case_and_punctuation_collapse(self) -> None:
        """Spelling variants collapse to one key."""
        assert normalize_name("Skechers Go-Walk!") == normalize_name(
            "skechers go walk"
        )

    def test_filler_words_removed(self) -> None:
        """Filler tokens do not prevent a match."""
        assert normalize_name("The Brooks Ghost Series") == normalize_name(
            "Brooks Ghost"
        )

    def test_distinct_products_stay_distinct(self) -> None:
        """Normalization is not so aggressive that products merge wrongly."""
        assert normalize_name("Hoka Bondi 8") != normalize_name(
            "Hoka Clifton 9"
        )


class TestAggregation:
    """Cross-provider merging."""

    @staticmethod
    def _three_providers():
        """Build findings where providers partly agree."""
        return [
            extract_findings("chatgpt", NUMBERED_ANSWER),
            extract_findings("claude", BULLETED_ANSWER),
            extract_findings("grok", PROSE_ANSWER),
        ]

    def test_merges_duplicate_products(self) -> None:
        """The same product from two providers becomes one entry."""
        aggregate = aggregate_findings("best shoes?", self._three_providers())
        hoka = next(
            p for p in aggregate.products if "hoka" in p.key
        )
        assert set(hoka.supporters) >= {"chatgpt", "claude"}
        assert hoka.support_count >= 2

    def test_counts_supporters_and_dissenters(self) -> None:
        """Providers that omitted a product are recorded as dissenters."""
        aggregate = aggregate_findings("best shoes?", self._three_providers())
        skechers = next(
            p for p in aggregate.products if "skechers" in p.key
        )
        assert skechers.supporters == ("claude",)
        assert "chatgpt" in skechers.dissenters

    def test_identifies_unique_recommendations(self) -> None:
        """Products named by exactly one provider are flagged unique."""
        aggregate = aggregate_findings("best shoes?", self._three_providers())
        unique_keys = {p.key for p in aggregate.unique}
        assert any("skechers" in key for key in unique_keys)

    def test_identifies_contested_products(self) -> None:
        """Products with support and omission are contested."""
        aggregate = aggregate_findings("best shoes?", self._three_providers())
        assert aggregate.has_disagreement is True

    def test_sorted_by_support(self) -> None:
        """The most corroborated product ranks first."""
        aggregate = aggregate_findings("best shoes?", self._three_providers())
        counts = [p.support_count for p in aggregate.products]
        assert counts == sorted(counts, reverse=True)

    def test_confidence_rewards_agreement(self) -> None:
        """A corroborated product outranks a lone assertive one."""
        aggregate = aggregate_findings("best shoes?", self._three_providers())
        agreed = next(p for p in aggregate.products if p.support_count >= 2)
        unique = next(p for p in aggregate.products if p.is_unique)
        assert agreed.confidence > unique.confidence

    def test_silent_provider_recorded_separately(self) -> None:
        """
        A provider that answered without recommending anything is silent,
        not dissenting — silence is not disagreement.
        """
        findings = [
            extract_findings("chatgpt", NUMBERED_ANSWER),
            extract_findings("gemini", "It really depends on your gait."),
        ]
        aggregate = aggregate_findings("best shoes?", findings)
        assert "gemini" in aggregate.silent_providers
        for product in aggregate.products:
            assert "gemini" not in product.dissenters

    def test_price_spread_detected(self) -> None:
        """Differing prices for one product produce a spread."""
        a = extract_findings("chatgpt", "1. **Widget** — costs $100")
        b = extract_findings("claude", "1. **Widget** — costs $200")
        aggregate = aggregate_findings("widget?", [a, b])
        widget = aggregate.products[0]

        assert widget.price_spread == pytest.approx(100.0)
        assert widget.price_disagreement_ratio == pytest.approx(1.0)

    def test_empty_findings_yield_empty_aggregate(self) -> None:
        """No findings is handled without raising."""
        aggregate = aggregate_findings("q", [])
        assert aggregate.products == ()
        assert aggregate.confidence == 0.0
        assert aggregate.has_disagreement is False
