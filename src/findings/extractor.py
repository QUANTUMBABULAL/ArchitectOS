"""
Deterministic extraction of structured findings from provider answers.

Providers return Markdown prose. This module converts that prose into
:class:`ProviderFindings` without calling a model, for three reasons: it
runs in microseconds rather than seconds, it is testable, and it cannot
hallucinate a recommendation that the provider never made.

The parser handles the shapes AI chat answers actually take:

* numbered lists (``1. Product — reason``)
* bulleted lists (``- **Product**: reason``)
* bold headings (``**Product**`` followed by a paragraph)
* inline recommendations (``I recommend Product because ...``)

Extraction is conservative. When a line cannot be confidently read as a
recommendation it is treated as reasoning rather than invented into a
product, and ``parse_failed`` records when nothing structured was found.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Pattern

from .models import Citation, Price, ProviderFindings, Recommendation

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_MARKDOWN_LINK: Pattern[str] = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BARE_URL: Pattern[str] = re.compile(r"(?<!\()\bhttps?://[^\s<>\]\)]+")

_CURRENCY_SYMBOLS = "$£€¥₹"

# Amounts are either comma-grouped (1,299.99) or a plain digit run
# (1500). The plain-run alternative is required: without it any price of
# four or more digits written without separators is silently missed.
_AMOUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"

_PRICE_SYMBOL: Pattern[str] = re.compile(
    rf"([{_CURRENCY_SYMBOLS}])\s?({_AMOUNT})"
)
_PRICE_CODE: Pattern[str] = re.compile(
    rf"\b({_AMOUNT})\s?(USD|EUR|GBP|INR|JPY)\b",
    re.IGNORECASE,
)

_NUMBERED_ITEM: Pattern[str] = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*)$")
_BULLET_ITEM: Pattern[str] = re.compile(r"^\s*[-*•]\s+(.*)$")
_HEADING: Pattern[str] = re.compile(r"^\s*#{1,6}\s+(.*)$")
_BOLD_LEAD: Pattern[str] = re.compile(r"^\*\*(.+?)\*\*\s*[:\-–—]?\s*(.*)$")
_INLINE_RECOMMEND: Pattern[str] = re.compile(
    r"\b(?:i\s+)?(?:recommend|suggest|would\s+go\s+with|best\s+choice\s+is|"
    r"top\s+pick\s+is)\b[:\s]+(.{3,80}?)(?:[.;]|\s+because\b|$)",
    re.IGNORECASE,
)

# Splits a recommendation line into name and explanation.
_NAME_SEPARATOR: Pattern[str] = re.compile(r"\s+[—–\-:]\s+|\s*[:—–]\s*")

_HEDGE_TERMS: tuple[str, ...] = (
    "might", "may", "possibly", "perhaps", "unclear", "not sure",
    "uncertain", "could be", "hard to say", "depends", "i think",
    "probably", "seems", "appears", "roughly", "approximately",
)
_ASSERTIVE_TERMS: tuple[str, ...] = (
    "definitely", "certainly", "clearly", "without doubt", "strongly",
    "best", "proven", "consistently", "always", "the answer is",
)
_EVIDENCE_TERMS: tuple[str, ...] = (
    "according to", "study", "review", "tested", "benchmark", "data",
    "research", "report", "survey", "measured", "documented", "source",
)

# Lines that introduce a list rather than being a recommendation.
_PREAMBLE_TERMS: tuple[str, ...] = (
    "here are", "here's", "based on", "options include", "consider",
    "in summary", "to summarize", "overall", "note that", "however",
)

_MAX_NAME_WORDS = 9


def _strip_markdown(text: str) -> str:
    """
    Remove inline Markdown emphasis and link syntax from text.

    Args:
        text: Raw Markdown text.

    Returns:
        Plain text with emphasis markers and link targets removed.
    """
    without_links = _MARKDOWN_LINK.sub(r"\1", text)
    return (
        without_links.replace("**", "")
        .replace("__", "")
        .replace("`", "")
        .strip()
    )


def extract_citations(text: str) -> tuple[Citation, ...]:
    """
    Extract citations from an answer.

    Markdown links contribute their title; bare URLs are included without
    one. Duplicate URLs collapse to a single citation.

    Args:
        text: Answer text.

    Returns:
        Citations in first-seen order.
    """
    citations: list[Citation] = []
    seen: set[str] = set()

    for title, url in _MARKDOWN_LINK.findall(text):
        if url not in seen:
            seen.add(url)
            citations.append(Citation(url=url, title=title.strip() or None))

    for url in _BARE_URL.findall(text):
        trimmed = url.rstrip(".,;:")
        if trimmed not in seen:
            seen.add(trimmed)
            citations.append(Citation(url=trimmed))

    return tuple(citations)


def extract_price(text: str) -> Optional[Price]:
    """
    Extract the first price mentioned in a fragment.

    Args:
        text: Text fragment.

    Returns:
        Parsed price, or None when no price is present.
    """
    symbol_match = _PRICE_SYMBOL.search(text)
    if symbol_match:
        currency, amount = symbol_match.groups()
        return _build_price(amount, currency, symbol_match.group(0))

    code_match = _PRICE_CODE.search(text)
    if code_match:
        amount, currency = code_match.groups()
        return _build_price(
            amount, currency.upper(), code_match.group(0)
        )
    return None


def _build_price(amount: str, currency: str, raw: str) -> Optional[Price]:
    """
    Build a price from matched components.

    Args:
        amount: Numeric text, possibly containing thousands separators.
        currency: Currency symbol or code.
        raw: Original matched text.

    Returns:
        Price, or None when the amount cannot be parsed.
    """
    try:
        value = float(amount.replace(",", ""))
    except ValueError:
        return None
    return Price(amount=value, currency=currency, raw=raw.strip())


def score_confidence(text: str) -> float:
    """
    Estimate confidence from the assertiveness of the language used.

    This is a linguistic heuristic, not a claim the provider made. Hedging
    lowers the score and assertive phrasing raises it, starting from a
    neutral midpoint.

    Args:
        text: Text to score.

    Returns:
        Confidence between 0.05 and 0.95.
    """
    lowered = text.lower()
    hedges = sum(1 for term in _HEDGE_TERMS if term in lowered)
    assertions = sum(1 for term in _ASSERTIVE_TERMS if term in lowered)
    evidence = sum(1 for term in _EVIDENCE_TERMS if term in lowered)

    score = 0.5 - 0.08 * hedges + 0.07 * assertions + 0.04 * evidence
    return max(0.05, min(0.95, score))


def _is_preamble(text: str) -> bool:
    """
    Report whether a line introduces a list rather than naming an item.

    Args:
        text: Candidate line.

    Returns:
        True when the line looks like framing text.
    """
    lowered = text.lower().strip()
    return any(lowered.startswith(term) for term in _PREAMBLE_TERMS)


def _split_name_and_reason(text: str) -> tuple[str, str]:
    """
    Split a recommendation line into a name and an explanation.

    Args:
        text: Line text with Markdown already stripped.

    Returns:
        Tuple of (name, reasoning). Reasoning is empty when the line
        contained only a name.
    """
    bold = _BOLD_LEAD.match(text.strip())
    if bold:
        return bold.group(1).strip(), bold.group(2).strip()

    parts = _NAME_SEPARATOR.split(text, maxsplit=1)
    if len(parts) == 2 and parts[0].strip():
        return parts[0].strip(), parts[1].strip()

    sentence_end = text.find(". ")
    if 0 < sentence_end <= 80:
        return text[:sentence_end].strip(), text[sentence_end + 1 :].strip()

    return text.strip(), ""


def _plausible_name(name: str) -> bool:
    """
    Report whether a fragment is plausibly a product or option name.

    Args:
        name: Candidate name.

    Returns:
        True when the fragment is short enough and not obviously prose.
    """
    stripped = name.strip()
    if not stripped or len(stripped) < 2:
        return False
    if len(stripped.split()) > _MAX_NAME_WORDS:
        return False
    if _is_preamble(stripped):
        return False
    return True


def _candidate_lines(answer: str) -> Iterable[tuple[Optional[int], str]]:
    """
    Yield lines that may contain a recommendation.

    Args:
        answer: Full answer text.

    Yields:
        Tuples of (rank, line text). Rank is set only for numbered items.
    """
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        numbered = _NUMBERED_ITEM.match(stripped)
        if numbered:
            yield int(numbered.group(1)), numbered.group(2)
            continue

        bullet = _BULLET_ITEM.match(stripped)
        if bullet:
            yield None, bullet.group(1)
            continue

        heading = _HEADING.match(stripped)
        if heading:
            yield None, heading.group(1)
            continue

        if stripped.startswith("**") and stripped.count("**") >= 2:
            yield None, stripped


def extract_findings(provider: str, answer: str) -> ProviderFindings:
    """
    Parse one provider's answer into structured findings.

    Args:
        provider: Provider name.
        answer: Answer text as returned by the provider.

    Returns:
        Structured findings. ``parse_failed`` is True when no
        recommendation could be identified, which is information rather
        than an error.
    """
    text = (answer or "").strip()
    if not text:
        return ProviderFindings(
            provider=provider,
            reasoning="",
            confidence=0.05,
            answer_chars=0,
            parse_failed=True,
        )

    all_citations = extract_citations(text)
    recommendations: list[Recommendation] = []
    seen_keys: set[str] = set()

    for rank, raw_line in _candidate_lines(text):
        plain = _strip_markdown(raw_line)
        if not plain or _is_preamble(plain):
            continue

        name, reasoning = _split_name_and_reason(plain)
        if not _plausible_name(name):
            continue

        candidate = Recommendation(
            name=name,
            provider=provider,
            reasoning=reasoning or None,
            price=extract_price(raw_line),
            citations=extract_citations(raw_line),
            confidence=score_confidence(raw_line),
            rank=rank,
            source_text=raw_line.strip(),
        )
        if candidate.key and candidate.key not in seen_keys:
            seen_keys.add(candidate.key)
            recommendations.append(candidate)

    # Inline recommendations catch answers written as prose with no list.
    if not recommendations:
        for match in _INLINE_RECOMMEND.finditer(text):
            name = _strip_markdown(match.group(1)).strip(" .,:;")
            if not _plausible_name(name):
                continue
            candidate = Recommendation(
                name=name,
                provider=provider,
                reasoning=None,
                price=extract_price(text),
                citations=(),
                confidence=score_confidence(text),
                source_text=match.group(0).strip(),
            )
            if candidate.key and candidate.key not in seen_keys:
                seen_keys.add(candidate.key)
                recommendations.append(candidate)

    evidence = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and any(term in line.lower() for term in _EVIDENCE_TERMS)
    )

    return ProviderFindings(
        provider=provider,
        recommendations=tuple(recommendations),
        reasoning=text,
        evidence=evidence,
        citations=all_citations,
        confidence=score_confidence(text),
        answer_chars=len(text),
        parse_failed=not recommendations,
    )


__all__ = [
    "extract_citations",
    "extract_findings",
    "extract_price",
    "score_confidence",
]
