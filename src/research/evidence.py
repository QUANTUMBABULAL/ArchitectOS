"""
Structured evidence extracted from provider answers.

A provider replies with prose. Prose cannot be cross-checked: two
paragraphs that disagree look exactly like two paragraphs that agree.
This module converts each answer into discrete claims — a fact, where it
came from, how confident the provider was, its links — so the consensus
stage compares *claims* rather than *paragraphs*.

Extraction is deliberately two-tier. Providers are asked to answer in a
labelled block (``FACT:`` / ``SOURCE:`` / ``CONFIDENCE:`` / ``LINKS:``)
which parses exactly; when a provider ignores the format — and some
will — a salience-based fallback recovers the important sentences and
their citations rather than discarding the answer. Every item keeps the
text it came from, so a reader can always audit the parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from src.logger import get_logger

logger = get_logger(__name__)

_MAX_ITEMS_PER_PROVIDER = 12
_MIN_FACT_CHARS = 25
_MAX_FACT_CHARS = 400

_LINK_PATTERN = re.compile(r"\[([^\]]{1,120})\]\((https?://[^\s)]+)\)")
_BARE_URL_PATTERN = re.compile(r"(?<!\()\bhttps?://[^\s<>\])},;\"']+")
_FACT_LINE = re.compile(r"^\s*(?:[-*]\s*)?\**\s*FACT\s*\**\s*[:\-]\s*(.+)$", re.I)
_SOURCE_LINE = re.compile(r"^\s*(?:[-*]\s*)?\**\s*SOURCE\s*\**\s*[:\-]\s*(.+)$", re.I)
_CONFIDENCE_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?\**\s*CONFIDENCE\s*\**\s*[:\-]\s*(.+)$", re.I
)
_LINKS_LINE = re.compile(r"^\s*(?:[-*]\s*)?\**\s*LINKS?\s*\**\s*[:\-]\s*(.+)$", re.I)
_CAVEAT_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?\**\s*(?:CAVEAT|CONTRADICTION|CONFLICT)S?\s*\**\s*"
    r"[:\-]\s*(.+)$",
    re.I,
)

_CONFIDENCE_WORDS: dict[str, float] = {
    "certain": 0.95,
    "very high": 0.92,
    "high": 0.85,
    "strong": 0.85,
    "medium": 0.6,
    "moderate": 0.6,
    "mixed": 0.5,
    "low": 0.35,
    "weak": 0.3,
    "speculative": 0.25,
    "unverified": 0.25,
    "unknown": 0.4,
}

# Sentences that assert something checkable are worth keeping; hedging
# filler is not. These markers are what separates the two.
_SALIENCE_MARKERS = (
    "%", "₹", "$", "€", "£", "mah", "ghz", "gb", "mp", "fps", "hours",
    "score", "benchmark", "price", "costs", "rated", "ranked", "fastest",
    "best", "worst", "average", "measured", "tested", "launched",
    "released", "supports", "lacks", "compared",
)
_HEDGE_OPENERS = (
    "in conclusion", "overall", "to summarize", "in summary", "i hope",
    "let me know", "feel free", "as an ai", "here's", "here is", "sure,",
    "certainly", "of course",
)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """
    One checkable claim taken from a provider's answer.

    Attributes:
        provider: Provider that supplied the claim.
        task_id: Subtask the claim answers.
        task_title: Human-readable subtask label.
        fact: The claim itself, one sentence.
        source: Where the provider says the claim comes from.
        confidence: Provider-stated confidence in [0, 1].
        links: Supporting URLs.
        caveats: Contradictions or conditions the provider flagged.
        raw: The source text the claim was parsed from, for auditing.
    """

    provider: str
    task_id: int
    task_title: str
    fact: str
    source: str = ""
    confidence: float = 0.5
    links: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    raw: str = ""

    @property
    def key(self) -> str:
        """
        Return a normalized key for grouping equivalent claims.

        Returns:
            Lower-cased content words, order-independent.
        """
        words = sorted(
            word
            for word in re.findall(r"[a-z0-9₹$€£.%]+", self.fact.lower())
            if len(word) > 2 and word not in _GROUPING_STOPWORDS
        )
        return " ".join(words[:12])

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream and the UI.

        Returns:
            JSON-compatible payload.
        """
        return {
            "provider": self.provider,
            "taskId": self.task_id,
            "taskTitle": self.task_title,
            "fact": self.fact,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "links": list(self.links),
            "caveats": list(self.caveats),
        }


_GROUPING_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "has",
        "are", "was", "were", "its", "it's", "they", "their", "than",
        "then", "but", "not", "can", "may", "will", "would", "should",
        "also", "more", "most", "some", "very", "which", "while", "into",
        "about", "over", "under", "between", "because", "however",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """
    All evidence gathered for one research request.

    Attributes:
        items: Every extracted claim, in gathering order.
        raw_answers: Full provider answers keyed by ``provider · task``,
            preserved for the expandable Evidence section.
    """

    items: list[EvidenceItem] = field(default_factory=list)
    raw_answers: dict[str, str] = field(default_factory=dict)

    @property
    def providers(self) -> list[str]:
        """
        Return the providers that contributed evidence.

        Returns:
            Provider names in first-seen order.
        """
        seen: list[str] = []
        for item in self.items:
            if item.provider not in seen:
                seen.append(item.provider)
        return seen

    @property
    def all_links(self) -> list[str]:
        """
        Return every unique supporting link.

        Returns:
            URLs in first-seen order.
        """
        seen: list[str] = []
        for item in self.items:
            for link in item.links:
                if link not in seen:
                    seen.append(link)
        return seen

    def grouped(self) -> list[list[EvidenceItem]]:
        """
        Group equivalent claims across providers.

        Two claims group together when their normalized content words
        overlap substantially. Corroboration is exactly this: the same
        claim arriving from independent providers.

        Returns:
            Groups ordered by corroboration strength, strongest first.
        """
        groups: list[list[EvidenceItem]] = []
        for item in self.items:
            placed = False
            for group in groups:
                if _similar(item.key, group[0].key):
                    group.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])

        groups.sort(
            key=lambda group: (
                len({member.provider for member in group}),
                sum(member.confidence for member in group),
            ),
            reverse=True,
        )
        return groups

    def corroborated(self, minimum_providers: int = 2) -> list[list[EvidenceItem]]:
        """
        Return claim groups supported by several providers.

        Args:
            minimum_providers: How many distinct providers must agree.

        Returns:
            Qualifying groups, strongest first.
        """
        return [
            group
            for group in self.grouped()
            if len({member.provider for member in group}) >= minimum_providers
        ]

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize for the event stream and the UI.

        Returns:
            JSON-compatible payload.
        """
        return {
            "items": [item.to_payload() for item in self.items],
            "rawAnswers": dict(self.raw_answers),
            "providers": self.providers,
        }


class EvidenceExtractor:
    """
    Converts provider answers into structured evidence.

    Stateless and inference-free: extraction is parsing, not judgement,
    so it stays fast and cannot fail because a model is unavailable.
    """

    def __init__(self) -> None:
        """Initialize the extractor."""
        self._logger = get_logger(__name__)

    def instruction_block(self) -> str:
        """
        Return the answer-format instruction appended to every subtask.

        Returns:
            Instruction text asking for labelled, checkable claims.
        """
        return (
            "\n\nAnswer as a list of discrete findings, not as an essay. "
            "Use exactly this format for each finding, and give between "
            "3 and 6 findings:\n\n"
            "FACT: <one specific, checkable statement>\n"
            "SOURCE: <who or what establishes it>\n"
            "CONFIDENCE: <high | medium | low>\n"
            "LINKS: <supporting URLs, or 'none'>\n"
            "CAVEAT: <anything that contradicts or limits this, or "
            "'none'>\n\n"
            "Include concrete numbers wherever they exist. Do not add an "
            "introduction or a conclusion."
        )

    def extract(
        self,
        provider: str,
        task_id: int,
        task_title: str,
        answer: str,
    ) -> list[EvidenceItem]:
        """
        Extract evidence from one provider answer.

        Args:
            provider: Provider that produced the answer.
            task_id: Subtask identifier.
            task_title: Subtask label.
            answer: Full answer text.

        Returns:
            Extracted evidence, capped per provider.
        """
        text = (answer or "").strip()
        if not text:
            return []

        items = self._parse_labelled(provider, task_id, task_title, text)
        if items:
            return items[:_MAX_ITEMS_PER_PROVIDER]

        self._logger.debug(
            "%s did not use the evidence format for task %d; "
            "falling back to salience extraction",
            provider,
            task_id,
        )
        return self._parse_salient(provider, task_id, task_title, text)[
            :_MAX_ITEMS_PER_PROVIDER
        ]

    def _parse_labelled(
        self,
        provider: str,
        task_id: int,
        task_title: str,
        text: str,
    ) -> list[EvidenceItem]:
        """
        Parse answers that used the requested FACT/SOURCE format.

        Args:
            provider: Provider name.
            task_id: Subtask identifier.
            task_title: Subtask label.
            text: Answer text.

        Returns:
            Parsed evidence, empty when the format was not used.
        """
        items: list[EvidenceItem] = []
        current: Optional[dict[str, Any]] = None
        buffer: list[str] = []

        def flush() -> None:
            if current is None:
                return
            fact = str(current.get("fact", "")).strip()
            if len(fact) < 8:
                return
            raw = "\n".join(buffer)
            links = tuple(
                dict.fromkeys(
                    list(current.get("links", ())) + _harvest_links(raw)
                )
            )
            items.append(
                EvidenceItem(
                    provider=provider,
                    task_id=task_id,
                    task_title=task_title,
                    fact=_trim(fact),
                    source=str(current.get("source", "")).strip(),
                    confidence=float(current.get("confidence", 0.6)),
                    links=links[:6],
                    caveats=tuple(current.get("caveats", ())),
                    raw=raw,
                )
            )

        for line in text.splitlines():
            fact_match = _FACT_LINE.match(line)
            if fact_match:
                flush()
                current = {"fact": _strip_markup(fact_match.group(1))}
                buffer = [line]
                continue

            if current is None:
                continue
            buffer.append(line)

            source_match = _SOURCE_LINE.match(line)
            if source_match:
                current["source"] = _strip_markup(source_match.group(1))
                continue

            confidence_match = _CONFIDENCE_LINE.match(line)
            if confidence_match:
                current["confidence"] = _parse_confidence(
                    confidence_match.group(1)
                )
                continue

            links_match = _LINKS_LINE.match(line)
            if links_match:
                current["links"] = tuple(
                    _harvest_links(links_match.group(1))
                )
                continue

            caveat_match = _CAVEAT_LINE.match(line)
            if caveat_match:
                caveat = _strip_markup(caveat_match.group(1))
                if caveat.lower().strip(" .") not in {"none", "n/a", "-"}:
                    current["caveats"] = (caveat,)

        flush()
        return items

    def _parse_salient(
        self,
        provider: str,
        task_id: int,
        task_title: str,
        text: str,
    ) -> list[EvidenceItem]:
        """
        Recover claims from an unformatted answer.

        Args:
            provider: Provider name.
            task_id: Subtask identifier.
            task_title: Subtask label.
            text: Answer text.

        Returns:
            Evidence recovered from the most substantive sentences.
        """
        links = _harvest_links(text)
        candidates: list[tuple[float, str]] = []

        for sentence in _sentences(text):
            cleaned = _strip_markup(sentence)
            if not _MIN_FACT_CHARS <= len(cleaned) <= _MAX_FACT_CHARS:
                continue
            lowered = cleaned.lower()
            if any(lowered.startswith(hedge) for hedge in _HEDGE_OPENERS):
                continue

            score = sum(
                1.0 for marker in _SALIENCE_MARKERS if marker in lowered
            )
            score += 1.5 if re.search(r"\d", cleaned) else 0.0
            if score <= 0:
                continue
            candidates.append((score, cleaned))

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return [
            EvidenceItem(
                provider=provider,
                task_id=task_id,
                task_title=task_title,
                fact=_trim(sentence),
                source=f"{provider} (unstructured answer)",
                # Unformatted answers are demonstrably less disciplined,
                # so their claims start below a stated-confidence claim.
                confidence=0.45,
                links=tuple(links[:3]),
                raw=sentence,
            )
            for _, sentence in candidates[:6]
        ]


def _sentences(text: str) -> Iterable[str]:
    """
    Split text into sentence-like fragments.

    Args:
        text: Source text.

    Returns:
        Fragments, including list items which often carry the real
        content in provider answers.
    """
    for block in text.splitlines():
        stripped = block.strip().lstrip("-*•0123456789. ").strip()
        if not stripped:
            continue
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", stripped):
            candidate = part.strip()
            if candidate:
                yield candidate


def _harvest_links(text: str) -> list[str]:
    """
    Collect URLs from markdown links and bare URLs.

    Args:
        text: Source text.

    Returns:
        Unique URLs in first-seen order.
    """
    urls: list[str] = [match.group(2) for match in _LINK_PATTERN.finditer(text)]
    urls.extend(match.group(0) for match in _BARE_URL_PATTERN.finditer(text))

    cleaned: list[str] = []
    for url in urls:
        trimmed = url.rstrip(".,);:'\"")
        if trimmed and trimmed not in cleaned:
            cleaned.append(trimmed)
    return cleaned


def _parse_confidence(raw: str) -> float:
    """
    Convert a stated confidence into a number.

    Args:
        raw: Confidence as written by the provider.

    Returns:
        Confidence in [0, 1], defaulting to 0.6.
    """
    text = _strip_markup(raw).lower().strip()

    percent = re.search(r"(\d{1,3})\s*%", text)
    if percent:
        return max(0.0, min(1.0, int(percent.group(1)) / 100))

    decimal = re.fullmatch(r"0?\.\d+|1(?:\.0+)?", text)
    if decimal:
        return max(0.0, min(1.0, float(text)))

    for word, value in _CONFIDENCE_WORDS.items():
        if word in text:
            return value
    return 0.6


def _strip_markup(text: str) -> str:
    """
    Remove markdown emphasis and collapse whitespace.

    Args:
        text: Source text.

    Returns:
        Plain text.
    """
    cleaned = _LINK_PATTERN.sub(r"\1", text)
    cleaned = re.sub(r"[*_`#]+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _trim(text: str) -> str:
    """
    Cap a fact at a readable length.

    Args:
        text: Fact text.

    Returns:
        Trimmed fact.
    """
    cleaned = _strip_markup(text)
    if len(cleaned) <= _MAX_FACT_CHARS:
        return cleaned
    return cleaned[: _MAX_FACT_CHARS - 1].rsplit(" ", 1)[0] + "…"


def _similar(key_a: str, key_b: str) -> bool:
    """
    Report whether two claim keys describe the same claim.

    Args:
        key_a: First normalized key.
        key_b: Second normalized key.

    Returns:
        True when the keys overlap enough to be one claim.
    """
    words_a = set(key_a.split())
    words_b = set(key_b.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b)) >= 0.6


__all__ = [
    "EvidenceExtractor",
    "EvidenceItem",
    "EvidenceSet",
]
