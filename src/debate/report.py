"""
Deterministic reporting for multi-round debates.

Report assembly is separated from debate execution so a report can be
produced without inference. The report records what each provider said,
who agreed, who did not, and why the debate stopped — including when it
stopped without converging, which must never be presented as agreement.
"""

from __future__ import annotations

from .debate_engine import DebateOutcome


def _render_findings(outcome: DebateOutcome) -> list[str]:
    """
    Render the merged recommendation table and its asymmetries.

    Args:
        outcome: Completed debate outcome.

    Returns:
        Report lines, empty when no structured findings were extracted.
    """
    aggregate = next(
        (
            round_.aggregate
            for round_ in reversed(outcome.rounds)
            if round_.aggregate is not None and round_.aggregate.products
        ),
        None,
    )
    if aggregate is None:
        return []

    lines: list[str] = ["", "## Merged recommendations", ""]
    lines.append("| Recommendation | Supported by | Confidence | Price |")
    lines.append("| --- | --- | --- | --- |")

    for product in aggregate.top(10):
        supporters = ", ".join(product.supporters)
        prices = (
            "; ".join(
                f"{provider} {price.render()}"
                for provider, price in product.prices
            )
            or "—"
        )
        lines.append(
            f"| {product.display_name} "
            f"| {supporters} ({product.support_count}) "
            f"| {product.confidence:.2f} "
            f"| {prices} |"
        )

    if aggregate.unanimous:
        lines.extend(
            [
                "",
                "**Unanimous:** "
                + ", ".join(p.display_name for p in aggregate.unanimous),
            ]
        )

    if aggregate.contested:
        lines.extend(["", "### Contested", ""])
        for product in aggregate.contested:
            lines.append(
                f"- **{product.display_name}** — recommended by "
                f"{', '.join(product.supporters)}; omitted by "
                f"{', '.join(product.dissenters)}"
            )

    if aggregate.unique:
        lines.extend(["", "### Unique to one provider", ""])
        for product in aggregate.unique:
            lines.append(
                f"- **{product.display_name}** — only "
                f"{product.supporters[0]}"
            )

    if aggregate.silent_providers:
        lines.extend(
            [
                "",
                "*Answered without a concrete recommendation: "
                + ", ".join(aggregate.silent_providers)
                + "*",
            ]
        )

    citations = {
        citation.url: citation
        for finding in aggregate.findings
        for citation in finding.citations
    }
    if citations:
        lines.extend(["", "## Citations", ""])
        for citation in list(citations.values())[:25]:
            label = citation.title or citation.domain or citation.url
            lines.append(f"- [{label}]({citation.url})")

    return lines


def render_report(outcome: DebateOutcome) -> str:
    """
    Render a debate outcome as a Markdown report.

    Args:
        outcome: Completed debate outcome.

    Returns:
        Markdown report text.
    """
    lines: list[str] = [
        f"# Research Report: {outcome.question}",
        "",
        "## Consensus",
        "",
    ]

    consensus = outcome.final_consensus
    if consensus is None:
        lines.append(
            "No consensus could be computed: no provider returned a "
            "usable answer."
        )
        return "\n".join(lines)

    verdict = (
        "Converged" if outcome.stop_reason.is_converged else "Not converged"
    )
    lines.extend(
        [
            f"- **Verdict:** {verdict} ({outcome.stop_reason.value})",
            f"- **Confidence:** {consensus.confidence:.2f}",
            f"- **Agreement:** {consensus.agreement_score:.2f}",
            f"- **Rounds:** {outcome.round_count}",
            f"- **Providers answering:** {consensus.opinion_count}",
        ]
    )

    if outcome.supporting:
        lines.append(
            f"- **Aligned:** {', '.join(outcome.supporting)}"
        )
    if outcome.opposing:
        lines.append(
            f"- **Still disagreeing:** {', '.join(outcome.opposing)}"
        )

    if not outcome.stop_reason.is_converged:
        lines.extend(
            [
                "",
                "> This debate ended without convergence. Treat the "
                "positions below as unresolved rather than settled.",
            ]
        )

    if consensus.contradictions:
        lines.extend(["", "## Unresolved contradictions", ""])
        for contradiction in consensus.contradictions:
            lines.append(
                f"- **{contradiction.source_a} vs "
                f"{contradiction.source_b}:** "
                f"{contradiction.description}"
            )

    lines.extend(_render_findings(outcome))

    if consensus.follow_up_questions:
        lines.extend(["", "## Open questions", ""])
        for question in consensus.follow_up_questions:
            lines.append(f"- {question}")

    lines.extend(["", "## Final positions", ""])
    for provider, answer in sorted(outcome.latest_answers().items()):
        lines.extend([f"### {provider}", "", answer.strip(), ""])

    lines.extend(["## Debate transcript", ""])
    for round_ in outcome.rounds:
        answered = len(round_.answered)
        total = len(round_.responses)
        lines.append(
            f"### Round {round_.number} "
            f"({answered}/{total} answered)"
        )
        lines.append("")
        if round_.consensus is not None:
            lines.append(
                f"*agreement {round_.consensus.agreement_score:.2f}, "
                f"confidence {round_.consensus.confidence:.2f}*"
            )
            lines.append("")
        for response in round_.responses:
            if response.success:
                lines.append(
                    f"- {response.worker_name}: answered in "
                    f"{response.elapsed_seconds:.1f}s"
                )
            else:
                lines.append(
                    f"- {response.worker_name}: failed — {response.error}"
                )
        lines.append("")

    return "\n".join(lines).strip()


__all__ = [
    "render_report",
]
