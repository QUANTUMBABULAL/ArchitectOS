"""
Latency probe for the local Ollama model.

Measures the real cost of each brain operation against the running Ollama
server so timeout regressions are diagnosed with numbers rather than
guesses. Reports prompt tokens, generated tokens, generation rate, and
whether output hit its cap.

Run from the project root with the project virtual environment:

    python -m scripts.probe_ollama
    python -m scripts.probe_ollama --task "compare rust and go for web servers"

Exit codes:
    0  every probe completed within its budget
    1  the server was unreachable, or a probe exceeded its budget
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Optional

from src.brain import DecisionEngine, OllamaClient
from src.config import get_settings
from src.exceptions import AIResearchOperatorError
from src.planner import Planner

_SIMPLE_TASK = "hello"
_MODERATE_TASK = "what is the capital of France"
_COMPLEX_TASK = (
    "compare the current state of the art in local inference runtimes "
    "and recommend one for a laptop"
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """
    Outcome of one probe.

    Attributes:
        name: Probe label.
        seconds: Wall-clock duration.
        budget: Expected upper bound in seconds.
        detail: Short description of what happened.
        failed: True when the probe errored.
    """

    name: str
    seconds: float
    budget: float
    detail: str
    failed: bool = False

    @property
    def over_budget(self) -> bool:
        """
        Return whether the probe exceeded its budget.

        Returns:
            True when slower than the budget or failed outright.
        """
        return self.failed or self.seconds > self.budget

    def render(self) -> str:
        """
        Render a single result line.

        Returns:
            Formatted result row.
        """
        status = "FAIL" if self.over_budget else "ok"
        return (
            f"[{status:>4}] {self.name:<28} {self.seconds:6.2f}s "
            f"(budget {self.budget:.0f}s)  {self.detail}"
        )


async def _timed(
    name: str,
    budget: float,
    coro: object,
    client: OllamaClient,
) -> ProbeResult:
    """
    Run one awaitable and capture timing plus server metrics.

    Args:
        name: Probe label.
        budget: Expected upper bound in seconds.
        coro: Awaitable to execute.
        client: Client whose metrics are read after completion.

    Returns:
        Probe result.
    """
    started = time.monotonic()
    try:
        outcome = await coro  # type: ignore[misc]
    except AIResearchOperatorError as exc:
        return ProbeResult(
            name=name,
            seconds=time.monotonic() - started,
            budget=budget,
            detail=str(exc)[:120],
            failed=True,
        )

    elapsed = time.monotonic() - started
    metrics = client.last_metrics
    if metrics is not None and metrics.operation != "preload":
        detail = metrics.describe()
    else:
        detail = str(outcome)[:80]

    return ProbeResult(
        name=name,
        seconds=elapsed,
        budget=budget,
        detail=detail,
    )


async def run_probes(task: Optional[str] = None) -> int:
    """
    Execute the probe suite.

    Args:
        task: Optional custom task used for the classification probes.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    client = OllamaClient(settings=settings)

    print(f"Server : {client.host}")
    print(f"Model  : {client.model}")
    print(
        f"Budget : fast={settings.ollama_fast_timeout:.0f}s "
        f"long={settings.ollama_timeout:.0f}s "
        f"think={settings.ollama_think}"
    )
    print("-" * 78)

    if not await client.health_check():
        print(f"[FAIL] Ollama is not reachable at {client.host}")
        return 1

    await client.resolve_model()

    engine = DecisionEngine(client, settings=settings)
    planner = Planner(engine, settings=settings)
    fast = settings.ollama_fast_timeout
    long = settings.ollama_timeout

    results: list[ProbeResult] = [
        await _timed("preload model", long, client.preload(), client),
        await _timed(
            "classify (fast path)",
            1.0,
            engine.classify_complexity(_SIMPLE_TASK),
            client,
        ),
        await _timed(
            "classify (model)",
            fast,
            engine.classify_complexity(task or _MODERATE_TASK),
            client,
        ),
        await _timed(
            "classify (cached)",
            1.0,
            engine.classify_complexity(task or _MODERATE_TASK),
            client,
        ),
        await _timed(
            "plan (complex goal)",
            long,
            planner.create_plan(_COMPLEX_TASK),
            client,
        ),
    ]

    for result in results:
        print(result.render())

    print("-" * 78)
    breached = [r for r in results if r.over_budget]
    if breached:
        print(f"{len(breached)} probe(s) exceeded budget or failed.")
        return 1

    print("All probes within budget.")
    return 0


def main() -> int:
    """
    Parse arguments and run the probe suite.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Measure local Ollama latency for brain operations."
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Custom task text for the classification probes.",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(run_probes(args.task))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
