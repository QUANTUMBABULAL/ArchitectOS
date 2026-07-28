"""
Tests for the provider registry and multi-provider fault tolerance.

The load-bearing property of the fast path is that one bad provider does
not sink the run. These tests assert that at the WorkerManager boundary,
where the concurrency and error isolation actually live.
"""

from __future__ import annotations

import asyncio

import pytest

from src.exceptions import WorkerError
from src.workers import (
    PROVIDER_SITES,
    WorkerHealth,
    WorkerQuery,
    WorkerResponse,
    WorkerState,
    available_providers,
    get_site,
    parse_provider_list,
    verified_providers,
)
from src.workers.worker_manager import WorkerManager


class TestRegistry:
    """Provider registry contents and lookup."""

    def test_all_providers_registered(self) -> None:
        """Every supported provider is present in the registry."""
        assert set(available_providers()) == {
            "chatgpt",
            "claude",
            "gemini",
            "grok",
            "deepseek",
            "perplexity",
            "mistral",
            "qwen",
        }

    def test_chatgpt_is_the_verified_provider(self) -> None:
        """
        Only ChatGPT has selectors exercised in this codebase. If another
        provider is later confirmed against the live site, update its
        config and this test together.
        """
        assert verified_providers() == ["chatgpt"]

    def test_every_site_has_required_selectors(self) -> None:
        """Each provider defines the selectors the worker depends on."""
        for name, site in PROVIDER_SITES.items():
            assert site.composer_selector, name
            assert site.assistant_message_selector, name
            assert site.base_url.startswith("https://"), name
            assert site.display_name, name

    def test_names_match_keys(self) -> None:
        """Registry keys agree with the site names used for registration."""
        for key, site in PROVIDER_SITES.items():
            assert key == site.name

    def test_lookup_is_case_insensitive(self) -> None:
        """Lookup tolerates casing and surrounding whitespace."""
        assert get_site("  ChatGPT ").name == "chatgpt"

    def test_unknown_provider_raises(self) -> None:
        """An unknown provider is an explicit error, not a silent default."""
        with pytest.raises(WorkerError) as excinfo:
            get_site("bard")
        assert excinfo.value.code == "PROVIDER_UNKNOWN"


class TestProviderListParsing:
    """Parsing the configured provider selection."""

    def test_parses_all(self) -> None:
        """A full list is preserved in order."""
        assert parse_provider_list("chatgpt,claude,gemini,grok") == [
            "chatgpt",
            "claude",
            "gemini",
            "grok",
        ]

    def test_tolerates_whitespace_and_case(self) -> None:
        """Formatting noise does not break parsing."""
        assert parse_provider_list(" ChatGPT , CLAUDE ") == [
            "chatgpt",
            "claude",
        ]

    def test_drops_unknown_names(self) -> None:
        """
        A typo degrades to the providers that were understood rather than
        failing startup.
        """
        assert parse_provider_list("chatgpt,bard,claude") == [
            "chatgpt",
            "claude",
        ]

    def test_deduplicates(self) -> None:
        """Repeats collapse so a provider is not registered twice."""
        assert parse_provider_list("claude,claude") == ["claude"]

    def test_empty_falls_back_to_default(self) -> None:
        """Empty or fully invalid input yields the default provider."""
        assert parse_provider_list("") == ["chatgpt"]
        assert parse_provider_list("nonsense,bogus") == ["chatgpt"]


class FakeWorker:
    """
    Minimal worker double for dispatch and startup tests.

    Implements only the surface WorkerManager touches.
    """

    def __init__(
        self,
        name: str,
        *,
        fail_start: bool = False,
        fail_ask: bool = False,
        raise_ask: bool = False,
    ) -> None:
        """
        Initialize the double.

        Args:
            name: Worker name.
            fail_start: Raise during start.
            fail_ask: Return an unsuccessful response.
            raise_ask: Raise from ask instead of returning.
        """
        self.name = name
        self.capabilities = frozenset({"general"})
        self.state = WorkerState.CREATED
        self._fail_start = fail_start
        self._fail_ask = fail_ask
        self._raise_ask = raise_ask
        self.asked = False

    async def start(self) -> None:
        """Start, or fail if configured to."""
        if self._fail_start:
            raise WorkerError(f"{self.name} login required")
        self.state = WorkerState.READY

    async def stop(self) -> None:
        """Stop the worker."""
        self.state = WorkerState.STOPPED

    async def ask(self, query: WorkerQuery) -> WorkerResponse:
        """Answer, fail, or raise depending on configuration."""
        self.asked = True
        if self._raise_ask:
            raise RuntimeError(f"{self.name} crashed")
        return WorkerResponse(
            query_id=query.query_id,
            worker_name=self.name,
            prompt=query.prompt,
            answer="" if self._fail_ask else f"answer from {self.name}",
            success=not self._fail_ask,
            error="provider error" if self._fail_ask else None,
        )

    async def health_check(self) -> WorkerHealth:
        """Report health based on state."""
        return WorkerHealth(
            worker_name=self.name,
            state=self.state,
            healthy=self.state is WorkerState.READY,
        )


class TestFaultTolerance:
    """One failing provider must not stop the others."""

    def test_start_available_skips_failures(self) -> None:
        """A provider that cannot start is skipped, others still start."""
        manager = WorkerManager()
        manager.register(FakeWorker("chatgpt"))  # type: ignore[arg-type]
        manager.register(
            FakeWorker("claude", fail_start=True)  # type: ignore[arg-type]
        )
        manager.register(FakeWorker("gemini"))  # type: ignore[arg-type]

        results = asyncio.run(manager.start_available())

        assert results["chatgpt"] is None
        assert results["gemini"] is None
        assert "login required" in (results["claude"] or "")
        assert {w.name for w in manager.ready_workers()} == {
            "chatgpt",
            "gemini",
        }

    def test_start_all_still_raises(self) -> None:
        """
        The strict variant keeps its original contract, so existing
        callers are unaffected by the tolerant addition.
        """
        manager = WorkerManager()
        manager.register(
            FakeWorker("claude", fail_start=True)  # type: ignore[arg-type]
        )
        with pytest.raises(WorkerError):
            asyncio.run(manager.start_all())

    def test_dispatch_continues_past_failures(self) -> None:
        """A failed and a crashed provider do not block a good one."""
        manager = WorkerManager()
        manager.register(FakeWorker("chatgpt"))  # type: ignore[arg-type]
        manager.register(
            FakeWorker("claude", fail_ask=True)  # type: ignore[arg-type]
        )
        manager.register(
            FakeWorker("grok", raise_ask=True)  # type: ignore[arg-type]
        )
        asyncio.run(manager.start_available())

        responses = asyncio.run(
            manager.dispatch_many(
                ["chatgpt", "claude", "grok"],
                WorkerQuery(prompt="test"),
            )
        )

        assert len(responses) == 3
        by_name = {r.worker_name: r for r in responses}
        assert by_name["chatgpt"].success is True
        assert by_name["claude"].success is False
        assert by_name["grok"].success is False
        assert "crashed" in (by_name["grok"].error or "")

    def test_every_provider_is_asked(self) -> None:
        """All providers receive the prompt, not just the first."""
        workers = [
            FakeWorker("chatgpt"),
            FakeWorker("claude"),
            FakeWorker("gemini"),
            FakeWorker("grok"),
        ]
        manager = WorkerManager()
        for worker in workers:
            manager.register(worker)  # type: ignore[arg-type]
        asyncio.run(manager.start_available())

        asyncio.run(
            manager.dispatch_many(
                [w.name for w in workers], WorkerQuery(prompt="q")
            )
        )
        assert all(worker.asked for worker in workers)

    def test_responses_preserve_request_order(self) -> None:
        """
        Response order matches the requested order, so consensus sources
        are attributable regardless of which provider finished first.
        """
        manager = WorkerManager()
        for name in ("chatgpt", "claude", "gemini"):
            manager.register(FakeWorker(name))  # type: ignore[arg-type]
        asyncio.run(manager.start_available())

        responses = asyncio.run(
            manager.dispatch_many(
                ["gemini", "chatgpt", "claude"],
                WorkerQuery(prompt="q"),
            )
        )
        assert [r.worker_name for r in responses] == [
            "gemini",
            "chatgpt",
            "claude",
        ]
