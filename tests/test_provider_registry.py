"""
Tests for provider enable and disable resolution.

The registry is the single authority on participation. Two properties
carry the weight: an explicitly disabled provider is disabled no matter
what else says otherwise, and Claude in particular stays dormant while its
code remains present and re-enableable.
"""

from __future__ import annotations

import pytest

from src.exceptions import WorkerError
from src.workers import (
    CLAUDE_SITE,
    PROVIDER_SITES,
    DisableReason,
    ProviderRegistry,
    default_disabled_providers,
    default_enabled_providers,
)


def registry(
    enabled: list[str] | None = None,
    disabled: list[str] | None = None,
) -> ProviderRegistry:
    """
    Build a registry over the real provider set.

    Args:
        enabled: Explicit enabled list.
        disabled: Explicit disabled list.

    Returns:
        Configured registry.
    """
    return ProviderRegistry(
        sites=PROVIDER_SITES, enabled=enabled, disabled=disabled
    )


class TestClaudeIsDisabled:
    """Claude must not participate, without being deleted."""

    def test_claude_ships_disabled(self) -> None:
        """Claude is excluded when no configuration is supplied."""
        assert "claude" not in registry().enabled_names()
        assert "claude" in registry().disabled_names()

    def test_claude_default_is_recorded_on_the_site(self) -> None:
        """The default lives on the site config, not in scattered checks."""
        assert CLAUDE_SITE.enabled_by_default is False
        assert "cloudflare" in CLAUDE_SITE.disabled_reason.lower()

    def test_claude_code_is_still_present(self) -> None:
        """
        Disabling is configuration, not deletion. The site config must
        remain complete so re-enabling requires no code change.
        """
        assert "claude" in PROVIDER_SITES
        assert CLAUDE_SITE.composer_selector
        assert CLAUDE_SITE.assistant_message_selector
        assert CLAUDE_SITE.base_url.startswith("https://")

    def test_claude_can_be_re_enabled_by_configuration_alone(self) -> None:
        """Naming Claude in the enabled list is sufficient to restore it."""
        resolved = registry(enabled=["chatgpt", "claude"], disabled=[])
        assert "claude" in resolved.enabled_names()
        assert resolved.is_enabled("claude") is True

    def test_default_lists_agree_with_registry(self) -> None:
        """The helper functions and the registry cannot drift apart."""
        assert "claude" in default_disabled_providers()
        assert "claude" not in default_enabled_providers()


class TestResolutionPrecedence:
    """Ordering of the enable and disable rules."""

    def test_disabled_beats_enabled(self) -> None:
        """
        An explicit refusal is never reversed by an explicit request. A
        provider in both lists stays disabled.
        """
        resolved = registry(enabled=["chatgpt", "gemini"], disabled=["gemini"])
        assert resolved.enabled_names() == ["chatgpt"]
        assert resolved.is_enabled("gemini") is False

    def test_explicit_list_excludes_unnamed_providers(self) -> None:
        """An enabled list is exhaustive, not additive."""
        resolved = registry(enabled=["chatgpt"], disabled=[])
        assert resolved.enabled_names() == ["chatgpt"]
        assert "grok" in resolved.disabled_names()

    def test_empty_enabled_uses_defaults(self) -> None:
        """With no list, every default-enabled provider participates."""
        resolved = registry(enabled=None, disabled=[])
        assert set(resolved.enabled_names()) == set(
            default_enabled_providers()
        )

    def test_order_follows_configuration(self) -> None:
        """Tab-open order is the operator's choice."""
        resolved = registry(enabled=["grok", "chatgpt", "gemini"])
        assert resolved.enabled_names() == ["grok", "chatgpt", "gemini"]

    def test_unknown_names_are_ignored(self) -> None:
        """A typo degrades rather than failing startup."""
        resolved = registry(enabled=["chatgpt", "bard"], disabled=[])
        assert resolved.enabled_names() == ["chatgpt"]

    def test_duplicates_collapse(self) -> None:
        """A provider is never registered twice."""
        resolved = registry(enabled=["chatgpt", "chatgpt"])
        assert resolved.enabled_names() == ["chatgpt"]


class TestDisableReasons:
    """Why a provider is excluded is always recorded."""

    def test_configured_reason(self) -> None:
        """Explicit refusal is distinguishable from other exclusions."""
        resolved = registry(enabled=["chatgpt"], disabled=["gemini"])
        entry = next(
            r for r in resolved.registrations if r.name == "gemini"
        )
        assert entry.reason is DisableReason.CONFIGURED

    def test_default_reason_carries_detail(self) -> None:
        """A provider disabled by default explains itself."""
        entry = next(
            r for r in registry().registrations if r.name == "claude"
        )
        assert entry.reason is DisableReason.DEFAULT
        assert entry.detail

    def test_not_selected_reason(self) -> None:
        """Omission from an explicit list is its own category."""
        resolved = registry(enabled=["chatgpt"], disabled=[])
        entry = next(r for r in resolved.registrations if r.name == "grok")
        assert entry.reason is DisableReason.NOT_SELECTED


class TestFiltering:
    """The choke point every consumer uses."""

    def test_filter_drops_disabled(self) -> None:
        """A disabled provider cannot pass through the filter."""
        resolved = registry(enabled=["chatgpt", "gemini"], disabled=["claude"])
        assert resolved.filter_enabled(
            ["chatgpt", "claude", "gemini"]
        ) == ["chatgpt", "gemini"]

    def test_filter_drops_unknown(self) -> None:
        """Unknown names are dropped, not passed through."""
        assert registry(enabled=["chatgpt"]).filter_enabled(
            ["chatgpt", "nonsense"]
        ) == ["chatgpt"]

    def test_filter_preserves_order_and_dedupes(self) -> None:
        """Caller ordering survives; duplicates do not."""
        resolved = registry(enabled=["chatgpt", "gemini", "grok"])
        assert resolved.filter_enabled(
            ["grok", "chatgpt", "grok"]
        ) == ["grok", "chatgpt"]

    def test_is_enabled_tolerates_case_and_space(self) -> None:
        """Lookup is forgiving about formatting."""
        assert registry(enabled=["chatgpt"]).is_enabled("  ChatGPT ") is True

    def test_is_enabled_false_for_unknown(self) -> None:
        """An unknown provider is disabled, not an error."""
        assert registry().is_enabled("bard") is False


class TestRequireEnabled:
    """Strict lookup for callers that must not proceed."""

    def test_returns_registration_when_enabled(self) -> None:
        """An enabled provider resolves to its registration."""
        entry = registry(enabled=["chatgpt"]).require_enabled("chatgpt")
        assert entry.enabled is True
        assert entry.display_name == "ChatGPT"

    def test_raises_for_disabled(self) -> None:
        """Requesting a disabled provider is an explicit error."""
        with pytest.raises(WorkerError) as excinfo:
            registry().require_enabled("claude")
        assert excinfo.value.code == "PROVIDER_DISABLED"

    def test_raises_for_unknown(self) -> None:
        """Requesting an unknown provider is a distinct error."""
        with pytest.raises(WorkerError) as excinfo:
            registry().require_enabled("bard")
        assert excinfo.value.code == "PROVIDER_UNKNOWN"


class TestStartupSummary:
    """The operator-facing report."""

    def test_lists_both_sections(self) -> None:
        """Enabled and disabled providers are both shown."""
        summary = registry(
            enabled=["chatgpt", "gemini", "grok"], disabled=["claude"]
        ).startup_summary()

        assert "Enabled Providers:" in summary
        assert "Disabled Providers:" in summary
        assert "ChatGPT" in summary
        assert "Gemini" in summary
        assert "Grok" in summary
        assert "Claude" in summary

    def test_claude_appears_only_under_disabled(self) -> None:
        """Claude must never be listed as enabled."""
        summary = registry().startup_summary()
        enabled_block, disabled_block = summary.split("Disabled Providers:")

        assert "Claude" not in enabled_block
        assert "Claude" in disabled_block

    def test_handles_no_enabled_providers(self) -> None:
        """An empty selection renders without raising."""
        summary = registry(enabled=[], disabled=[]).startup_summary()
        assert "(none)" in summary
