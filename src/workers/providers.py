"""
Provider registry for browser-driven AI chat workers.

Each entry describes one provider as data. Adding a provider means adding
a :class:`ChatSiteConfig` here; no worker code changes.

Selector confidence
-------------------
``verified=True`` marks providers whose selectors have been exercised
against the live site in this codebase. The remaining providers are
best-effort: the selectors reflect each site's published DOM conventions
but have not been confirmed here, and these sites change without notice.
An unverified provider that fails is expected to degrade — WorkerManager
collects per-provider failures without aborting the others — and the
composer-not-found error names the selector to fix.

Treat every selector in this file as maintenance surface, not as a
settled fact.
"""

from __future__ import annotations

from typing import Optional

from src.browser import BrowserSession
from src.config import Settings
from src.exceptions import WorkerError

from .base_worker import WorkerConfig
from .chat_site import ChatSiteConfig
from .registry import ProviderRegistry
from .web_chat_worker import WebChatWorker

CHATGPT_SITE = ChatSiteConfig(
    name="chatgpt",
    display_name="ChatGPT",
    base_url="https://chatgpt.com/",
    composer_selector="#prompt-textarea",
    send_button_selector='[data-testid="send-button"]',
    stop_button_selector='[data-testid="stop-button"]',
    assistant_message_selector='[data-message-author-role="assistant"]',
    login_wall_selector=(
        '[data-testid="login-button"], '
        '[data-testid="welcome-login-button"]'
    ),
    capabilities=frozenset({"general", "reasoning", "code", "writing"}),
    response_timeout_seconds=150.0,
    verified=True,
)

# Claude ships disabled rather than deleted. Its interface is behind a
# Cloudflare human-verification challenge that browser automation cannot
# clear, so every attempt produced a paused provider, a recovery cycle,
# and log noise for no benefit. The configuration below stays complete and
# correct: adding "claude" to ENABLED_PROVIDERS is all that is needed to
# bring it back if that challenge is ever lifted or a manual session
# proves durable.
CLAUDE_SITE = ChatSiteConfig(
    name="claude",
    display_name="Claude",
    base_url="https://claude.ai/new",
    enabled_by_default=False,
    disabled_reason=(
        "Cloudflare human-verification challenge cannot be cleared by "
        "browser automation"
    ),
    composer_selector='div[contenteditable="true"].ProseMirror',
    send_button_selector='button[aria-label*="Send"]',
    stop_button_selector='button[aria-label*="Stop"]',
    assistant_message_selector='div[data-is-streaming], .font-claude-message',
    login_wall_selector='a[href*="/login"], button:has-text("Sign in")',
    capabilities=frozenset({"general", "reasoning", "code", "writing"}),
    response_timeout_seconds=180.0,
    verified=False,
)

GEMINI_SITE = ChatSiteConfig(
    name="gemini",
    display_name="Gemini",
    base_url="https://gemini.google.com/app",
    composer_selector='div.ql-editor[contenteditable="true"]',
    send_button_selector='button[aria-label*="Send"]',
    stop_button_selector='button[aria-label*="Stop"]',
    assistant_message_selector="model-response, message-content",
    login_wall_selector='a[href*="accounts.google.com"]',
    capabilities=frozenset({"general", "reasoning", "search"}),
    response_timeout_seconds=180.0,
    verified=False,
)

GROK_SITE = ChatSiteConfig(
    name="grok",
    display_name="Grok",
    base_url="https://grok.com/",
    composer_selector="textarea",
    send_button_selector='button[type="submit"]',
    stop_button_selector='button[aria-label*="Stop"]',
    assistant_message_selector=".message-bubble, .response-content-markdown",
    login_wall_selector='button:has-text("Sign in")',
    capabilities=frozenset({"general", "reasoning", "current_events"}),
    response_timeout_seconds=180.0,
    verified=False,
)

DEEPSEEK_SITE = ChatSiteConfig(
    name="deepseek",
    display_name="DeepSeek",
    base_url="https://chat.deepseek.com/",
    composer_selector="textarea#chat-input, textarea",
    send_button_selector='div[role="button"][aria-disabled="false"]',
    stop_button_selector='div[role="button"]:has(rect)',
    assistant_message_selector="div.ds-markdown, div._4f9bf79",
    login_wall_selector='button:has-text("Log in")',
    capabilities=frozenset({"general", "reasoning", "code", "math"}),
    verified=False,
)

PERPLEXITY_SITE = ChatSiteConfig(
    name="perplexity",
    display_name="Perplexity",
    base_url="https://www.perplexity.ai/",
    composer_selector='textarea[placeholder], div[contenteditable="true"]',
    send_button_selector='button[aria-label*="Submit"]',
    stop_button_selector='button[aria-label*="Stop"]',
    assistant_message_selector='div[class*="prose"], div[id^="markdown-content"]',
    login_wall_selector='button:has-text("Sign in")',
    capabilities=frozenset({"general", "search", "citations", "current_events"}),
    verified=False,
)

MISTRAL_SITE = ChatSiteConfig(
    name="mistral",
    display_name="Mistral",
    base_url="https://chat.mistral.ai/chat",
    composer_selector='textarea, div[contenteditable="true"]',
    send_button_selector='button[type="submit"]',
    stop_button_selector='button[aria-label*="Stop"]',
    assistant_message_selector='div[data-message-author-role="assistant"], .prose',
    login_wall_selector='a[href*="login"]',
    capabilities=frozenset({"general", "reasoning", "code"}),
    verified=False,
)

QWEN_SITE = ChatSiteConfig(
    name="qwen",
    display_name="Qwen Chat",
    base_url="https://chat.qwen.ai/",
    composer_selector='textarea#chat-input, textarea',
    send_button_selector='button[id*="send"], button[type="submit"]',
    stop_button_selector='button[aria-label*="Stop"]',
    assistant_message_selector="div.markdown-body, div[class*='assistant']",
    login_wall_selector='button:has-text("Log in")',
    capabilities=frozenset({"general", "reasoning", "code"}),
    verified=False,
)

PROVIDER_SITES: dict[str, ChatSiteConfig] = {
    site.name: site
    for site in (
        CHATGPT_SITE,
        CLAUDE_SITE,
        GEMINI_SITE,
        GROK_SITE,
        DEEPSEEK_SITE,
        PERPLEXITY_SITE,
        MISTRAL_SITE,
        QWEN_SITE,
    )
}

DEFAULT_PROVIDER = CHATGPT_SITE.name


def available_providers() -> list[str]:
    """
    Return every registered provider name.

    Returns:
        Provider names in registration order.
    """
    return list(PROVIDER_SITES)


def verified_providers() -> list[str]:
    """
    Return providers whose selectors have been confirmed in this codebase.

    Returns:
        Verified provider names.
    """
    return [
        name for name, site in PROVIDER_SITES.items() if site.verified
    ]


def get_site(name: str) -> ChatSiteConfig:
    """
    Look up a provider site description.

    Args:
        name: Provider name, case-insensitive.

    Returns:
        Site configuration.

    Raises:
        WorkerError: If the provider is not registered.
    """
    site = PROVIDER_SITES.get(name.strip().lower())
    if site is None:
        raise WorkerError(
            f"Unknown provider '{name}'. Available providers: "
            f"{', '.join(PROVIDER_SITES)}",
            code="PROVIDER_UNKNOWN",
        )
    return site


def build_worker(
    name: str,
    session: BrowserSession,
    config: Optional[WorkerConfig] = None,
) -> WebChatWorker:
    """
    Construct a worker for a registered provider.

    Args:
        name: Provider name.
        session: Browser session the worker operates in.
        config: Optional shared worker configuration.

    Returns:
        Worker bound to the provider's site description.

    Raises:
        WorkerError: If the provider is not registered.
    """
    return WebChatWorker(
        session=session,
        site=get_site(name),
        config=config,
    )


def parse_provider_list(raw: str) -> list[str]:
    """
    Parse a comma-separated provider selection.

    Unknown names are dropped rather than raising, so a typo in
    configuration degrades to the providers that were understood.

    Note that this performs no enable or disable resolution. Callers
    deciding participation must use :class:`ProviderRegistry`, which is
    the single authority on which providers may launch.

    Args:
        raw: Comma-separated provider names.

    Returns:
        Recognized provider names, de-duplicated, in the given order.
        Falls back to the default provider when nothing is recognized.
    """
    seen: list[str] = []
    for candidate in (raw or "").split(","):
        name = candidate.strip().lower()
        if name in PROVIDER_SITES and name not in seen:
            seen.append(name)
    return seen or [DEFAULT_PROVIDER]


def build_registry(settings: Optional[Settings] = None) -> ProviderRegistry:
    """
    Build the provider registry for the current configuration.

    Args:
        settings: Optional application settings.

    Returns:
        Registry resolving enabled and disabled providers.
    """
    return ProviderRegistry.from_settings(PROVIDER_SITES, settings)


def default_enabled_providers() -> list[str]:
    """
    Return providers that participate when no selection is configured.

    Returns:
        Names of providers that ship enabled.
    """
    return [
        name
        for name, site in PROVIDER_SITES.items()
        if site.enabled_by_default
    ]


def default_disabled_providers() -> list[str]:
    """
    Return providers that ship disabled.

    Returns:
        Names of providers that stay dormant until explicitly enabled.
    """
    return [
        name
        for name, site in PROVIDER_SITES.items()
        if not site.enabled_by_default
    ]


__all__ = [
    "CHATGPT_SITE",
    "CLAUDE_SITE",
    "DEEPSEEK_SITE",
    "DEFAULT_PROVIDER",
    "GEMINI_SITE",
    "GROK_SITE",
    "MISTRAL_SITE",
    "PERPLEXITY_SITE",
    "PROVIDER_SITES",
    "QWEN_SITE",
    "available_providers",
    "build_registry",
    "build_worker",
    "default_disabled_providers",
    "default_enabled_providers",
    "get_site",
    "parse_provider_list",
    "verified_providers",
]
