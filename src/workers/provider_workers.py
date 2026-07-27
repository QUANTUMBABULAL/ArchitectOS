"""
Named workers for the non-ChatGPT AI providers.

Each class binds :class:`WebChatWorker` to one provider's site
description. They exist as named types so registration, logging, and
dependency injection can refer to a provider explicitly, and so a
provider needing genuinely different behaviour later has an obvious place
to override a single method.

Selectors for these providers are unverified in this codebase — see
``providers.py``. A provider whose selectors have drifted fails on its
own and the remaining providers continue.
"""

from __future__ import annotations

from typing import Optional

from src.browser import (
    BrowserSession,
    Extractor,
    KeyboardController,
    MouseController,
    TabManager,
)

from .base_worker import WorkerConfig
from .chat_site import ChatSiteConfig
from .providers import CLAUDE_SITE, GEMINI_SITE, GROK_SITE
from .web_chat_worker import WebChatWorker


class _ProviderWorker(WebChatWorker):
    """
    Shared constructor for provider bindings.

    Subclasses supply ``SITE``; everything else is inherited. This keeps
    each concrete provider to a two-line declaration.
    """

    SITE: ChatSiteConfig

    def __init__(
        self,
        session: BrowserSession,
        config: Optional[WorkerConfig] = None,
        site_config: Optional[ChatSiteConfig] = None,
        tab_manager: Optional[TabManager] = None,
        keyboard: Optional[KeyboardController] = None,
        mouse: Optional[MouseController] = None,
        extractor: Optional[Extractor] = None,
    ) -> None:
        """
        Initialize the provider worker.

        Args:
            session: Browser session to operate in.
            config: Optional shared worker configuration.
            site_config: Optional site override, for adapting to a layout
                change without editing the registry.
            tab_manager: Optional tab manager dependency.
            keyboard: Optional keyboard controller dependency.
            mouse: Optional mouse controller dependency.
            extractor: Optional extractor dependency.
        """
        super().__init__(
            session=session,
            site=site_config or self.SITE,
            config=config,
            tab_manager=tab_manager,
            keyboard=keyboard,
            mouse=mouse,
            extractor=extractor,
        )


class ClaudeWorker(_ProviderWorker):
    """Worker that consults Claude through the browser."""

    SITE = CLAUDE_SITE
    WORKER_NAME = CLAUDE_SITE.name


class GeminiWorker(_ProviderWorker):
    """Worker that consults Gemini through the browser."""

    SITE = GEMINI_SITE
    WORKER_NAME = GEMINI_SITE.name


class GrokWorker(_ProviderWorker):
    """Worker that consults Grok through the browser."""

    SITE = GROK_SITE
    WORKER_NAME = GROK_SITE.name


__all__ = [
    "ClaudeWorker",
    "GeminiWorker",
    "GrokWorker",
]
