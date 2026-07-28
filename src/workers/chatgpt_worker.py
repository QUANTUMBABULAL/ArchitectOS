"""
ChatGPT worker.

The generic interaction logic lives in :class:`WebChatWorker`; ChatGPT's
selectors and timings live in :data:`CHATGPT_SITE`. This module keeps the
``ChatGPTWorker`` name stable for existing callers while carrying no
duplicated automation logic.

Login is always a manual, one-time user action. When the worker detects a
login wall it fails with an explicit error rather than attempting to
authenticate.
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
from .providers import CHATGPT_SITE
from .web_chat_worker import WebChatWorker

# Retained for backward compatibility with earlier imports.
ChatGPTWorkerConfig = ChatSiteConfig


class ChatGPTWorker(WebChatWorker):
    """
    Worker that consults ChatGPT through the browser.

    A thin binding of :class:`WebChatWorker` to ChatGPT's site
    description. Present as a named class so logs, registration, and
    dependency injection can refer to the provider explicitly.
    """

    WORKER_NAME = CHATGPT_SITE.name

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
        Initialize the ChatGPT worker.

        Args:
            session: Browser session to operate in.
            config: Optional shared worker configuration.
            site_config: Optional site override, for adapting to a
                ChatGPT layout change without editing the registry.
            tab_manager: Optional tab manager dependency.
            keyboard: Optional keyboard controller dependency.
            mouse: Optional mouse controller dependency.
            extractor: Optional extractor dependency.
        """
        super().__init__(
            session=session,
            site=site_config or CHATGPT_SITE,
            config=config,
            tab_manager=tab_manager,
            keyboard=keyboard,
            mouse=mouse,
            extractor=extractor,
        )


__all__ = [
    "ChatGPTWorker",
    "ChatGPTWorkerConfig",
]
