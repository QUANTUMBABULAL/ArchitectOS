"""
Generic worker driving any browser-based AI chat provider.

The interaction pattern is identical across ChatGPT, Claude, Gemini, and
Grok: open a conversation, wait for the composer, submit a prompt, wait
for the streamed answer to settle, extract it. WebChatWorker implements
that once and takes the provider's differences as a
:class:`ChatSiteConfig`, so a new provider costs a configuration entry
rather than a new implementation.

Each worker owns exactly one tab. Tabs are never activated during normal
operation: input is delivered through page-scoped Playwright locators,
which do not require the tab to be frontmost. That is what makes it safe
to drive several providers concurrently in one browser — bringing tabs to
the front would make concurrent typing race for focus.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from playwright.async_api import Page

from src.browser import (
    BrowserSession,
    Extractor,
    KeyboardController,
    MouseController,
    TabManager,
)
from src.browser.diagnostics import ConsoleRecorder, capture_diagnostics
from src.browser.page_state import (
    PageState,
    StabilizationFailure,
    StabilizationResult,
)
from src.browser.stabilizer import PageStabilizer
from src.config import get_settings
from src.exceptions import (
    ProviderAuthError,
    ProviderChallengeError,
    WorkerError,
)
from src.logger import get_logger

from .auth import (
    AuthState,
    AuthStatus,
    challenge_prompt,
    expiry_notice,
    login_prompt,
)
from .base_worker import (
    BaseWorker,
    WorkerConfig,
    WorkerQuery,
    WorkerResponse,
    WorkerState,
)
from .chat_site import ChatSiteConfig
from .conversation import ConversationState


class WebChatWorker(BaseWorker):
    """
    Worker that consults a browser-based AI chat provider.

    The worker owns one tab. Each query with ``new_conversation=True``
    returns to the provider's base URL so answers are not contaminated by
    earlier conversation state. Completion is detected by waiting for a
    new assistant message, the disappearance of the streaming indicator,
    and stable message text across consecutive polls.

    Authentication is never automated. When a login wall is detected the
    worker fails with an explicit, actionable error.
    """

    def __init__(
        self,
        session: BrowserSession,
        site: ChatSiteConfig,
        config: Optional[WorkerConfig] = None,
        tab_manager: Optional[TabManager] = None,
        keyboard: Optional[KeyboardController] = None,
        mouse: Optional[MouseController] = None,
        extractor: Optional[Extractor] = None,
    ) -> None:
        """
        Initialize the worker.

        Args:
            session: Browser session to operate in.
            site: Provider site description.
            config: Optional shared worker configuration.
            tab_manager: Optional tab manager dependency.
            keyboard: Optional keyboard controller dependency.
            mouse: Optional mouse controller dependency.
            extractor: Optional extractor dependency.
        """
        super().__init__(
            name=site.name,
            session=session,
            config=config,
            tab_manager=tab_manager,
            keyboard=keyboard,
            mouse=mouse,
            extractor=extractor,
        )
        self._site = site
        self._page: Optional[Page] = None
        self._conversation = ConversationState(provider=site.name)
        self._auth_state = AuthState.UNKNOWN
        self._last_page_state = PageState.UNKNOWN
        self._active_input_selector: Optional[str] = None
        self._console = ConsoleRecorder(provider=site.name)
        self._diagnostics_dir = (
            Path(get_settings().data_dir) / "diagnostics"
        )
        self._stabilizer = PageStabilizer(
            input_selectors=site.prompt_input_selectors(),
            login_selectors=site.login_wall_selector,
            challenge_selectors=site.challenge_selector,
            login_url_patterns=site.login_urls(),
            challenge_url_patterns=site.challenge_urls(),
            settle_seconds=site.settle_seconds,
            max_url_changes=site.max_url_changes,
            logger_name=site.name,
        )
        self._logger = get_logger(f"{__name__}.{site.name}")

    @property
    def page_state(self) -> PageState:
        """
        Return the last observed page state.

        Returns:
            Page state from the most recent stabilization.
        """
        return self._last_page_state

    @property
    def site(self) -> ChatSiteConfig:
        """
        Return the provider site description.

        Returns:
            Site configuration.
        """
        return self._site

    @property
    def display_name(self) -> str:
        """
        Return the human-readable provider name.

        Returns:
            Display name.
        """
        return self._site.display_name

    @property
    def capabilities(self) -> frozenset[str]:
        """
        Return capability tags for this provider.

        Returns:
            Capability tags used for consultation routing.
        """
        return self._site.capabilities

    @property
    def conversation(self) -> ConversationState:
        """
        Return this provider's conversation state.

        Returns:
            Mutable conversation record.
        """
        return self._conversation

    # ------------------------------------------------------------------
    # Public provider interface
    #
    # These methods form the contract the session manager and debate
    # engine depend on. Adding a provider requires implementing nothing
    # new: WebChatWorker satisfies the interface for any site describable
    # by a ChatSiteConfig.
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Open the provider tab and make it ready for prompts.

        Idempotent: a provider already in the ready state is left alone,
        so calling this on a live session never reloads a page.

        Raises:
            WorkerError: If the provider cannot become ready.
        """
        await self.start()

    async def is_ready(self) -> bool:
        """
        Report whether the provider can accept a prompt right now.

        Returns:
            True when the tab is open and the composer is usable.
        """
        return await self._is_site_ready()

    # ------------------------------------------------------------------
    # Authentication
    #
    # ArchitectOS never handles credentials. These methods only observe
    # what the page shows; signing in is always a manual user action
    # performed in the browser window.
    # ------------------------------------------------------------------

    @property
    def auth_state(self) -> AuthState:
        """
        Return the last observed authentication state.

        Returns:
            Authentication state.
        """
        return self._auth_state

    async def is_logged_in(self) -> bool:
        """
        Report whether the provider currently appears signed in.

        A provider is considered authenticated when its composer is
        usable and neither a sign-in wall nor a verification challenge is
        displayed. Providers that need no account are always reported as
        signed in.

        Returns:
            True when the provider is usable without further sign-in.
        """
        if not self._site.requires_auth:
            return True

        page = self._page
        if page is None or page.is_closed():
            return False

        if await self._challenge_visible(page):
            return False
        if await self._login_wall_visible(page):
            return False

        return await self._is_site_ready()

    async def check_auth(self) -> AuthStatus:
        """
        Classify the provider's current authentication state.

        Distinguishes the four conditions that need different responses:
        signed in, needs a sign-in, blocked by a challenge, or offline.
        State transitions are logged so an operator can see exactly when
        a session lapsed.

        Returns:
            Authentication snapshot.
        """
        display = self._site.display_name

        if not self._site.requires_auth:
            return self._record_auth(
                AuthState.READY, "Provider requires no account"
            )

        page = self._page
        if page is None or page.is_closed():
            return self._record_auth(
                AuthState.OFFLINE,
                "Tab is not open",
                action="Recovery will reopen this tab automatically.",
            )

        try:
            if await self._challenge_visible(page):
                return self._record_auth(
                    AuthState.CAPTCHA_REQUIRED,
                    "Human verification challenge displayed",
                    action=challenge_prompt(display),
                )

            if await self._login_wall_visible(page):
                was_ready = self._auth_state is AuthState.READY
                return self._record_auth(
                    AuthState.LOGIN_REQUIRED,
                    "Sign-in screen displayed",
                    action=(
                        expiry_notice(display)
                        if was_ready
                        else login_prompt(display, self._site.base_url)
                    ),
                )

            if await self._is_site_ready():
                return self._record_auth(AuthState.READY, "Signed in")

            return self._record_auth(
                AuthState.OFFLINE,
                "Composer is not usable and no sign-in screen was found",
                action=(
                    "The page may have failed to load or its layout may "
                    "have changed."
                ),
            )
        except Exception as exc:
            return self._record_auth(
                AuthState.OFFLINE, f"Authentication check failed: {exc}"
            )

    async def wait_for_login(
        self,
        timeout_seconds: float,
        poll_interval_seconds: float = 3.0,
    ) -> AuthStatus:
        """
        Wait for the user to complete an interactive sign-in.

        Polls the page rather than driving it: nothing is typed, clicked,
        or submitted. The user signs in at their own pace in the visible
        browser window and the session is detected when it appears.

        Args:
            timeout_seconds: How long to wait before giving up. A
                non-positive value performs a single check.
            poll_interval_seconds: Delay between checks.

        Returns:
            Final authentication snapshot.
        """
        status = await self.check_auth()
        if status.is_ready or timeout_seconds <= 0:
            return status

        self._record_auth(
            AuthState.LOGIN_REQUIRED,
            f"Waiting up to {timeout_seconds:.0f}s for manual sign-in",
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        while loop.time() < deadline:
            await asyncio.sleep(poll_interval_seconds)

            status = await self.check_auth()
            if status.is_ready:
                self._logger.info(
                    "%s sign-in detected; continuing automatically",
                    self._site.display_name,
                )
                return status

            if status.state is AuthState.CAPTCHA_REQUIRED:
                return status

        self._logger.warning(
            "%s was not signed in within %.0fs; leaving it paused",
            self._site.display_name,
            timeout_seconds,
        )
        return status

    def _record_auth(
        self,
        state: AuthState,
        detail: str = "",
        action: Optional[str] = None,
    ) -> AuthStatus:
        """
        Store an authentication state, logging any transition.

        Args:
            state: Newly observed state.
            detail: Explanation of the observation.
            action: Operator instruction, when action is required.

        Returns:
            Authentication snapshot.
        """
        if state is not self._auth_state:
            self._logger.info(
                "%s auth state: %s -> %s%s",
                self._site.display_name,
                self._auth_state.value,
                state.value,
                f" ({detail})" if detail else "",
            )
            self._auth_state = state

        return AuthStatus(
            provider=self._site.name,
            state=state,
            display_name=self._site.display_name,
            detail=detail,
            action=action,
        )

    async def send_prompt(
        self,
        prompt: str,
        new_conversation: bool = False,
    ) -> int:
        """
        Submit a prompt without waiting for the answer.

        Exposed separately from :meth:`wait_for_response` so callers can
        fan out submissions across providers before collecting any
        answers. Most callers should prefer :meth:`ask`, which combines
        both with retry and timing.

        Args:
            prompt: Prompt text.
            new_conversation: Whether to start a fresh conversation first.
                Defaults to False so context is preserved.

        Returns:
            Assistant message count before submission, to be passed to
            :meth:`wait_for_response` as the baseline.

        Raises:
            WorkerError: If submission fails.
        """
        page = await self._require_page()

        if new_conversation:
            await self._open_new_conversation(page)
            self._conversation.reset()

        await self._require_composer(page)
        baseline = await page.locator(
            self._site.assistant_message_selector
        ).count()

        self._logger.info(
            "Prompt dispatched to %s (%d chars, turn %d)",
            self._site.display_name,
            len(prompt),
            self._conversation.turns + 1,
        )
        await self._submit_prompt(page, prompt)
        return baseline

    async def wait_for_response(self, baseline: int) -> str:
        """
        Wait for and extract the answer to a submitted prompt.

        Args:
            baseline: Assistant message count returned by
                :meth:`send_prompt`.

        Returns:
            Extracted answer text.

        Raises:
            WorkerError: If the answer never completes or is empty.
        """
        page = await self._require_page()
        await self._wait_for_response(page, baseline)
        answer = await self._extract_answer(page)
        self._logger.info(
            "Response received from %s (%d chars)",
            self._site.display_name,
            len(answer),
        )
        return answer

    async def continue_conversation(self, prompt: str) -> WorkerResponse:
        """
        Send a follow-up prompt inside the existing conversation.

        The page is not reloaded and no new conversation is started, so
        the provider retains everything it has already said. This is the
        mechanism behind multi-round debate.

        Args:
            prompt: Follow-up prompt text.

        Returns:
            Structured worker response.

        Raises:
            WorkerError: If the worker has not been started.
        """
        self._logger.info(
            "Follow-up question sent to %s (turn %d)",
            self._site.display_name,
            self._conversation.turns + 1,
        )
        return await self.ask(
            WorkerQuery(prompt=prompt, new_conversation=False)
        )

    async def restart(self) -> None:
        """
        Recover this provider by reopening its tab.

        Used by the session manager when health monitoring finds a dead
        tab. Only this provider is affected; the browser and the other
        providers keep running.

        Raises:
            WorkerError: If the provider cannot be made ready again.
        """
        self._logger.warning(
            "Restarting %s provider tab", self._site.display_name
        )
        try:
            await self._cleanup()
        except Exception as exc:
            self._logger.warning(
                "Cleanup during %s restart failed: %s",
                self._site.display_name,
                exc,
            )

        self._conversation.reset()
        self._state = WorkerState.CREATED
        await self.start()
        self._logger.info("Provider recovered: %s", self._site.display_name)

    async def reset_conversation(self) -> None:
        """
        Start a fresh conversation with this provider.

        Navigates to the provider's base URL, discarding conversational
        context. Called on explicit user request or when accumulated
        context exceeds the configured budget.

        Raises:
            WorkerError: If navigation fails.
        """
        page = await self._require_page()
        await self._open_new_conversation(page)
        await self._require_composer(page)
        self._conversation.reset()
        self._logger.info(
            "Conversation reset for %s", self._site.display_name
        )

    async def _prepare(self) -> None:
        """
        Open the provider tab and verify the composer is usable.

        Raises:
            WorkerError: If navigation fails or a login wall is detected.
        """
        self._logger.info(
            "Opening %s tab at %s",
            self._site.display_name,
            self._site.base_url,
        )
        self._page = await self._tabs.open_tab(
            url=self._site.base_url,
            reuse_existing=False,
            wait_until="domcontentloaded",
            timeout_seconds=self._site.navigation_timeout_seconds,
        )
        # Attached before stabilization so console output from the
        # problematic load is captured, not just what follows a failure.
        self._console.attach(self._page)
        await self._require_composer(self._page)
        self._logger.info("%s tab ready", self._site.display_name)

    async def _execute(self, query: WorkerQuery) -> str:
        """
        Submit one prompt and return the complete answer.

        Args:
            query: Query to submit.

        Returns:
            Assistant answer extracted as Markdown.

        Raises:
            WorkerError: If any stage fails or the answer never completes.
        """
        page = await self._require_page()

        if query.new_conversation:
            await self._open_new_conversation(page)
            self._conversation.reset()

        await self._require_composer(page)

        prompt = query.prompt
        if query.context:
            prompt = f"{query.context.strip()}\n\n{query.prompt}"

        baseline = await page.locator(
            self._site.assistant_message_selector
        ).count()

        self._logger.info(
            "Prompt dispatched to %s (%d chars, turn %d, continuing=%s)",
            self._site.display_name,
            len(prompt),
            self._conversation.turns + 1,
            not query.new_conversation,
        )
        await self._submit_prompt(page, prompt)

        await self._wait_for_response(page, baseline)
        answer = await self._extract_answer(page)

        self._conversation.record_turn(prompt, answer)
        self._capture_conversation_id(page)
        self._logger.info(
            "Response received from %s (%d chars, %s)",
            self._site.display_name,
            len(answer),
            self._conversation.describe(),
        )
        return answer

    def _capture_conversation_id(self, page: Page) -> None:
        """
        Record the provider's conversation identifier when discoverable.

        Most providers move the tab to a per-conversation URL once a
        thread exists. The trailing path segment is stored for
        traceability; failure to determine it is not an error.

        Args:
            page: Provider page.
        """
        try:
            url = page.url
        except Exception:
            return

        if not url:
            return

        segment = url.rstrip("/").rsplit("/", 1)[-1]
        if segment and segment not in {"new", "app", "chat"}:
            self._conversation.conversation_id = segment

    async def _recover(self) -> None:
        """
        Recover after a failed attempt.

        Reloads the provider tab; if the tab is gone, opens a new one.
        """
        try:
            if self._page is not None and not self._page.is_closed():
                await self._page.reload(
                    wait_until="domcontentloaded",
                    timeout=self._site.navigation_timeout_seconds * 1000,
                )
                return
        except Exception as exc:
            self._logger.warning(
                "%s tab reload failed: %s", self._site.display_name, exc
            )

        self._page = await self._tabs.open_tab(
            url=self._site.base_url,
            reuse_existing=False,
            wait_until="domcontentloaded",
            timeout_seconds=self._site.navigation_timeout_seconds,
        )

    async def _is_site_ready(self) -> bool:
        """
        Check whether the provider is loaded with a usable composer.

        Returns:
            True when the composer is visible.
        """
        if self._page is None or self._page.is_closed():
            return False
        try:
            return await self._page.locator(
                self._site.composer_selector
            ).first.is_visible()
        except Exception:
            return False

    async def _cleanup(self) -> None:
        """Close the worker's tab during shutdown."""
        if self._page is not None and not self._page.is_closed():
            await self._tabs.close_tab(self._page)
        self._page = None

    async def _require_page(self) -> Page:
        """
        Return the worker's page, reopening it if necessary.

        The tab is deliberately not brought to the front: page-scoped
        locators do not need focus, and activating tabs would make
        concurrent multi-provider execution race.

        Returns:
            Usable Playwright page.

        Raises:
            WorkerError: If a page cannot be obtained.
        """
        if self._page is None or self._page.is_closed():
            self._page = await self._tabs.open_tab(
                url=self._site.base_url,
                reuse_existing=False,
                wait_until="domcontentloaded",
                timeout_seconds=self._site.navigation_timeout_seconds,
            )
        return self._page

    async def _open_new_conversation(self, page: Page) -> None:
        """
        Navigate to a fresh conversation.

        Args:
            page: Provider page.

        Raises:
            WorkerError: If navigation fails.
        """
        try:
            await page.goto(
                self._site.base_url,
                wait_until="domcontentloaded",
                timeout=self._site.navigation_timeout_seconds * 1000,
            )
        except Exception as exc:
            raise WorkerError(
                f"Failed to open a new {self._site.display_name} "
                f"conversation: {exc}",
                code=f"{self._site.name.upper()}_NAVIGATION_FAILED",
            ) from exc

    async def _require_composer(self, page: Page) -> None:
        """
        Wait for the page to settle, then require a usable prompt input.

        No selector is searched for until the page has stopped navigating.
        Searching during a redirect chain is what produced the original
        reload-loop failure: the search raced the redirects, never
        settled, and reported a misleading "composer not found" timeout
        instead of the real cause.

        Args:
            page: Provider page.

        Raises:
            ProviderChallengeError: If a verification challenge is shown.
            ProviderAuthError: If a sign-in screen is shown.
            WorkerError: If the page loops, never settles, or settles
                without exposing any usable input.
        """
        result = await self._stabilizer.stabilize(
            page, timeout_seconds=self._site.ready_timeout_seconds
        )
        self._last_page_state = result.state
        self._logger.info(
            "%s page stabilization: %s",
            self._site.display_name,
            result.describe(),
        )

        if result.ok:
            self._active_input_selector = result.matched_selector
            return

        # A challenge is checked first: it often sits in front of a
        # sign-in screen, and it is the blocking condition.
        if result.state is PageState.CHALLENGE_PAGE:
            self._record_auth(
                AuthState.CAPTCHA_REQUIRED, result.reason
            )
            raise ProviderChallengeError(
                challenge_prompt(self._site.display_name),
                code=f"{self._site.name.upper()}_CHALLENGE",
            )

        if result.state is PageState.LOGIN_PAGE:
            # Not a WorkerError: an unauthenticated provider is not a
            # broken one, and treating it as broken would trigger tab
            # restarts that destroy the sign-in page.
            self._record_auth(AuthState.LOGIN_REQUIRED, result.reason)
            raise ProviderAuthError(
                login_prompt(
                    self._site.display_name, self._site.base_url
                ),
                code=f"{self._site.name.upper()}_LOGIN_REQUIRED",
            )

        if result.state is PageState.RELOAD_LOOP:
            # Reported with its diagnosed cause rather than as a generic
            # timeout, and never "fixed" by another reload.
            if result.failure is StabilizationFailure.AUTH_EXPIRED:
                self._record_auth(AuthState.LOGIN_REQUIRED, result.reason)
                raise ProviderAuthError(
                    f"{expiry_notice(self._site.display_name)}\n"
                    f"{result.reason}",
                    code=f"{self._site.name.upper()}_RELOAD_LOOP_AUTH",
                )
            if result.failure is StabilizationFailure.CHALLENGE:
                self._record_auth(
                    AuthState.CAPTCHA_REQUIRED, result.reason
                )
                raise ProviderChallengeError(
                    challenge_prompt(self._site.display_name),
                    code=f"{self._site.name.upper()}_RELOAD_LOOP_CHALLENGE",
                )

        await self._report_failure(page, result)

        raise WorkerError(
            f"{self._site.display_name} did not reach a usable state: "
            f"{result.reason}",
            code=(
                f"{self._site.name.upper()}_"
                f"{(result.failure.value if result.failure else 'unknown').upper()}"
            ),
        )

    async def _report_failure(
        self,
        page: Page,
        result: StabilizationResult,
    ) -> None:
        """
        Capture and print diagnostics for a page that never became usable.

        Args:
            page: Provider page.
            result: Stabilization verdict.
        """
        try:
            diagnostics = await capture_diagnostics(
                page=page,
                provider=self._site.display_name,
                reason=result.reason,
                output_dir=self._diagnostics_dir,
                state=result.state.value,
                auth_state=self._auth_state.value,
                console=self._console,
            )
        except Exception as exc:
            self._logger.warning(
                "Diagnostics capture failed for %s: %s",
                self._site.display_name,
                exc,
            )
            return

        print("\n" + diagnostics.render() + "\n")
        self._logger.error(
            "%s failed to become usable: %s",
            self._site.display_name,
            result.reason,
        )

    async def _challenge_visible(self, page: Page) -> bool:
        """
        Check whether a CAPTCHA or bot-detection challenge is displayed.

        Args:
            page: Provider page.

        Returns:
            True when a challenge indicator is visible.
        """
        if not self._site.challenge_selector:
            return False
        try:
            return await page.locator(
                self._site.challenge_selector
            ).first.is_visible()
        except Exception:
            return False

    async def _login_wall_visible(self, page: Page) -> bool:
        """
        Check whether a login wall is displayed.

        Args:
            page: Provider page.

        Returns:
            True when a login indicator is visible.
        """
        if not self._site.login_wall_selector:
            return False
        try:
            return await page.locator(
                self._site.login_wall_selector
            ).first.is_visible()
        except Exception:
            return False

    async def _submit_prompt(self, page: Page, prompt: str) -> None:
        """
        Enter the prompt into the composer and send it.

        Long prompts are pasted for speed and reliability; short prompts
        are typed with human-like delays.

        Args:
            page: Provider page.
            prompt: Full prompt text.

        Raises:
            WorkerError: If input or submission fails.
        """
        # The selector that actually matched during stabilization is used,
        # so a provider whose markup differs from its primary selector
        # still receives input at the element that was verified usable.
        selector = (
            self._active_input_selector or self._site.composer_selector
        )

        if len(prompt) > self._site.paste_threshold_chars:
            await self._keyboard.paste_text(
                page=page,
                text=prompt,
                selector=selector,
            )
        else:
            await self._keyboard.type_text(
                page=page,
                text=prompt,
                selector=selector,
                clear_first=True,
            )

        if self._site.submit_delay_seconds:
            await asyncio.sleep(self._site.submit_delay_seconds)

        if await self._click_send(page):
            return

        await self._keyboard.press_key(
            page=page,
            key="Enter",
            selector=selector,
        )

    async def _click_send(self, page: Page) -> bool:
        """
        Attempt to submit by clicking the send affordance.

        Args:
            page: Provider page.

        Returns:
            True when the send button was clicked successfully.
        """
        if not self._site.send_button_selector:
            return False
        try:
            button = page.locator(self._site.send_button_selector).first
            await button.wait_for(state="visible", timeout=5000)
            await button.click()
            return True
        except Exception:
            return False

    async def _wait_for_response(self, page: Page, baseline: int) -> None:
        """
        Wait until a new assistant answer is complete.

        Completion requires a new assistant message beyond ``baseline``,
        no visible streaming indicator, and identical message text across
        consecutive stability polls.

        Args:
            page: Provider page.
            baseline: Assistant message count before submission.

        Raises:
            WorkerError: If no complete answer appears within the timeout.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._site.response_timeout_seconds
        assistant = self._site.assistant_message_selector
        last_text: Optional[str] = None
        stable = 0

        while loop.time() < deadline:
            await asyncio.sleep(self._site.poll_interval_seconds)

            try:
                count = await page.locator(assistant).count()
                if count <= baseline:
                    continue

                streaming = await self._is_streaming(page)
                text = await page.locator(assistant).nth(-1).inner_text()
            except Exception:
                continue

            if streaming or not text.strip():
                last_text = text
                stable = 0
                continue

            if text == last_text:
                stable += 1
                if stable >= self._site.stability_checks:
                    return
            else:
                stable = 0
                last_text = text

        raise WorkerError(
            f"{self._site.display_name} response did not complete within "
            f"{self._site.response_timeout_seconds}s",
            code=f"{self._site.name.upper()}_RESPONSE_TIMEOUT",
        )

    async def _is_streaming(self, page: Page) -> bool:
        """
        Check whether a response is still being generated.

        Args:
            page: Provider page.

        Returns:
            True when the streaming indicator is visible. False when the
            provider defines no indicator, in which case text stability
            alone decides completion.
        """
        if not self._site.stop_button_selector:
            return False
        try:
            return await page.locator(
                self._site.stop_button_selector
            ).first.is_visible()
        except Exception:
            return False

    async def _extract_answer(self, page: Page) -> str:
        """
        Extract the latest assistant answer as Markdown.

        Args:
            page: Provider page.

        Returns:
            Answer text.

        Raises:
            WorkerError: If extraction yields no content.
        """
        selector = f"{self._site.assistant_message_selector} >> nth=-1"

        try:
            answer = await self._extractor.extract_markdown(
                page=page,
                selector=selector,
            )
        except Exception:
            answer = await self._extractor.extract_text(
                page=page,
                selector=selector,
            )

        if not answer.strip():
            raise WorkerError(
                f"{self._site.display_name} answer extraction produced "
                f"empty content",
                code=f"{self._site.name.upper()}_EMPTY_ANSWER",
            )
        return answer.strip()


__all__ = [
    "WebChatWorker",
]
