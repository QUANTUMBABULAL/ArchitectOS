"""
Tab management utilities for browser sessions.

TabManager provides the basic page operations needed by future automation:
opening, closing, switching, reusing, finding, and listing tabs. It works
only with BrowserSession state and Playwright pages, so it remains reusable
across any site or worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
from urllib.parse import urlparse

from playwright.async_api import Page

from src.exceptions import BrowserError
from src.logger import get_logger

from .browser_session import BrowserSession


@dataclass(frozen=True, slots=True)
class TabInfo:
    """
    Snapshot of one browser tab.

    Attributes:
        index: Zero-based tab index within the current session.
        url: Current page URL.
        title: Current page title.
        is_active: Whether this tab is the session's active page.
        is_closed: Whether Playwright reports the tab as closed.
    """

    index: int
    url: str
    title: str
    is_active: bool
    is_closed: bool


TabTarget = Union[Page, int, str]


class TabManager:
    """
    Manages pages inside a browser session.

    The manager updates BrowserSession page tracking after each operation and
    never owns the Playwright lifecycle itself. This keeps tab operations
    composable with BrowserManager and other controllers.
    """

    def __init__(self, session: BrowserSession) -> None:
        """
        Initialize the tab manager.

        Args:
            session: Browser session whose pages will be managed.
        """
        self._session = session
        self._logger = get_logger(__name__)

    async def open_tab(
        self,
        url: Optional[str] = None,
        reuse_existing: bool = False,
        wait_until: str = "load",
        timeout_seconds: Optional[float] = None,
    ) -> Page:
        """
        Open a new tab and optionally navigate it.

        Args:
            url: Optional URL to navigate to after opening.
            reuse_existing: Reuse an existing tab with the same URL when found.
            wait_until: Playwright navigation load state.
            timeout_seconds: Optional navigation timeout in seconds.

        Returns:
            Opened or reused page.

        Raises:
            BrowserError: If the tab cannot be opened or navigated.
        """
        if reuse_existing and url:
            reused_page = await self.reuse_existing_tab(url=url, exact=True)
            if reused_page is not None:
                return reused_page

        try:
            page = await self._session.context.new_page()
            self._session.add_page(page)

            if url:
                goto_kwargs: dict[str, object] = {"wait_until": wait_until}
                if timeout_seconds is not None:
                    goto_kwargs["timeout"] = timeout_seconds * 1000
                await page.goto(url, **goto_kwargs)

            self._logger.debug("Opened tab: %s", page.url)
            return page
        except Exception as exc:
            raise BrowserError(
                f"Failed to open browser tab: {exc}",
                code="BROWSER_TAB_OPEN_FAILED",
            ) from exc

    async def acquire_page(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout_seconds: Optional[float] = None,
    ) -> Page:
        """
        Obtain a usable page for ``url``, creating one only if required.

        The acquisition order is deliberate and is what keeps the browser
        free of stray tabs:

        1. **Reuse** an open tab already on the same host. A provider that
           already has its tab is never given a second one, and the page
           is *not* re-navigated, so an in-progress conversation survives.
        2. **Adopt** an unused blank page. ``launch_persistent_context``
           always opens one ``about:blank`` page at startup; adopting it
           is what stops that page lingering as an empty tab forever.
        3. **Create** a new page only when neither exists.

        Args:
            url: URL identifying the wanted host, navigated to only when a
                page has to be adopted or created.
            wait_until: Playwright navigation load state.
            timeout_seconds: Optional navigation timeout in seconds.

        Returns:
            Page ready for use.

        Raises:
            BrowserError: If no page can be obtained or navigated.
        """
        existing = self.find_tab_by_host(url)
        if existing is not None:
            self._logger.debug("Reusing existing tab for %s", url)
            self._session.add_page(existing)
            return existing

        goto_kwargs: dict[str, object] = {"wait_until": wait_until}
        if timeout_seconds is not None:
            goto_kwargs["timeout"] = timeout_seconds * 1000

        blank = self.find_blank_page()
        if blank is not None:
            self._logger.debug("Adopting blank page for %s", url)
            try:
                await blank.goto(url, **goto_kwargs)
                self._session.add_page(blank)
                return blank
            except Exception as exc:
                raise BrowserError(
                    f"Failed to navigate the blank tab to {url}: {exc}",
                    code="BROWSER_TAB_NAVIGATION_FAILED",
                ) from exc

        return await self.open_tab(
            url=url,
            reuse_existing=False,
            wait_until=wait_until,
            timeout_seconds=timeout_seconds,
        )

    def find_tab_by_host(self, url: str) -> Optional[Page]:
        """
        Find an open tab on the same host as ``url``.

        Host matching, rather than exact-URL matching, is what allows a
        provider's tab to be recognized after it has navigated into a
        conversation-specific URL.

        Args:
            url: URL whose host identifies the wanted tab.

        Returns:
            Matching page, or None.
        """
        wanted = _host_of(url)
        if not wanted:
            return None
        for page in self._session.sync_pages():
            if _host_of(page.url) == wanted:
                return page
        return None

    def find_blank_page(self) -> Optional[Page]:
        """
        Find an open page that carries no content.

        Returns:
            A blank page, or None when every page is in use.
        """
        for page in self._session.sync_pages():
            if _is_blank(page.url):
                return page
        return None

    async def close_blank_pages(self, keep_last: bool = True) -> int:
        """
        Close leftover blank tabs.

        Called after providers have their tabs so the window never shows
        an empty ``about:blank`` the user did not ask for. Chrome closes
        the whole window when its last tab closes, so by default one page
        is always kept when nothing else is open.

        Args:
            keep_last: Never close the final remaining page.

        Returns:
            Number of tabs closed.
        """
        pages = self._session.sync_pages()
        blanks = [page for page in pages if _is_blank(page.url)]
        if not blanks:
            return 0

        if keep_last and len(blanks) == len(pages):
            blanks = blanks[1:]

        closed = 0
        for page in blanks:
            try:
                await page.close()
                closed += 1
            except Exception as exc:
                self._logger.debug("Could not close blank tab: %s", exc)

        if closed:
            self._session.sync_pages()
            self._logger.info("Closed %d unused blank tab(s)", closed)
        return closed

    async def close_tab(
        self,
        target: Optional[TabTarget] = None,
    ) -> None:
        """
        Close a tab.

        Args:
            target: Page, index, or URL identifying the tab. Defaults to the
                active page.

        Raises:
            BrowserError: If the target tab cannot be resolved or closed.
        """
        page = self._resolve_page(target)

        try:
            await page.close()
            self._session.remove_page(page)

            if self._session.active_page is not None:
                await self._session.active_page.bring_to_front()

            self._logger.debug("Closed tab")
        except Exception as exc:
            raise BrowserError(
                f"Failed to close browser tab: {exc}",
                code="BROWSER_TAB_CLOSE_FAILED",
            ) from exc

    async def switch_tab(self, target: TabTarget) -> Page:
        """
        Switch the active tab.

        Args:
            target: Page, index, or URL identifying the tab.

        Returns:
            Newly active page.

        Raises:
            BrowserError: If the target tab cannot be resolved.
        """
        page = self._resolve_page(target)

        try:
            await page.bring_to_front()
            self._session.set_active_page(page)
            self._logger.debug("Switched to tab: %s", page.url)
            return page
        except Exception as exc:
            raise BrowserError(
                f"Failed to switch browser tab: {exc}",
                code="BROWSER_TAB_SWITCH_FAILED",
            ) from exc

    async def reuse_existing_tab(
        self,
        url: str,
        exact: bool = True,
    ) -> Optional[Page]:
        """
        Reuse a tab that already matches a URL.

        Args:
            url: URL to search for.
            exact: Require exact URL equality when True; otherwise substring
                matching is used.

        Returns:
            Matching page or None.
        """
        page = self.find_tab_by_url(url=url, exact=exact)
        if page is None:
            return None
        return await self.switch_tab(page)

    def find_tab_by_url(
        self,
        url: str,
        exact: bool = True,
    ) -> Optional[Page]:
        """
        Find an active tab by URL.

        Args:
            url: URL to search for.
            exact: Require exact URL equality when True; otherwise substring
                matching is used.

        Returns:
            Matching page or None.
        """
        for page in self._session.sync_pages():
            page_url = page.url
            if exact and page_url == url:
                return page
            if not exact and url in page_url:
                return page
        return None

    async def list_active_tabs(self) -> list[TabInfo]:
        """
        Return metadata for all active tabs.

        Returns:
            List of tab snapshots.
        """
        tabs: list[TabInfo] = []
        for index, page in enumerate(self._session.sync_pages()):
            title = ""
            if not page.is_closed():
                try:
                    title = await page.title()
                except Exception:
                    title = ""

            tabs.append(
                TabInfo(
                    index=index,
                    url=page.url,
                    title=title,
                    is_active=page is self._session.active_page,
                    is_closed=page.is_closed(),
                )
            )
        return tabs

    def _resolve_page(
        self,
        target: Optional[TabTarget],
    ) -> Page:
        """Resolve a page, index, URL, or default active tab target."""
        pages = self._session.sync_pages()

        if target is None:
            return self._session.get_active_page()

        if isinstance(target, Page):
            if target.is_closed():
                raise BrowserError(
                    "Target tab is already closed",
                    code="BROWSER_TAB_CLOSED",
                )
            if target not in pages:
                raise BrowserError(
                    "Target tab does not belong to this session",
                    code="BROWSER_TAB_NOT_IN_SESSION",
                )
            return target

        if isinstance(target, int):
            if target < 0 or target >= len(pages):
                raise BrowserError(
                    f"Tab index is out of range: {target}",
                    code="BROWSER_TAB_INDEX_INVALID",
                )
            return pages[target]

        page = self.find_tab_by_url(url=target, exact=True)
        if page is None:
            page = self.find_tab_by_url(url=target, exact=False)
        if page is None:
            raise BrowserError(
                f"No tab matched URL: {target}",
                code="BROWSER_TAB_NOT_FOUND",
            )
        return page


_BLANK_URLS: frozenset[str] = frozenset(
    {"", "about:blank", "chrome://newtab/", "chrome://new-tab-page/"}
)


def _host_of(url: str) -> str:
    """
    Extract a comparable host from a URL.

    Args:
        url: URL to inspect.

    Returns:
        Lower-cased host without a leading ``www.``, or an empty string.
    """
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _is_blank(url: str) -> bool:
    """
    Report whether a URL identifies an empty tab.

    Args:
        url: Page URL.

    Returns:
        True when the page holds no site content.
    """
    return (url or "").strip().lower() in _BLANK_URLS


__all__ = [
    "TabInfo",
    "TabManager",
    "TabTarget",
]
