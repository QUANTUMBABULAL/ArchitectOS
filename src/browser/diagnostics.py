"""
Failure diagnostics for provider pages.

When a provider does not become usable, a timeout message alone is close
to worthless: it says something did not happen without saying why. This
module captures the evidence needed to answer that question — the URL, the
title, visible error text, a screenshot, an HTML snapshot, and recent
console output — and writes it somewhere durable.

Capture is entirely best-effort. A page that failed to stabilize is often
partly broken, so every step is individually guarded: losing the
screenshot must not cost the console log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.async_api import Page

from src.logger import get_logger

# Selectors that commonly carry a user-visible error or blocking notice.
_ERROR_SELECTORS: tuple[str, ...] = (
    '[role="alert"]',
    ".error-message",
    ".error",
    '[class*="error" i]',
    '[data-testid*="error" i]',
)

_MAX_ERROR_MESSAGES = 5
_MAX_CONSOLE_LINES = 40
_MAX_HTML_CHARS = 400_000


@dataclass
class ConsoleRecorder:
    """
    Collects console output from a page.

    Attached when the tab is created so messages emitted during the
    problematic load are captured, not just those after a failure.

    Attributes:
        provider: Provider the page belongs to.
        limit: Maximum retained lines, oldest dropped first.
        lines: Recorded console lines.
    """

    provider: str
    limit: int = _MAX_CONSOLE_LINES
    lines: list[str] = field(default_factory=list)

    def attach(self, page: Page) -> None:
        """
        Subscribe to console and page-error events.

        Args:
            page: Page to observe.
        """
        try:
            page.on("console", self._on_console)
            page.on("pageerror", self._on_page_error)
        except Exception:
            # Observation is optional; never block startup for it.
            pass

    def _record(self, line: str) -> None:
        """
        Store one line, trimming the oldest when over the limit.

        Args:
            line: Text to record.
        """
        self.lines.append(line)
        if len(self.lines) > self.limit:
            del self.lines[: len(self.lines) - self.limit]

    def _on_console(self, message: object) -> None:
        """
        Record a console message.

        Args:
            message: Playwright console message.
        """
        try:
            self._record(
                f"[{message.type}] {message.text}"  # type: ignore[attr-defined]
            )
        except Exception:
            pass

    def _on_page_error(self, error: object) -> None:
        """
        Record an uncaught page error.

        Args:
            error: Playwright page error.
        """
        try:
            self._record(f"[pageerror] {error}")
        except Exception:
            pass

    def snapshot(self) -> list[str]:
        """
        Return the recorded lines.

        Returns:
            Copy of the console lines.
        """
        return list(self.lines)


@dataclass(frozen=True, slots=True)
class PageDiagnostics:
    """
    Evidence captured when a provider failed to become usable.

    Attributes:
        provider: Provider name.
        reason: Why the provider was considered failed.
        state: Page state at capture time.
        auth_state: Authentication state at capture time.
        url: Current URL.
        title: Page title.
        error_messages: Visible error text found on the page.
        console_lines: Recent console output.
        screenshot_path: Where the screenshot was written.
        html_path: Where the HTML snapshot was written.
        captured_at: When capture ran.
    """

    provider: str
    reason: str
    state: str = ""
    auth_state: str = ""
    url: str = ""
    title: str = ""
    error_messages: tuple[str, ...] = ()
    console_lines: tuple[str, ...] = ()
    screenshot_path: Optional[Path] = None
    html_path: Optional[Path] = None
    captured_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def render(self) -> str:
        """
        Render the diagnostics as an operator-facing report.

        Returns:
            Multi-line report.
        """
        rule = "-" * 66
        lines = [
            rule,
            f"DIAGNOSTICS: {self.provider}",
            rule,
            f"  Reason        : {self.reason}",
            f"  Page state    : {self.state or 'unknown'}",
            f"  Auth state    : {self.auth_state or 'unknown'}",
            f"  URL           : {self.url or 'unknown'}",
            f"  Title         : {self.title or 'unknown'}",
        ]

        if self.error_messages:
            lines.append("  Page errors   :")
            lines.extend(f"      {text}" for text in self.error_messages)

        if self.screenshot_path:
            lines.append(f"  Screenshot    : {self.screenshot_path}")
        if self.html_path:
            lines.append(f"  HTML snapshot : {self.html_path}")

        if self.console_lines:
            lines.append("  Console (last lines):")
            lines.extend(f"      {text}" for text in self.console_lines[-12:])

        lines.append(rule)
        return "\n".join(lines)


async def capture_diagnostics(
    page: Optional[Page],
    provider: str,
    reason: str,
    output_dir: Path,
    state: str = "",
    auth_state: str = "",
    console: Optional[ConsoleRecorder] = None,
) -> PageDiagnostics:
    """
    Capture everything useful about a failed provider page.

    Args:
        page: Page to inspect. None yields a metadata-only report.
        provider: Provider name.
        reason: Why the provider is considered failed.
        output_dir: Directory for screenshot and HTML artefacts.
        state: Page state at failure.
        auth_state: Authentication state at failure.
        console: Optional console recorder attached to the page.

    Returns:
        Captured diagnostics. Never raises; missing pieces are simply
        absent from the report.
    """
    logger = get_logger(__name__)
    console_lines = tuple(console.snapshot()) if console else ()

    if page is None or page.is_closed():
        return PageDiagnostics(
            provider=provider,
            reason=reason,
            state=state,
            auth_state=auth_state,
            console_lines=console_lines,
        )

    url = ""
    title = ""
    errors: list[str] = []
    screenshot_path: Optional[Path] = None
    html_path: Optional[Path] = None

    try:
        url = page.url or ""
    except Exception:
        pass

    try:
        title = await page.title()
    except Exception:
        pass

    for selector in _ERROR_SELECTORS:
        if len(errors) >= _MAX_ERROR_MESSAGES:
            break
        try:
            locator = page.locator(selector)
            for index in range(min(await locator.count(), 3)):
                element = locator.nth(index)
                if not await element.is_visible():
                    continue
                text = (await element.inner_text()).strip()
                if text and text not in errors:
                    errors.append(text[:300])
        except Exception:
            continue

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not create the diagnostics directory %s: %s",
            output_dir,
            exc,
        )
        output_dir = None  # type: ignore[assignment]

    if output_dir is not None:
        candidate = output_dir / f"{provider}-{stamp}.png"
        try:
            await page.screenshot(path=str(candidate), full_page=False)
            screenshot_path = candidate
        except Exception as exc:
            logger.debug("Screenshot capture failed: %s", exc)

        candidate = output_dir / f"{provider}-{stamp}.html"
        try:
            content = await page.content()
            candidate.write_text(
                content[:_MAX_HTML_CHARS], encoding="utf-8"
            )
            html_path = candidate
        except Exception as exc:
            logger.debug("HTML capture failed: %s", exc)

    return PageDiagnostics(
        provider=provider,
        reason=reason,
        state=state,
        auth_state=auth_state,
        url=url,
        title=title,
        error_messages=tuple(errors),
        console_lines=console_lines,
        screenshot_path=screenshot_path,
        html_path=html_path,
    )


__all__ = [
    "ConsoleRecorder",
    "PageDiagnostics",
    "capture_diagnostics",
]
