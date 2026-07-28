"""
Generic page extraction utilities.

Extractor reads plain text, HTML, Markdown, page titles, and code blocks from
Playwright pages. It deliberately avoids AI-specific parsing and keeps output
formats useful for any browser automation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from playwright.async_api import Page
from pydantic import BaseModel, Field

from src.exceptions import BrowserError


class ExtractorConfig(BaseModel):
    """
    Configuration for extraction operations.

    Attributes:
        selector_timeout_seconds: Timeout for selector-based extraction.
    """

    selector_timeout_seconds: float = Field(default=10.0, gt=0)


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """
    Extracted code block from a page.

    Attributes:
        index: Zero-based index in document order.
        text: Code text.
        language: Detected language, if available.
        source: Element source type, such as ``pre code`` or ``code``.
    """

    index: int
    text: str
    language: Optional[str]
    source: str


class Extractor:
    """
    Extracts generic content from Playwright pages.

    Extraction can target the entire page or a selector. Markdown conversion
    is intentionally conservative and based on common semantic HTML tags.
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
    ) -> None:
        """
        Initialize the extractor.

        Args:
            config: Optional extraction configuration.
        """
        self._config = config or ExtractorConfig()

    async def extract_title(self, page: Page) -> str:
        """
        Extract the current page title.

        Args:
            page: Playwright page to read.

        Returns:
            Page title.

        Raises:
            BrowserError: If Playwright cannot read the title.
        """
        try:
            return await page.title()
        except Exception as exc:
            raise BrowserError(
                f"Failed to extract page title: {exc}",
                code="BROWSER_EXTRACT_TITLE_FAILED",
            ) from exc

    async def extract_text(
        self,
        page: Page,
        selector: Optional[str] = None,
    ) -> str:
        """
        Extract plain visible text.

        Args:
            page: Playwright page to read.
            selector: Optional selector limiting extraction scope.

        Returns:
            Visible text content.

        Raises:
            BrowserError: If extraction fails.
        """
        try:
            if selector is not None:
                locator = page.locator(selector).first
                await locator.wait_for(
                    state="attached",
                    timeout=self._timeout_ms(),
                )
                return (await locator.inner_text()).strip()

            return (
                await page.locator("body").inner_text(
                    timeout=self._timeout_ms()
                )
            ).strip()
        except Exception as exc:
            raise BrowserError(
                f"Failed to extract text: {exc}",
                code="BROWSER_EXTRACT_TEXT_FAILED",
            ) from exc

    async def extract_html(
        self,
        page: Page,
        selector: Optional[str] = None,
    ) -> str:
        """
        Extract HTML from the page or a selected element.

        Args:
            page: Playwright page to read.
            selector: Optional selector limiting extraction scope.

        Returns:
            HTML string.

        Raises:
            BrowserError: If extraction fails.
        """
        try:
            if selector is not None:
                locator = page.locator(selector).first
                await locator.wait_for(
                    state="attached",
                    timeout=self._timeout_ms(),
                )
                return await locator.evaluate("element => element.outerHTML")

            return await page.content()
        except Exception as exc:
            raise BrowserError(
                f"Failed to extract HTML: {exc}",
                code="BROWSER_EXTRACT_HTML_FAILED",
            ) from exc

    async def extract_markdown(
        self,
        page: Page,
        selector: Optional[str] = None,
    ) -> str:
        """
        Extract a Markdown representation of page content.

        Args:
            page: Playwright page to read.
            selector: Optional selector limiting extraction scope.

        Returns:
            Markdown text.

        Raises:
            BrowserError: If extraction fails.
        """
        try:
            locator = page.locator(selector or "body").first
            await locator.wait_for(
                state="attached",
                timeout=self._timeout_ms(),
            )
            markdown = await locator.evaluate(_MARKDOWN_SCRIPT)
            return str(markdown).strip()
        except Exception as exc:
            raise BrowserError(
                f"Failed to extract Markdown: {exc}",
                code="BROWSER_EXTRACT_MARKDOWN_FAILED",
            ) from exc

    async def extract_code_blocks(
        self,
        page: Page,
        selector: Optional[str] = None,
    ) -> list[CodeBlock]:
        """
        Extract code blocks from ``pre`` and ``code`` elements.

        Args:
            page: Playwright page to read.
            selector: Optional selector limiting extraction scope.

        Returns:
            List of code blocks in document order.

        Raises:
            BrowserError: If extraction fails.
        """
        try:
            locator = page.locator(selector or "body").first
            await locator.wait_for(
                state="attached",
                timeout=self._timeout_ms(),
            )
            raw_blocks = await locator.evaluate(_CODE_BLOCK_SCRIPT)
            return [
                CodeBlock(
                    index=int(block["index"]),
                    text=str(block["text"]),
                    language=(
                        str(block["language"])
                        if block.get("language")
                        else None
                    ),
                    source=str(block["source"]),
                )
                for block in raw_blocks
            ]
        except Exception as exc:
            raise BrowserError(
                f"Failed to extract code blocks: {exc}",
                code="BROWSER_EXTRACT_CODE_BLOCKS_FAILED",
            ) from exc

    def _timeout_ms(self) -> float:
        """Return selector timeout in Playwright milliseconds."""
        return self._config.selector_timeout_seconds * 1000


_MARKDOWN_SCRIPT = """
(root) => {
    const blockTags = new Set([
        "ARTICLE", "ASIDE", "DIV", "FOOTER", "HEADER", "MAIN",
        "NAV", "SECTION"
    ]);

    const normalize = (text) => text
        .replace(/[ \\t\\f\\v]+/g, " ")
        .replace(/\\n{3,}/g, "\\n\\n")
        .trim();

    const escapeMarkdown = (text) => text.replace(/([*_`\\[\\]])/g, "\\\\$1");

    const languageFromClass = (element) => {
        for (const className of element.classList || []) {
            if (className.startsWith("language-")) {
                return className.slice("language-".length);
            }
            if (className.startsWith("lang-")) {
                return className.slice("lang-".length);
            }
        }
        return "";
    };

    const childrenMarkdown = (node) => Array.from(node.childNodes)
        .map(child => toMarkdown(child))
        .filter(Boolean)
        .join("");

    const listItems = (element, ordered) => Array.from(element.children)
        .filter(child => child.tagName === "LI")
        .map((child, index) => {
            const marker = ordered ? `${index + 1}. ` : "- ";
            return marker + normalize(childrenMarkdown(child));
        })
        .join("\\n");

    const tableMarkdown = (table) => {
        const rows = Array.from(table.querySelectorAll("tr")).map(row =>
            Array.from(row.children).map(cell => normalize(cell.innerText || ""))
        );
        if (!rows.length) {
            return "";
        }
        const header = rows[0];
        const separator = header.map(() => "---");
        const body = rows.slice(1);
        return [header, separator, ...body]
            .map(row => `| ${row.join(" | ")} |`)
            .join("\\n");
    };

    const toMarkdown = (node) => {
        if (node.nodeType === Node.TEXT_NODE) {
            return escapeMarkdown(node.textContent || "");
        }

        if (node.nodeType !== Node.ELEMENT_NODE) {
            return "";
        }

        const element = node;
        const tag = element.tagName;
        const text = normalize(childrenMarkdown(element));

        if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") {
            return "";
        }
        if (/^H[1-6]$/.test(tag)) {
            return `\\n\\n${"#".repeat(Number(tag[1]))} ${text}\\n\\n`;
        }
        if (tag === "P") {
            return `\\n\\n${text}\\n\\n`;
        }
        if (tag === "BR") {
            return "\\n";
        }
        if (tag === "STRONG" || tag === "B") {
            return text ? `**${text}**` : "";
        }
        if (tag === "EM" || tag === "I") {
            return text ? `*${text}*` : "";
        }
        if (tag === "A") {
            const href = element.getAttribute("href");
            return href ? `[${text}](${href})` : text;
        }
        if (tag === "UL" || tag === "OL") {
            return `\\n\\n${listItems(element, tag === "OL")}\\n\\n`;
        }
        if (tag === "BLOCKQUOTE") {
            return "\\n\\n" + text.split("\\n").map(line => `> ${line}`).join("\\n") + "\\n\\n";
        }
        if (tag === "PRE") {
            const code = element.querySelector("code") || element;
            const language = languageFromClass(code);
            return `\\n\\n\\`\\`\\`${language}\\n${code.innerText || ""}\\n\\`\\`\\`\\n\\n`;
        }
        if (tag === "CODE") {
            return `\\`${element.innerText || ""}\\``;
        }
        if (tag === "TABLE") {
            return `\\n\\n${tableMarkdown(element)}\\n\\n`;
        }
        if (blockTags.has(tag)) {
            return `\\n${text}\\n`;
        }
        return text;
    };

    return normalize(toMarkdown(root));
}
"""


_CODE_BLOCK_SCRIPT = """
(root) => {
    const languageFromClass = (element) => {
        for (const className of element.classList || []) {
            if (className.startsWith("language-")) {
                return className.slice("language-".length);
            }
            if (className.startsWith("lang-")) {
                return className.slice("lang-".length);
            }
        }
        return null;
    };

    const blocks = [];
    const seen = new Set();
    const addBlock = (element, source) => {
        if (!element || seen.has(element)) {
            return;
        }
        seen.add(element);
        const text = element.innerText || element.textContent || "";
        if (!text.trim()) {
            return;
        }
        blocks.push({
            index: blocks.length,
            text,
            language: languageFromClass(element),
            source
        });
    };

    root.querySelectorAll("pre code").forEach(element => addBlock(element, "pre code"));
    root.querySelectorAll("pre").forEach(element => {
        if (!element.querySelector("code")) {
            addBlock(element, "pre");
        }
    });
    root.querySelectorAll("code").forEach(element => {
        if (!element.closest("pre")) {
            addBlock(element, "code");
        }
    });

    return blocks;
}
"""


__all__ = [
    "CodeBlock",
    "Extractor",
    "ExtractorConfig",
]
