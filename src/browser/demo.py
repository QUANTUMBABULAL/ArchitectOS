"""
Demonstration script for the browser operating system.

The demo starts BrowserManager, launches Google Chrome with a persistent
profile, opens two tabs, navigates to two neutral websites, switches between
tabs, extracts page titles, and closes the browser gracefully.

Usage:
    python -m src.browser.demo --profile-directory Default
    python -m src.browser.demo --user-data-dir "C:\\path\\to\\User Data"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.browser import (
    BrowserLaunchConfig,
    BrowserManager,
    Extractor,
    TabManager,
)
from src.exceptions import BrowserError
from src.logger import configure_logging, get_logger


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the browser demo.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a generic Browser Operating System demo.",
    )
    parser.add_argument(
        "--chrome-executable",
        type=Path,
        default=None,
        help="Optional path to the Google Chrome executable.",
    )
    parser.add_argument(
        "--user-data-dir",
        type=Path,
        default=None,
        help="Optional Chrome user data directory.",
    )
    parser.add_argument(
        "--profile-directory",
        default="Default",
        help="Chrome profile directory inside the user data directory.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Chrome in headless mode.",
    )
    parser.add_argument(
        "--first-url",
        default="https://example.com",
        help="First website to open.",
    )
    parser.add_argument(
        "--second-url",
        default="https://www.python.org",
        help="Second website to open.",
    )
    return parser.parse_args(argv)


async def run_demo(args: argparse.Namespace) -> None:
    """
    Run the browser operating system demonstration.

    Args:
        args: Parsed command-line arguments.

    Raises:
        BrowserError: If browser startup or tab operations fail.
    """
    logger = get_logger(__name__)
    manager = BrowserManager()
    extractor = Extractor()

    launch_config = BrowserLaunchConfig(
        headless=args.headless,
        persistent_context=True,
        executable_path=args.chrome_executable,
        user_data_dir=args.user_data_dir,
        profile_directory=args.profile_directory,
    )

    try:
        logger.info("Starting BrowserManager")
        session = await manager.start(launch_config)
        tabs = TabManager(session)

        logger.info("Opening first tab: %s", args.first_url)
        first_page = await tabs.open_tab(args.first_url)

        logger.info("Opening second tab: %s", args.second_url)
        second_page = await tabs.open_tab(args.second_url)

        logger.info("Switching to first tab")
        await tabs.switch_tab(first_page)
        first_title = await extractor.extract_title(first_page)

        logger.info("Switching to second tab")
        await tabs.switch_tab(second_page)
        second_title = await extractor.extract_title(second_page)

        print(f"First tab title:  {first_title}")
        print(f"Second tab title: {second_title}")

        health = await manager.health_check()
        print(f"Browser healthy:  {health.healthy}")

    finally:
        logger.info("Closing BrowserManager")
        await manager.stop()


def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point for the browser demo.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Process exit code.
    """
    configure_logging(level=logging.INFO)
    args = parse_args(argv)

    try:
        asyncio.run(run_demo(args))
        return 0
    except BrowserError as exc:
        get_logger(__name__).error("Browser demo failed: %s", exc)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
