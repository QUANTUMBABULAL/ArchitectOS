"""
Google Chrome profile discovery and validation utilities.

This module resolves the installed Chrome executable and a persistent user
profile directory for Playwright. It never attempts to automate sign-in;
instead it validates an existing profile so browser sessions can reuse
cookies and local state that the user has already created manually.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field, field_validator

from src.exceptions import BrowserError, ChromeNotFoundError
from src.logger import get_logger

# Directory layout for the dedicated ArchitectOS browser profile:
#     <data_dir>/chrome-profile/ArchitectOS
# The root is separate from the user's own Chrome user data directory so
# ArchitectOS never contends with, or writes into, normal browsing.
AUTOMATION_ROOT_DIRNAME = "chrome-profile"
ARCHITECTOS_PROFILE_NAME = "ArchitectOS"


class ChromeProfileConfig(BaseModel):
    """
    Configuration for resolving a Google Chrome profile.

    Attributes:
        executable_path: Optional explicit path to the Chrome executable.
        user_data_dir: Optional Chrome user data root directory.
        profile_directory: Profile folder inside the user data directory.
        additional_search_paths: Extra executable paths to check before
            platform defaults.
        create_if_missing: Whether a missing user data root or profile
            directory should be created instead of raising. Used for
            dedicated automation profiles, which do not exist until the
            first run.
    """

    executable_path: Optional[Path] = Field(
        default=None,
        description="Explicit path to the Google Chrome executable.",
    )
    user_data_dir: Optional[Path] = Field(
        default=None,
        description="Chrome user data root directory.",
    )
    profile_directory: str = Field(
        default="Default",
        min_length=1,
        description="Profile folder within the Chrome user data directory.",
    )
    additional_search_paths: tuple[Path, ...] = Field(
        default_factory=tuple,
        description="Additional Chrome executable paths to search first.",
    )
    create_if_missing: bool = Field(
        default=False,
        description=(
            "Create the user data root and profile directory when they do "
            "not exist yet."
        ),
    )

    @field_validator(
        "executable_path",
        "user_data_dir",
        mode="before",
    )
    @classmethod
    def expand_optional_path(cls, value: object) -> object:
        """
        Expand user markers in optional filesystem paths.

        Args:
            value: Raw field value supplied to Pydantic.

        Returns:
            Expanded path value or the original value when absent.
        """
        if value is None:
            return value
        return Path(value).expanduser()

    @field_validator("additional_search_paths", mode="before")
    @classmethod
    def expand_search_paths(cls, value: object) -> object:
        """
        Expand user markers for additional Chrome search paths.

        Args:
            value: Raw search path collection.

        Returns:
            Tuple of expanded paths.
        """
        if value is None:
            return tuple()
        return tuple(Path(path).expanduser() for path in value)  # type: ignore[arg-type]

    @field_validator("profile_directory")
    @classmethod
    def validate_profile_directory(cls, value: str) -> str:
        """
        Validate a Chrome profile directory name.

        Args:
            value: Profile directory name.

        Returns:
            Trimmed profile directory name.

        Raises:
            ValueError: If the profile directory is empty.
        """
        profile_directory = value.strip()
        if not profile_directory:
            raise ValueError("profile_directory cannot be empty")
        return profile_directory


@dataclass(frozen=True, slots=True)
class ChromeProfile:
    """
    Resolved Google Chrome executable and persistent profile paths.

    The ``user_data_dir`` is the Chrome user data root passed to
    Playwright's persistent context launcher. ``profile_directory`` is the
    specific Chrome profile folder selected through Chrome's
    ``--profile-directory`` argument.

    Attributes:
        executable_path: Installed Google Chrome executable path.
        user_data_dir: Chrome user data root directory.
        profile_directory: Selected profile directory name.
        profile_path: Full path to the selected profile directory.
    """

    executable_path: Path
    user_data_dir: Path
    profile_directory: str
    profile_path: Path

    @classmethod
    def resolve(
        cls,
        config: Optional[ChromeProfileConfig] = None,
    ) -> "ChromeProfile":
        """
        Resolve and validate a Chrome executable and profile.

        Args:
            config: Optional profile configuration.

        Returns:
            Resolved Chrome profile.

        Raises:
            BrowserError: If Chrome or the requested profile cannot be found.
        """
        profile_config = config or ChromeProfileConfig()
        executable_path = cls.locate_executable(profile_config)
        user_data_dir = (
            profile_config.user_data_dir
            or cls.default_user_data_dir()
        )
        profile_path = cls.validate_profile(
            user_data_dir=user_data_dir,
            profile_directory=profile_config.profile_directory,
            create_if_missing=profile_config.create_if_missing,
        )

        return cls(
            executable_path=executable_path,
            user_data_dir=user_data_dir,
            profile_directory=profile_config.profile_directory,
            profile_path=profile_path,
        )

    @classmethod
    def locate_executable(
        cls,
        config: Optional[ChromeProfileConfig] = None,
    ) -> Path:
        """
        Locate an installed Google Chrome executable.

        Args:
            config: Optional profile configuration with explicit or extra paths.

        Returns:
            Path to the Chrome executable.

        Raises:
            BrowserError: If Chrome cannot be located.
        """
        profile_config = config or ChromeProfileConfig()

        explicit_path = profile_config.executable_path
        if explicit_path is not None:
            return cls._validate_executable(explicit_path)

        env_path = os.getenv("CHROME_EXECUTABLE_PATH")
        if env_path:
            return cls._validate_executable(Path(env_path).expanduser())

        for candidate in cls._candidate_executables(profile_config):
            if candidate.exists() and candidate.is_file():
                return candidate

        for executable_name in cls._which_names():
            resolved = shutil.which(executable_name)
            if resolved:
                return cls._validate_executable(Path(resolved))

        searched = "\n".join(
            f"      {path}"
            for path in cls._candidate_executables(profile_config)
        )
        raise ChromeNotFoundError(
            "Google Chrome could not be found on this system.\n"
            "    ArchitectOS uses your installed Google Chrome rather "
            "than a bundled browser.\n"
            "    Searched:\n"
            f"{searched}\n"
            "    Fix this by either:\n"
            "      - installing Google Chrome from "
            "https://www.google.com/chrome/, or\n"
            "      - setting CHROME_PATH in your .env to the full path of "
            "chrome.exe\n"
            "        (for example: C:\\Program Files\\Google\\Chrome\\"
            "Application\\chrome.exe)",
            code="CHROME_EXECUTABLE_NOT_FOUND",
        )

    @classmethod
    def default_user_data_dir(cls) -> Path:
        """
        Return the platform default Chrome user data directory.

        Returns:
            Default Chrome user data directory for the current platform.

        Raises:
            BrowserError: If the current platform is unsupported.
        """
        if sys.platform.startswith("win"):
            local_app_data = os.getenv("LOCALAPPDATA")
            if local_app_data:
                return Path(local_app_data) / "Google" / "Chrome" / "User Data"

        if sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Google"
                / "Chrome"
            )

        if sys.platform.startswith("linux"):
            return Path.home() / ".config" / "google-chrome"

        raise BrowserError(
            f"Unsupported platform for Chrome profile discovery: {sys.platform}",
            code="CHROME_PLATFORM_UNSUPPORTED",
        )

    @classmethod
    def automation_user_data_dir(
        cls,
        profile_name: str,
        base_dir: Path,
    ) -> Path:
        """
        Return the dedicated Chrome user data root for ArchitectOS.

        Chrome's singleton lock covers the whole user data root, not the
        individual profile folder inside it. Reusing the user's default
        root therefore fails whenever their normal Chrome is open, so
        ArchitectOS gets its own root under the application data
        directory. The named profile lives inside it:

            <data_dir>/chrome-profile/<profile_name>

        Args:
            profile_name: Profile folder name inside the root.
            base_dir: Application data directory.

        Returns:
            Dedicated Chrome user data root.
        """
        return Path(base_dir).expanduser() / AUTOMATION_ROOT_DIRNAME

    @classmethod
    def legacy_user_data_dirs(cls, base_dir: Path) -> list[Path]:
        """
        Return user data roots used by earlier versions.

        Args:
            base_dir: Application data directory.

        Returns:
            Candidate legacy roots, newest layout first.
        """
        root = Path(base_dir).expanduser()
        return [root / "chrome-profiles" / "automation"]

    @classmethod
    def migrate_legacy_profile(
        cls,
        base_dir: Path,
        profile_directory: str = ARCHITECTOS_PROFILE_NAME,
    ) -> Optional[Path]:
        """
        Move a profile from a previous layout into the current one.

        Existing signed-in sessions are valuable: discarding them would
        force the user to authenticate every provider again. When a legacy
        profile exists and the current one does not, it is moved rather
        than copied or deleted, so authentication survives the migration
        and no duplicate is left behind.

        Nothing is ever removed. If the current profile already exists the
        legacy one is left untouched for the user to inspect.

        Args:
            base_dir: Application data directory.
            profile_directory: Profile folder name to migrate into.

        Returns:
            Destination path when a migration happened, otherwise None.
        """
        logger = get_logger(__name__)
        destination_root = cls.automation_user_data_dir(
            profile_name=profile_directory, base_dir=base_dir
        )
        destination = destination_root / profile_directory

        if destination.exists():
            return None

        for legacy_root in cls.legacy_user_data_dirs(base_dir):
            legacy_profile = legacy_root / "Default"
            if not legacy_profile.is_dir():
                continue

            try:
                destination_root.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy_profile), str(destination))
            except OSError as exc:
                logger.warning(
                    "Could not migrate the existing browser profile from "
                    "%s to %s: %s. A new profile will be created and you "
                    "will need to sign in again.",
                    legacy_profile,
                    destination,
                    exc,
                )
                return None

            logger.info(
                "Migrated existing browser profile from %s to %s; "
                "signed-in sessions were preserved",
                legacy_profile,
                destination,
            )
            return destination

        return None

    @classmethod
    def validate_profile(
        cls,
        user_data_dir: Path,
        profile_directory: str,
        create_if_missing: bool = False,
    ) -> Path:
        """
        Validate that a Chrome profile exists.

        Args:
            user_data_dir: Chrome user data root directory.
            profile_directory: Profile directory name inside the root.
            create_if_missing: Create the root and profile directory when
                they are absent instead of raising. Used for dedicated
                automation profiles on their first run.

        Returns:
            Full profile path.

        Raises:
            BrowserError: If the user data root or profile directory is
                absent and creation was not requested, or if creation
                fails.
        """
        expanded_user_data_dir = user_data_dir.expanduser()

        if not expanded_user_data_dir.exists():
            if not create_if_missing:
                raise BrowserError(
                    f"Chrome user data directory does not exist: "
                    f"{expanded_user_data_dir}",
                    code="CHROME_USER_DATA_DIR_NOT_FOUND",
                )
            try:
                expanded_user_data_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise BrowserError(
                    f"Failed to create Chrome user data directory "
                    f"{expanded_user_data_dir}: {exc}",
                    code="CHROME_USER_DATA_DIR_CREATE_FAILED",
                ) from exc

        if not expanded_user_data_dir.is_dir():
            raise BrowserError(
                f"Chrome user data path is not a directory: "
                f"{expanded_user_data_dir}",
                code="CHROME_USER_DATA_DIR_INVALID",
            )

        profile_path = expanded_user_data_dir / profile_directory
        if not profile_path.exists():
            if not create_if_missing:
                raise BrowserError(
                    f"Chrome profile directory does not exist: "
                    f"{profile_path}",
                    code="CHROME_PROFILE_NOT_FOUND",
                )
            try:
                profile_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise BrowserError(
                    f"Failed to create Chrome profile directory "
                    f"{profile_path}: {exc}",
                    code="CHROME_PROFILE_CREATE_FAILED",
                ) from exc

        if not profile_path.is_dir():
            raise BrowserError(
                f"Chrome profile path is not a directory: {profile_path}",
                code="CHROME_PROFILE_INVALID",
            )

        return profile_path

    def launch_args(self, maximized: bool = True) -> list[str]:
        """
        Return Chrome launch arguments for this profile.

        The flags keep the window behaving like an ordinary Chrome
        install: the profile is selected explicitly, first-run and default
        browser interstitials are suppressed, and the automation banner is
        removed so the UI matches what the user expects while signing in.

        Args:
            maximized: Whether to open the window maximized.

        Returns:
            Chrome arguments.
        """
        args = [
            f"--profile-directory={self.profile_directory}",
            "--no-first-run",
            "--no-default-browser-check",
            # Removes the "Chrome is being controlled by automated test
            # software" infobar and the navigator.webdriver flag. Sites
            # that gate sign-in on automation signals are less likely to
            # refuse the session.
            "--disable-blink-features=AutomationControlled",
            "--disable-features=Translate",
        ]
        if maximized:
            args.append("--start-maximized")
        return args

    @classmethod
    def _validate_executable(cls, executable_path: Path) -> Path:
        """Validate a Chrome executable path."""
        expanded_path = executable_path.expanduser()
        if not expanded_path.exists():
            raise BrowserError(
                f"Chrome executable does not exist: {expanded_path}",
                code="CHROME_EXECUTABLE_NOT_FOUND",
            )
        if not expanded_path.is_file():
            raise BrowserError(
                f"Chrome executable path is not a file: {expanded_path}",
                code="CHROME_EXECUTABLE_INVALID",
            )
        return expanded_path

    @classmethod
    def _candidate_executables(
        cls,
        config: ChromeProfileConfig,
    ) -> Iterable[Path]:
        """Yield likely Chrome executable paths for the current platform."""
        for path in config.additional_search_paths:
            yield path

        if sys.platform.startswith("win"):
            program_files = [
                os.getenv("PROGRAMFILES"),
                os.getenv("PROGRAMFILES(X86)"),
                os.getenv("LOCALAPPDATA"),
            ]
            for root in program_files:
                if root:
                    yield Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
            return

        if sys.platform == "darwin":
            yield Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
            return

        if sys.platform.startswith("linux"):
            yield Path("/usr/bin/google-chrome")
            yield Path("/usr/bin/google-chrome-stable")
            yield Path("/opt/google/chrome/chrome")

    @classmethod
    def _which_names(cls) -> tuple[str, ...]:
        """Return executable names to try through PATH lookup."""
        if sys.platform.startswith("win"):
            return ("chrome.exe", "chrome")
        if sys.platform == "darwin":
            return ("google-chrome", "chrome")
        return ("google-chrome", "google-chrome-stable", "chromium-browser")


__all__ = [
    "ChromeProfile",
    "ChromeProfileConfig",
]
