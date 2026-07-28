"""
Tests for the dedicated ArchitectOS Chrome profile.

Two properties carry real risk. The profile root must stay separate from
the user's own Chrome user data directory, because Chrome's singleton lock
covers the whole root and sharing it would fail whenever normal Chrome is
open. And migration must move an existing profile rather than discard it,
because discarding it silently signs the user out of every provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.browser.chrome_profile import (
    ARCHITECTOS_PROFILE_NAME,
    AUTOMATION_ROOT_DIRNAME,
    ChromeProfile,
)
from src.exceptions import BrowserError


class TestProfileLayout:
    """Where the profile lives."""

    def test_root_is_under_the_data_directory(self, tmp_path: Path) -> None:
        """The profile root sits inside the application data directory."""
        root = ChromeProfile.automation_user_data_dir(
            profile_name=ARCHITECTOS_PROFILE_NAME, base_dir=tmp_path
        )
        assert root == tmp_path / AUTOMATION_ROOT_DIRNAME
        assert tmp_path in root.parents

    def test_layout_matches_specification(self, tmp_path: Path) -> None:
        """The documented layout is data/chrome-profile/ArchitectOS."""
        root = ChromeProfile.automation_user_data_dir(
            profile_name=ARCHITECTOS_PROFILE_NAME, base_dir=tmp_path
        )
        profile = root / ARCHITECTOS_PROFILE_NAME

        assert root.name == "chrome-profile"
        assert profile.name == "ArchitectOS"

    def test_root_is_not_the_users_chrome_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """
        The root must never be Chrome's own user data directory. Sharing
        it would collide with the singleton lock whenever the user has
        Chrome open.
        """
        root = ChromeProfile.automation_user_data_dir(
            profile_name=ARCHITECTOS_PROFILE_NAME, base_dir=tmp_path
        )
        assert "User Data" not in str(root)


class TestProfileCreation:
    """Creating the profile on first run."""

    def test_creates_root_and_profile_when_allowed(
        self,
        tmp_path: Path,
    ) -> None:
        """A first run creates both directories."""
        root = tmp_path / AUTOMATION_ROOT_DIRNAME
        profile = ChromeProfile.validate_profile(
            user_data_dir=root,
            profile_directory=ARCHITECTOS_PROFILE_NAME,
            create_if_missing=True,
        )

        assert root.is_dir()
        assert profile.is_dir()
        assert profile.name == ARCHITECTOS_PROFILE_NAME

    def test_refuses_to_create_when_not_allowed(
        self,
        tmp_path: Path,
    ) -> None:
        """Without permission a missing profile is an explicit error."""
        with pytest.raises(BrowserError) as excinfo:
            ChromeProfile.validate_profile(
                user_data_dir=tmp_path / "absent",
                profile_directory=ARCHITECTOS_PROFILE_NAME,
                create_if_missing=False,
            )
        assert excinfo.value.code == "CHROME_USER_DATA_DIR_NOT_FOUND"

    def test_existing_profile_is_reused(self, tmp_path: Path) -> None:
        """
        An existing profile is returned untouched. Its contents are the
        user's signed-in sessions and must never be recreated.
        """
        root = tmp_path / AUTOMATION_ROOT_DIRNAME
        profile = root / ARCHITECTOS_PROFILE_NAME
        profile.mkdir(parents=True)
        marker = profile / "Cookies"
        marker.write_text("session-data", encoding="utf-8")

        resolved = ChromeProfile.validate_profile(
            user_data_dir=root,
            profile_directory=ARCHITECTOS_PROFILE_NAME,
            create_if_missing=True,
        )

        assert resolved == profile
        assert marker.read_text(encoding="utf-8") == "session-data"


class TestLegacyMigration:
    """Preserving sessions from the previous layout."""

    @staticmethod
    def _make_legacy(base: Path) -> Path:
        """Create a legacy profile containing a session marker."""
        legacy = base / "chrome-profiles" / "automation" / "Default"
        legacy.mkdir(parents=True)
        (legacy / "Cookies").write_text("old-session", encoding="utf-8")
        return legacy

    def test_moves_legacy_profile(self, tmp_path: Path) -> None:
        """An existing profile is moved, preserving signed-in sessions."""
        legacy = self._make_legacy(tmp_path)

        destination = ChromeProfile.migrate_legacy_profile(
            base_dir=tmp_path,
            profile_directory=ARCHITECTOS_PROFILE_NAME,
        )

        assert destination is not None
        assert destination.is_dir()
        assert (destination / "Cookies").read_text(
            encoding="utf-8"
        ) == "old-session"
        assert not legacy.exists()

    def test_does_nothing_without_a_legacy_profile(
        self,
        tmp_path: Path,
    ) -> None:
        """A clean install migrates nothing."""
        assert (
            ChromeProfile.migrate_legacy_profile(base_dir=tmp_path) is None
        )

    def test_never_overwrites_an_existing_profile(
        self,
        tmp_path: Path,
    ) -> None:
        """
        A current profile always wins. Overwriting it would replace live
        sessions with stale ones.
        """
        self._make_legacy(tmp_path)
        current = (
            tmp_path
            / AUTOMATION_ROOT_DIRNAME
            / ARCHITECTOS_PROFILE_NAME
        )
        current.mkdir(parents=True)
        (current / "Cookies").write_text("current", encoding="utf-8")

        assert (
            ChromeProfile.migrate_legacy_profile(base_dir=tmp_path) is None
        )
        assert (current / "Cookies").read_text(
            encoding="utf-8"
        ) == "current"

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Running migration twice is safe."""
        self._make_legacy(tmp_path)
        first = ChromeProfile.migrate_legacy_profile(base_dir=tmp_path)
        second = ChromeProfile.migrate_legacy_profile(base_dir=tmp_path)

        assert first is not None
        assert second is None


class TestLaunchArguments:
    """Flags that make the window behave like normal Chrome."""

    @staticmethod
    def _profile(tmp_path: Path) -> ChromeProfile:
        """Build a profile object without touching the filesystem."""
        root = tmp_path / AUTOMATION_ROOT_DIRNAME
        return ChromeProfile(
            executable_path=tmp_path / "chrome.exe",
            user_data_dir=root,
            profile_directory=ARCHITECTOS_PROFILE_NAME,
            profile_path=root / ARCHITECTOS_PROFILE_NAME,
        )

    def test_selects_the_architectos_profile(self, tmp_path: Path) -> None:
        """Chrome is told explicitly which profile folder to use."""
        args = self._profile(tmp_path).launch_args()
        assert f"--profile-directory={ARCHITECTOS_PROFILE_NAME}" in args

    def test_maximized_by_default(self, tmp_path: Path) -> None:
        """The window opens maximized."""
        assert "--start-maximized" in self._profile(tmp_path).launch_args()

    def test_maximize_can_be_disabled(self, tmp_path: Path) -> None:
        """Headless runs do not request a maximized window."""
        args = self._profile(tmp_path).launch_args(maximized=False)
        assert "--start-maximized" not in args

    def test_suppresses_automation_banner(self, tmp_path: Path) -> None:
        """
        The automation infobar is removed so the browser looks normal
        while the user signs in.
        """
        args = self._profile(tmp_path).launch_args()
        assert "--disable-blink-features=AutomationControlled" in args

    def test_suppresses_first_run_interstitials(
        self,
        tmp_path: Path,
    ) -> None:
        """First-run and default-browser prompts do not block startup."""
        args = self._profile(tmp_path).launch_args()
        assert "--no-first-run" in args
        assert "--no-default-browser-check" in args

    def test_does_not_disable_normal_chrome_ui(
        self,
        tmp_path: Path,
    ) -> None:
        """
        The window keeps normal Chrome chrome. Flags that strip the UI
        would make manual sign-in awkward and look untrustworthy.
        """
        args = self._profile(tmp_path).launch_args()
        for hostile in (
            "--headless",
            "--app=",
            "--kiosk",
            "--disable-extensions",
            "--incognito",
        ):
            assert not any(hostile in arg for arg in args)
