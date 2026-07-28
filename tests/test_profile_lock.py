"""
Tests for the advisory automation profile lock.

The lock exists to prevent two ArchitectOS instances from driving one
automation profile. Reclamation must be conservative: a lock belonging to
a live process, or to a different host, must never be stolen.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from src.browser.profile_lock import LockOwner, ProfileLock
from src.exceptions import ProfileLockedError


class TestAcquireRelease:
    """Basic acquisition and release behaviour."""

    def test_acquire_creates_lock_file(self, tmp_path: Path) -> None:
        """Acquiring writes a lock file into the user data directory."""
        lock = ProfileLock(tmp_path / "automation")
        owner = lock.acquire()

        assert lock.is_held is True
        assert lock.lock_path.exists()
        assert owner.pid == os.getpid()
        assert owner.hostname == socket.gethostname()

    def test_acquire_creates_missing_directory(self, tmp_path: Path) -> None:
        """A profile directory that does not exist yet is created."""
        target = tmp_path / "nested" / "automation"
        ProfileLock(target).acquire()
        assert target.is_dir()

    def test_release_removes_lock_file(self, tmp_path: Path) -> None:
        """Releasing removes the lock file and clears held state."""
        lock = ProfileLock(tmp_path)
        lock.acquire()
        lock.release()

        assert lock.is_held is False
        assert not lock.lock_path.exists()

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        """Releasing twice must not raise."""
        lock = ProfileLock(tmp_path)
        lock.acquire()
        lock.release()
        lock.release()

    def test_reacquire_is_noop_while_held(self, tmp_path: Path) -> None:
        """Acquiring an already-held lock returns the existing owner."""
        lock = ProfileLock(tmp_path)
        first = lock.acquire()
        second = lock.acquire()
        assert first.pid == second.pid

    def test_context_manager_releases(self, tmp_path: Path) -> None:
        """The context manager releases on exit, including on error."""
        lock = ProfileLock(tmp_path)
        with pytest.raises(RuntimeError):
            with lock:
                assert lock.lock_path.exists()
                raise RuntimeError("boom")
        assert lock.is_held is False
        assert not lock.lock_path.exists()


class TestContention:
    """Behaviour when a lock is already held."""

    def test_live_process_blocks_acquisition(self, tmp_path: Path) -> None:
        """A lock held by this live process blocks a second instance."""
        first = ProfileLock(tmp_path)
        first.acquire()

        second = ProfileLock(tmp_path)
        with pytest.raises(ProfileLockedError) as excinfo:
            second.acquire()

        assert excinfo.value.code == "PROFILE_LOCK_HELD"
        assert str(os.getpid()) in str(excinfo.value)

    def test_error_message_is_actionable(self, tmp_path: Path) -> None:
        """The contention error names the lock file to remove."""
        ProfileLock(tmp_path).acquire()
        with pytest.raises(ProfileLockedError) as excinfo:
            ProfileLock(tmp_path).acquire()
        assert "architectos.lock" in str(excinfo.value)


class TestStaleReclamation:
    """Reclamation rules for abandoned locks."""

    @staticmethod
    def _write_lock(path: Path, payload: dict[str, object]) -> None:
        """Write a raw lock record for test setup."""
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "architectos.lock").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_dead_local_pid_is_reclaimed(self, tmp_path: Path) -> None:
        """
        A lock from this host whose process is gone is reclaimed.

        PID 999999999 is above the maximum on supported platforms, so it
        is reliably absent.
        """
        self._write_lock(
            tmp_path / "x",
            {
                "pid": 999999999,
                "hostname": socket.gethostname(),
                "acquired_at": 0.0,
                "user_data_dir": str(tmp_path),
            },
        )

        lock = ProfileLock(tmp_path)
        owner = lock.acquire()
        assert owner.pid == os.getpid()

    def test_foreign_host_lock_is_not_reclaimed(self, tmp_path: Path) -> None:
        """
        A lock from another host is never stolen.

        Remote process liveness cannot be determined locally, so the
        conservative outcome is to refuse.
        """
        self._write_lock(
            tmp_path / "x",
            {
                "pid": 999999999,
                "hostname": "some-other-machine",
                "acquired_at": 0.0,
                "user_data_dir": str(tmp_path),
            },
        )

        with pytest.raises(ProfileLockedError) as excinfo:
            ProfileLock(tmp_path).acquire()
        assert "some-other-machine" in str(excinfo.value)

    def test_unparseable_lock_is_reclaimed(self, tmp_path: Path) -> None:
        """A corrupt lock file must not deadlock the system forever."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "architectos.lock").write_text(
            "not json at all", encoding="utf-8"
        )

        lock = ProfileLock(tmp_path)
        assert lock.acquire().pid == os.getpid()


class TestLockOwner:
    """Serialization of the owner record."""

    def test_round_trip(self) -> None:
        """An owner record survives serialization unchanged."""
        owner = LockOwner(
            pid=42,
            hostname="host-a",
            acquired_at=1.5,
            user_data_dir="/tmp/x",
        )
        restored = LockOwner.from_json(owner.to_json())

        assert restored is not None
        assert restored == owner

    def test_malformed_payload_returns_none(self) -> None:
        """Malformed records parse to None rather than raising."""
        assert LockOwner.from_json("{}") is None
        assert LockOwner.from_json("]") is None
        assert LockOwner.from_json("") is None

    def test_is_local_reflects_hostname(self) -> None:
        """Locality is decided by comparing against this host."""
        local = LockOwner(1, socket.gethostname(), 0.0, "/tmp")
        remote = LockOwner(1, "elsewhere", 0.0, "/tmp")

        assert local.is_local is True
        assert remote.is_local is False
