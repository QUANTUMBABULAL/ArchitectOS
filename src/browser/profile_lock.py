"""
Advisory lock owned by ArchitectOS over an automation user data directory.

Chromium enforces one browser instance per user data directory through its
own process singleton, which is keyed on that directory
(``chrome/browser/process_singleton.h``). That mechanism protects the
browser, but it reports contention only *after* a launch attempt, and on
a successful hand-off it exits with a normal status code — which a
supervisor can easily misread as success.

This module adds a lock ArchitectOS controls, acquired *before* any
launch. Its purpose is narrow and worth stating precisely:

* It prevents two ArchitectOS processes from driving the same automation
  profile concurrently, which is a self-inflicted collision that the
  dedicated-directory strategy does not address.
* It fails fast with an actionable error instead of deferring to an
  ambiguous browser-side failure.

It is explicitly **not** a substitute for Chromium's singleton and makes
no claim to prevent an unrelated browser from opening the directory. It
is advisory: it coordinates cooperating ArchitectOS processes only.

Stale locks are reclaimed conservatively. A lock is reclaimed only when
it was created on this host *and* its recorded process is no longer
alive. A lock from a different host is never reclaimed automatically,
because liveness of a remote process cannot be determined locally.

Depends only on the standard library so it is unit testable without a
browser.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Optional

from src.exceptions import ProfileLockedError
from src.logger import get_logger

_LOCK_FILENAME = "architectos.lock"


def _process_alive(pid: int) -> bool:
    """
    Check whether a process id is currently alive on this host.

    Implemented per platform without third-party dependencies. An
    indeterminate result is reported as alive, because treating a
    possibly-live process as dead would risk two instances driving one
    profile.

    Args:
        pid: Process identifier to test.

    Returns:
        True when the process exists or its state cannot be determined.
    """
    if pid <= 0:
        return False

    if sys.platform.startswith("win"):
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        error_invalid_parameter = 87

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(
            wintypes.DWORD(synchronize),
            wintypes.BOOL(False),
            wintypes.DWORD(pid),
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # ERROR_INVALID_PARAMETER means no such process. Anything else
        # (typically ERROR_ACCESS_DENIED) means it exists but is not
        # openable by this account.
        return kernel32.GetLastError() != error_invalid_parameter

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by another user.
        return True
    except OSError:
        return True
    return True


@dataclass(frozen=True, slots=True)
class LockOwner:
    """
    Recorded owner of a profile lock.

    Attributes:
        pid: Process id that acquired the lock.
        hostname: Host on which the lock was acquired.
        acquired_at: Unix timestamp of acquisition.
        user_data_dir: Directory the lock guards.
    """

    pid: int
    hostname: str
    acquired_at: float
    user_data_dir: str

    def to_json(self) -> str:
        """
        Serialize the owner record.

        Returns:
            Compact JSON representation.
        """
        return json.dumps(
            {
                "pid": self.pid,
                "hostname": self.hostname,
                "acquired_at": self.acquired_at,
                "user_data_dir": self.user_data_dir,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> Optional["LockOwner"]:
        """
        Parse an owner record.

        Args:
            payload: JSON text previously written by ``to_json``.

        Returns:
            Parsed owner, or None when the payload is absent, truncated,
            or malformed. A malformed lock is treated as unparseable
            rather than as an error, so callers can decide policy.
        """
        try:
            data = json.loads(payload)
            return cls(
                pid=int(data["pid"]),
                hostname=str(data["hostname"]),
                acquired_at=float(data["acquired_at"]),
                user_data_dir=str(data["user_data_dir"]),
            )
        except (ValueError, TypeError, KeyError):
            return None

    @property
    def is_local(self) -> bool:
        """
        Return whether the lock was acquired on this host.

        Returns:
            True when the recorded hostname matches this host.
        """
        return self.hostname == socket.gethostname()


class ProfileLock:
    """
    Exclusive advisory lock over one automation user data directory.

    Acquisition is atomic: the lock file is created with ``O_CREAT |
    O_EXCL``, so exactly one process wins a race. The winner records its
    identity in the file so a later process can distinguish a live owner
    from a stale one.

    Usable directly or as a context manager. Release is idempotent and
    never raises, so it is safe in shutdown paths.
    """

    def __init__(
        self,
        user_data_dir: Path,
        lock_filename: str = _LOCK_FILENAME,
    ) -> None:
        """
        Initialize the lock.

        Args:
            user_data_dir: Directory to guard. Created if absent at
                acquisition time.
            lock_filename: Name of the lock file inside the directory.
        """
        self._user_data_dir = Path(user_data_dir).expanduser()
        self._lock_path = self._user_data_dir / lock_filename
        self._held = False
        self._logger = get_logger(__name__)

    @property
    def lock_path(self) -> Path:
        """
        Return the path of the lock file.

        Returns:
            Lock file path.
        """
        return self._lock_path

    @property
    def is_held(self) -> bool:
        """
        Return whether this instance currently holds the lock.

        Returns:
            True when acquired and not yet released.
        """
        return self._held

    def read_owner(self) -> Optional[LockOwner]:
        """
        Read the current lock owner, if any.

        Returns:
            Parsed owner record, or None when no lock exists or the
            record cannot be parsed.
        """
        try:
            payload = self._lock_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None
        return LockOwner.from_json(payload)

    def acquire(self) -> LockOwner:
        """
        Acquire the lock, reclaiming it if it is provably stale.

        Returns:
            The owner record written for this process.

        Raises:
            ProfileLockedError: If the directory is held by a live
                ArchitectOS process, by a process on another host, or if
                the lock file cannot be created for filesystem reasons.
        """
        if self._held:
            return self._require_owner()

        self._user_data_dir.mkdir(parents=True, exist_ok=True)

        owner = LockOwner(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            acquired_at=time.time(),
            user_data_dir=str(self._user_data_dir),
        )

        try:
            self._create_exclusive(owner)
        except FileExistsError:
            self._reclaim_if_stale()
            try:
                self._create_exclusive(owner)
            except FileExistsError as exc:
                existing = self.read_owner()
                raise ProfileLockedError(
                    self._contention_message(existing),
                    code="PROFILE_LOCK_HELD",
                ) from exc
        except OSError as exc:
            raise ProfileLockedError(
                f"Failed to create profile lock at {self._lock_path}: {exc}",
                code="PROFILE_LOCK_CREATE_FAILED",
            ) from exc

        self._held = True
        self._logger.info(
            "Acquired automation profile lock at %s (pid %d)",
            self._lock_path,
            owner.pid,
        )
        return owner

    def release(self) -> None:
        """
        Release the lock if held.

        Never raises. Failures to remove the lock file are logged, since
        this runs in shutdown paths where raising would mask the original
        cause of shutdown.
        """
        if not self._held:
            return

        try:
            self._lock_path.unlink(missing_ok=True)
            self._logger.info(
                "Released automation profile lock at %s", self._lock_path
            )
        except OSError as exc:
            self._logger.warning(
                "Failed to remove profile lock %s: %s", self._lock_path, exc
            )
        finally:
            self._held = False

    def _create_exclusive(self, owner: LockOwner) -> None:
        """
        Atomically create the lock file with an owner record.

        Args:
            owner: Owner record to persist.

        Raises:
            FileExistsError: If the lock already exists.
            OSError: If creation fails for another reason.
        """
        descriptor = os.open(
            self._lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, owner.to_json().encode("utf-8"))
        finally:
            os.close(descriptor)

    def _reclaim_if_stale(self) -> None:
        """
        Remove the existing lock when it is provably abandoned.

        A lock is reclaimed only if it is unparseable, or was created on
        this host by a process that is no longer alive. Locks belonging
        to live processes and to other hosts are left untouched.
        """
        existing = self.read_owner()

        if existing is None:
            self._logger.warning(
                "Removing unparseable profile lock at %s", self._lock_path
            )
            self._unlink_quietly()
            return

        if not existing.is_local:
            self._logger.warning(
                "Profile lock at %s belongs to host '%s'; not reclaiming "
                "because remote process liveness cannot be determined",
                self._lock_path,
                existing.hostname,
            )
            return

        if _process_alive(existing.pid):
            return

        self._logger.warning(
            "Reclaiming stale profile lock at %s (pid %d is no longer "
            "running)",
            self._lock_path,
            existing.pid,
        )
        self._unlink_quietly()

    def _unlink_quietly(self) -> None:
        """Remove the lock file, ignoring removal failures."""
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError as exc:
            self._logger.warning(
                "Failed to reclaim profile lock %s: %s", self._lock_path, exc
            )

    def _require_owner(self) -> LockOwner:
        """
        Return the current owner record, reconstructing it if needed.

        Returns:
            Owner record for the held lock.
        """
        owner = self.read_owner()
        if owner is not None:
            return owner
        return LockOwner(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            acquired_at=time.time(),
            user_data_dir=str(self._user_data_dir),
        )

    def _contention_message(self, existing: Optional[LockOwner]) -> str:
        """
        Build an actionable contention error message.

        Args:
            existing: Owner record read from the lock, if parseable.

        Returns:
            Operator-facing message.
        """
        if existing is None:
            return (
                f"Automation profile {self._user_data_dir} is locked by "
                f"another process and the lock record is unreadable. "
                f"Remove {self._lock_path} if no ArchitectOS instance is "
                f"running."
            )

        scope = "this host" if existing.is_local else f"host '{existing.hostname}'"
        return (
            f"Automation profile {self._user_data_dir} is already in use by "
            f"an ArchitectOS instance (pid {existing.pid} on {scope}). Stop "
            f"that instance, or configure a different automation profile. "
            f"If no such process exists, remove {self._lock_path}."
        )

    def __enter__(self) -> "ProfileLock":
        """
        Acquire the lock on context entry.

        Returns:
            This lock instance.
        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Release the lock on context exit."""
        self.release()


__all__ = [
    "LockOwner",
    "ProfileLock",
]
