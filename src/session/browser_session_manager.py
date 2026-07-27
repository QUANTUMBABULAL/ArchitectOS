"""
Persistent multi-provider research session.

BrowserSessionManager turns stateless browser automation into a standing
research laboratory. It launches the browser exactly once, opens one tab
per configured provider, keeps every tab alive for the lifetime of the
process, and repairs individual tabs without disturbing the others.

Responsibilities, in the order they matter:

* Launch the browser once. Subsequent research requests reuse it.
* Open and register one tab per provider, tracking readiness per provider.
* Dispatch prompts to all ready providers concurrently, with a per-provider
  timeout, continuing when a provider fails.
* Preserve each provider's conversation so follow-ups have context.
* Monitor health in the background and reopen only the tabs that died.

It deliberately does not decide *what* to ask. Prompt content, debate
strategy, and consensus belong to the layers above; this component owns
browser and provider lifecycle only.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.browser import BrowserLaunchConfig, BrowserManager, BrowserSession
from src.config import Settings, get_settings
from src.exceptions import (
    ProviderAuthError,
    ProviderChallengeError,
    WorkerError,
)
from src.logger import get_logger
from src.workers import (
    AuthState,
    AuthStatus,
    ProviderRegistry,
    WorkerManager,
    WorkerQuery,
    WorkerResponse,
    WorkerState,
    build_registry,
    build_worker,
    get_site,
)
from src.workers.web_chat_worker import WebChatWorker


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """
    Readiness snapshot for one provider.

    Attributes:
        name: Provider name.
        display_name: Human-readable provider name.
        state: Current worker state.
        ready: True when the provider can accept a prompt.
        turns: Completed conversation turns.
        detail: Failure or pause description when not ready.
        verified: Whether the provider's selectors are confirmed.
        paused: True when the provider is awaiting manual resolution of a
            verification challenge.
    """

    name: str
    display_name: str
    state: WorkerState
    ready: bool
    turns: int
    detail: Optional[str] = None
    verified: bool = False
    paused: bool = False


@dataclass
class SessionStats:
    """
    Cumulative counters for the running session.

    Attributes:
        started_at: When the session was opened.
        prompts_dispatched: Total prompts sent to providers.
        responses_received: Total successful responses.
        provider_failures: Total failed provider calls.
        recoveries: Total individual tab recoveries performed.
    """

    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    prompts_dispatched: int = 0
    responses_received: int = 0
    provider_failures: int = 0
    recoveries: int = 0


class BrowserSessionManager:
    """
    Owns a persistent browser and a live tab per AI provider.

    One instance is expected to live for the lifetime of the process. The
    browser is launched on :meth:`open` and stays up; providers keep their
    tabs and their conversations. Health monitoring runs as a background
    task and repairs tabs individually.
    """

    def __init__(
        self,
        browser: BrowserManager,
        workers: WorkerManager,
        settings: Optional[Settings] = None,
        launch_config: Optional[BrowserLaunchConfig] = None,
        registry: Optional[ProviderRegistry] = None,
    ) -> None:
        """
        Initialize the session manager.

        Args:
            browser: Browser lifecycle manager.
            workers: Worker registry used to hold provider workers.
            settings: Optional application settings.
            launch_config: Optional launch configuration override.
            registry: Optional provider registry. When omitted one is
                built from settings. The registry is the only source of
                which providers may participate.
        """
        self._browser = browser
        self._workers = workers
        self._settings = settings or get_settings()
        self._launch_config = launch_config
        self._registry = registry or build_registry(self._settings)
        self._logger = get_logger(__name__)

        self._session: Optional[BrowserSession] = None
        self._provider_names: list[str] = []
        self._failures: dict[str, str] = {}
        self._paused: dict[str, str] = {}
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._closing = False
        self._stats = SessionStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """
        Return whether the session is open with at least one provider.

        Returns:
            True when the browser is running and a provider is ready.
        """
        return self._session is not None and bool(self.ready_providers())

    @property
    def stats(self) -> SessionStats:
        """
        Return cumulative session counters.

        Returns:
            Session statistics.
        """
        return self._stats

    async def open(self) -> list[str]:
        """
        Launch the browser and bring every configured provider online.

        Idempotent: calling this on an open session returns the providers
        already ready without relaunching the browser or reloading tabs.
        That is what makes repeated research requests free of setup cost.

        Returns:
            Names of providers that became ready.
        """
        if self.is_open:
            self._logger.debug(
                "Session already open with %d provider(s)",
                len(self.ready_providers()),
            )
            return self.ready_providers()

        self._registry.log_summary()
        self._provider_names = self._registry.enabled_names()

        if not self._provider_names:
            self._logger.error(
                "No providers are enabled; nothing to open. Set "
                "ENABLED_PROVIDERS to at least one provider."
            )
            return []

        await self._purge_disabled()

        if self._session is None:
            self._logger.info("Launching persistent browser")
            self._session = await self._browser.start(self._launch_config)
            self._logger.info(
                "Chrome launched (session %s)", self._session.session_id
            )

        await self._register_providers(self._session)
        ready = await self._initialize_providers()

        # Startup ends with a verified picture rather than an assumed
        # one: launch -> profile -> tabs -> verify auth -> ready.
        await self.verify_authentication()
        ready = self.ready_providers()

        if ready:
            self._start_monitor()

        return ready

    async def close(self) -> None:
        """
        Stop monitoring, close providers, and shut the browser down.

        Never raises; shutdown problems are logged so process exit is
        always possible.
        """
        self._closing = True
        await self._stop_monitor()

        try:
            await self._workers.stop_all()
        except Exception as exc:
            self._logger.warning("Provider shutdown reported: %s", exc)

        try:
            await self._browser.stop()
        except Exception as exc:
            self._logger.warning("Browser shutdown reported: %s", exc)

        self._session = None
        self._logger.info("Research session closed")

    async def _purge_disabled(self) -> list[str]:
        """
        Remove any registered worker whose provider is now disabled.

        Configuration can change between runs, and a provider disabled
        after being registered must not linger in the worker registry
        where dispatch, monitoring, or recovery could still reach it.

        Returns:
            Names of providers that were removed.
        """
        stale = [
            name
            for name in self._workers.worker_names
            if not self._registry.is_enabled(name)
        ]

        for name in stale:
            try:
                worker = self._workers.get(name)
                await worker.stop()
            except Exception as exc:
                self._logger.warning(
                    "Error stopping disabled provider %s: %s", name, exc
                )
            finally:
                self._workers.unregister(name)
                self._paused.pop(name, None)
                self._failures.pop(name, None)
                self._logger.info(
                    "Unregistered disabled provider: %s", name
                )

        return stale

    async def _register_providers(self, session: BrowserSession) -> None:
        """
        Register a worker for each configured provider.

        Args:
            session: Browser session the workers operate in.
        """
        unverified = [
            name
            for name in self._provider_names
            if not get_site(name).verified
        ]
        if unverified:
            self._logger.warning(
                "Providers with unverified selectors: %s. Failures for "
                "these are most likely selector drift.",
                ", ".join(unverified),
            )

        for name in self._provider_names:
            # Defence in depth: _provider_names already comes from the
            # registry, but registration is the point of no return for a
            # provider, so participation is confirmed here too.
            if not self._registry.is_enabled(name):
                continue
            if name in self._workers.worker_names:
                continue
            self._workers.register(build_worker(name, session))

    async def _initialize_providers(self) -> list[str]:
        """
        Bring each registered provider to readiness.

        Providers are initialized sequentially: they share one browser and
        must not race while creating and navigating tabs. Individual
        failures are recorded and skipped.

        Returns:
            Names of providers that became ready.
        """
        ready: list[str] = []
        self._failures.clear()

        for name in self._provider_names:
            worker = self._provider(name)
            display = worker.display_name
            try:
                await worker.initialize()
                ready.append(name)
                self._paused.pop(name, None)
                self._logger.info("%s ready", display)
            except ProviderAuthError as exc:
                # Not a failure: the provider is fine, it just needs a
                # human. Wait for the sign-in rather than giving up.
                if await self._await_login(name, str(exc)):
                    ready.append(name)
            except ProviderChallengeError as exc:
                self.pause_provider(name, str(exc))
            except Exception as exc:
                self._failures[name] = str(exc)
                self._logger.warning("%s unavailable: %s", display, exc)

        self._logger.info(
            "Research session online with %d/%d provider(s): %s",
            len(ready),
            len(self._provider_names),
            ", ".join(ready) or "none",
        )
        return ready

    # ------------------------------------------------------------------
    # Provider access
    # ------------------------------------------------------------------

    async def _await_login(self, name: str, instruction: str) -> bool:
        """
        Prompt for a manual sign-in and wait for it to complete.

        The provider's tab is already open at its sign-in page. Nothing is
        typed or clicked: the user signs in themselves and the session is
        detected when it appears. Credentials are never seen by
        ArchitectOS and exist only inside the browser profile.

        Args:
            name: Provider name.
            instruction: Operator instruction to display.

        Returns:
            True when the provider became authenticated.
        """
        worker = self._provider(name)
        budget = self._settings.login_wait_seconds

        print(f"\n[!] {instruction}\n")
        self._logger.warning(
            "%s requires manual sign-in", worker.display_name
        )

        if budget <= 0:
            self.pause_provider(name, "Sign-in required")
            return False

        status = await worker.wait_for_login(
            timeout_seconds=budget,
            poll_interval_seconds=self._settings.login_poll_seconds,
        )

        if status.is_ready:
            print(f"[+] {worker.display_name} signed in — continuing.")
            self._paused.pop(name, None)
            self._failures.pop(name, None)
            return True

        self.pause_provider(
            name, status.detail or "Sign-in not completed in time"
        )
        return False

    async def verify_authentication(self) -> dict[str, AuthStatus]:
        """
        Check every enabled provider's authentication state.

        Run after tabs open so startup ends with a known-good picture of
        which providers are usable. Providers needing a human are paused,
        never restarted.

        Returns:
            Mapping of provider name to authentication snapshot.
        """
        statuses: dict[str, AuthStatus] = {}

        for name in self._provider_names:
            if name not in self._workers.worker_names:
                continue

            worker = self._provider(name)
            status = await worker.check_auth()
            statuses[name] = status

            if status.is_ready:
                self._paused.pop(name, None)
                continue

            if status.state.needs_human:
                self.pause_provider(
                    name, status.detail or status.state.value
                )
                if status.action:
                    print(f"\n[!] {status.action}\n")

        ready = [n for n, s in statuses.items() if s.is_ready]
        self._logger.info(
            "Authentication verified: %d/%d provider(s) signed in%s",
            len(ready),
            len(statuses),
            f" ({', '.join(ready)})" if ready else "",
        )
        return statuses

    async def login_status(self) -> dict[str, AuthStatus]:
        """
        Return the current authentication state of every provider.

        Returns:
            Mapping of provider name to authentication snapshot.
        """
        return {
            name: await self._provider(name).check_auth()
            for name in self._provider_names
            if name in self._workers.worker_names
        }

    async def await_pending_logins(self) -> list[str]:
        """
        Re-check paused providers and wait for outstanding sign-ins.

        Returns:
            Names of providers that became authenticated.
        """
        signed_in: list[str] = []

        for name in list(self._paused):
            if name not in self._workers.worker_names:
                continue

            worker = self._provider(name)
            status = await worker.check_auth()

            if status.is_ready:
                self.resume_provider(name)
                signed_in.append(name)
                continue

            if status.state is AuthState.LOGIN_REQUIRED:
                if await self._await_login(
                    name, status.action or "Sign-in required"
                ):
                    self.resume_provider(name)
                    signed_in.append(name)

        return signed_in

    def _provider(self, name: str) -> WebChatWorker:
        """
        Return a registered provider worker.

        Args:
            name: Provider name.

        Returns:
            Provider worker.

        Raises:
            WorkerError: If the provider is not registered, or is not a
                web chat worker.
        """
        worker = self._workers.get(name)
        if not isinstance(worker, WebChatWorker):
            raise WorkerError(
                f"Worker '{name}' is not a web chat provider",
                code="PROVIDER_WRONG_TYPE",
            )
        return worker

    def ready_providers(self) -> list[str]:
        """
        Return providers currently able to accept prompts.

        Paused providers are excluded even if their worker still reports
        READY: a provider behind a CAPTCHA must not be handed prompts.

        Returns:
            Provider names in configured order.
        """
        ready = {worker.name for worker in self._workers.ready_workers()}
        return self._registry.filter_enabled(
            name
            for name in self._provider_names
            if name in ready and name not in self._paused
        )

    @property
    def registry(self) -> ProviderRegistry:
        """
        Return the provider registry governing participation.

        Returns:
            Provider registry.
        """
        return self._registry

    def enabled_providers(self) -> list[str]:
        """
        Return every provider configuration permits, ready or not.

        Returns:
            Enabled provider names in configured order.
        """
        return self._registry.enabled_names()

    def disabled_providers(self) -> list[str]:
        """
        Return providers configuration excludes entirely.

        Returns:
            Disabled provider names.
        """
        return self._registry.disabled_names()

    @property
    def paused_providers(self) -> dict[str, str]:
        """
        Return paused providers and the reason each was paused.

        Returns:
            Mapping of provider name to pause reason.
        """
        return dict(self._paused)

    def pause_provider(self, name: str, reason: str) -> None:
        """
        Pause one provider without affecting the others.

        Used for human-verification challenges, which cannot be resolved
        automatically. The provider keeps its tab and its authenticated
        state so the user can solve the challenge in place and resume.

        Args:
            name: Provider name.
            reason: Why the provider was paused.
        """
        self._paused[name] = reason
        self._logger.warning(
            "Provider paused: %s — %s. Research continues with the "
            "remaining providers.",
            name,
            reason,
        )

    def resume_provider(self, name: str) -> bool:
        """
        Clear a provider's paused state.

        Args:
            name: Provider name.

        Returns:
            True when the provider was paused and is now resumed.
        """
        if self._paused.pop(name, None) is None:
            return False
        self._logger.info("Provider resumed: %s", name)
        return True

    async def resume_paused(self) -> list[str]:
        """
        Re-check every paused provider and resume the ones now usable.

        Called after a user reports having solved a challenge manually.

        Returns:
            Names of providers that were resumed.
        """
        resumed: list[str] = []

        for name in list(self._paused):
            try:
                worker = self._provider(name)
                if await worker.is_ready():
                    self.resume_provider(name)
                    resumed.append(name)
                    continue
                await worker.initialize()
                self.resume_provider(name)
                resumed.append(name)
            except ProviderChallengeError:
                self._logger.info(
                    "%s still shows a challenge; staying paused", name
                )
            except Exception as exc:
                self._logger.warning(
                    "Could not resume %s: %s", name, exc
                )
        return resumed

    async def provider_status(self) -> list[ProviderStatus]:
        """
        Collect a readiness snapshot for every configured provider.

        Returns:
            Status entries in configured order.
        """
        statuses: list[ProviderStatus] = []

        for name in self._provider_names:
            try:
                worker = self._provider(name)
            except WorkerError as exc:
                statuses.append(
                    ProviderStatus(
                        name=name,
                        display_name=name,
                        state=WorkerState.ERROR,
                        ready=False,
                        turns=0,
                        detail=str(exc),
                    )
                )
                continue

            paused_reason = self._paused.get(name)
            ready = (
                paused_reason is None
                and worker.state is WorkerState.READY
                and await worker.is_ready()
            )
            statuses.append(
                ProviderStatus(
                    name=name,
                    display_name=worker.display_name,
                    state=worker.state,
                    ready=ready,
                    turns=worker.conversation.turns,
                    detail=paused_reason or self._failures.get(name),
                    verified=worker.site.verified,
                    paused=paused_reason is not None,
                )
            )
        return statuses

    def conversation_report(self) -> str:
        """
        Summarize each provider's conversation state.

        Returns:
            Multi-line human-readable summary.
        """
        if not self._provider_names:
            return "No providers configured."

        lines = ["Conversations:"]
        for name in self._provider_names:
            try:
                worker = self._provider(name)
            except WorkerError:
                lines.append(f"  {name}: not registered")
                continue
            state = worker.conversation
            lines.append(
                f"  {worker.display_name:<12} turns={state.turns:<3} "
                f"context~{state.approx_context_chars:<7}c "
                f"resets={state.resets} "
                f"id={state.conversation_id or '-'}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        prompt: str,
        providers: Optional[list[str]] = None,
        new_conversation: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> list[WorkerResponse]:
        """
        Send a prompt to providers concurrently and collect the answers.

        Submission is fanned out with no ordering between providers, so a
        slow provider never delays the others. Each provider has its own
        timeout; exceeding it produces a failed response rather than
        aborting the batch.

        Args:
            prompt: Prompt text.
            providers: Providers to consult. Defaults to all ready ones.
            new_conversation: Whether each provider should start a fresh
                conversation. Defaults to False so context is preserved.
            timeout_seconds: Per-provider timeout. Defaults to the
                configured provider timeout.

        Returns:
            Responses in the same order as the requested providers.

        Raises:
            WorkerError: If no providers are available.
        """
        targets = providers if providers is not None else self.ready_providers()
        # Disabled providers can never receive a prompt, even when a
        # caller names one explicitly.
        targets = [
            name
            for name in self._registry.filter_enabled(targets)
            if name in self._workers.worker_names
        ]

        if not targets:
            raise WorkerError(
                "No providers are ready to receive prompts",
                code="SESSION_NO_PROVIDERS",
            )

        budget = timeout_seconds or self._settings.provider_timeout
        self._stats.prompts_dispatched += len(targets)
        self._logger.info(
            "Dispatching prompt to %d provider(s) concurrently: %s",
            len(targets),
            ", ".join(targets),
        )

        results = await asyncio.gather(
            *(
                self._ask_one(name, prompt, new_conversation, budget)
                for name in targets
            ),
            return_exceptions=False,
        )

        for response in results:
            if response.success:
                self._stats.responses_received += 1
            else:
                self._stats.provider_failures += 1

        succeeded = [r.worker_name for r in results if r.success]
        self._logger.info(
            "Collected %d/%d response(s): %s",
            len(succeeded),
            len(results),
            ", ".join(succeeded) or "none",
        )
        return list(results)

    async def _ask_one(
        self,
        name: str,
        prompt: str,
        new_conversation: bool,
        timeout_seconds: float,
    ) -> WorkerResponse:
        """
        Ask one provider, converting any failure into a failed response.

        Args:
            name: Provider name.
            prompt: Prompt text.
            new_conversation: Whether to start a fresh conversation.
            timeout_seconds: Timeout for this provider.

        Returns:
            Worker response; ``success`` is False on timeout or error.
        """
        query = WorkerQuery(prompt=prompt, new_conversation=new_conversation)

        try:
            worker = self._provider(name)
            return await asyncio.wait_for(
                worker.ask(query),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "Provider timeout: %s exceeded %.0fs",
                name,
                timeout_seconds,
            )
            return WorkerResponse(
                query_id=query.query_id,
                worker_name=name,
                prompt=prompt,
                answer="",
                success=False,
                error=f"Timed out after {timeout_seconds:.0f}s",
            )
        except ProviderChallengeError as exc:
            self.pause_provider(name, str(exc))
            return WorkerResponse(
                query_id=query.query_id,
                worker_name=name,
                prompt=prompt,
                answer="",
                success=False,
                error=f"Paused: {exc}",
            )
        except ProviderAuthError as exc:
            # A session that lapsed mid-run. Pause this provider only;
            # the others are unaffected and the round continues.
            self.pause_provider(name, "Session expired")
            print(f"\n[!] {exc}\n")
            return WorkerResponse(
                query_id=query.query_id,
                worker_name=name,
                prompt=prompt,
                answer="",
                success=False,
                error="Session expired; sign in again to re-enable",
            )
        except Exception as exc:
            self._logger.warning("Provider %s failed: %s", name, exc)
            return WorkerResponse(
                query_id=query.query_id,
                worker_name=name,
                prompt=prompt,
                answer="",
                success=False,
                error=str(exc),
            )

    async def reset_conversations(
        self,
        providers: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Start fresh conversations with the given providers.

        Args:
            providers: Providers to reset. Defaults to all ready ones.

        Returns:
            Names of providers that were reset successfully.
        """
        targets = providers if providers is not None else self.ready_providers()
        reset: list[str] = []

        for name in targets:
            try:
                await self._provider(name).reset_conversation()
                reset.append(name)
            except Exception as exc:
                self._logger.warning(
                    "Failed to reset conversation for %s: %s", name, exc
                )
        return reset

    async def maybe_reset_for_context(self) -> list[str]:
        """
        Reset conversations that have grown past the context budget.

        Returns:
            Names of providers that were reset.
        """
        limit = self._settings.conversation_context_chars
        if limit <= 0:
            return []

        overflowing = [
            name
            for name in self.ready_providers()
            if self._provider(name).conversation.needs_reset(limit)
        ]
        if not overflowing:
            return []

        self._logger.info(
            "Resetting %d conversation(s) at the context budget: %s",
            len(overflowing),
            ", ".join(overflowing),
        )
        return await self.reset_conversations(overflowing)

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def profile_path(self) -> Optional[Path]:
        """
        Return the persistent Chrome profile directory.

        Returns:
            User data directory, or None when no explicit profile is
            configured.
        """
        config = self._launch_config
        if config is None or config.user_data_dir is None:
            return None
        return Path(config.user_data_dir)

    def profile_report(self) -> str:
        """
        Summarize the persistent profile backing authentication.

        Returns:
            Multi-line human-readable summary.
        """
        path = self.profile_path()
        if path is None:
            return (
                "No persistent profile configured; sessions will not "
                "survive restarts."
            )

        exists = path.exists()
        size_mb = 0.0
        if exists:
            size_mb = sum(
                item.stat().st_size
                for item in path.rglob("*")
                if item.is_file()
            ) / (1024 * 1024)

        return "\n".join(
            [
                "Persistent browser profile:",
                f"  path    : {path}",
                f"  exists  : {'yes' if exists else 'no (created on launch)'}",
                f"  size    : {size_mb:.1f} MB",
                "  contents: cookies, local storage, IndexedDB, session "
                "storage, OAuth tokens",
                "  note    : ArchitectOS never reads or stores credentials; "
                "authentication lives only here.",
                "  reset   : /reset-profile (the only command that removes "
                "authentication)",
            ]
        )

    async def reset_profile(self, confirmed: bool = False) -> bool:
        """
        Delete the persistent browser profile.

        This is the only operation in the system permitted to destroy
        authentication, and it is never triggered automatically — not by
        recovery, not by a failed sign-in, not by shutdown. Every provider
        must be signed in again afterwards.

        The browser is closed first so Chrome is not writing to the
        directory during removal. The path is validated to sit under the
        configured data directory, so a misconfiguration cannot turn this
        into a delete of something unrelated.

        Args:
            confirmed: Must be True. Guards against an accidental call.

        Returns:
            True when a profile was removed.

        Raises:
            WorkerError: If called without confirmation, or if the
                resolved path fails its safety checks.
        """
        if not confirmed:
            raise WorkerError(
                "reset_profile requires explicit confirmation",
                code="PROFILE_RESET_UNCONFIRMED",
            )

        path = self.profile_path()
        if path is None:
            raise WorkerError(
                "No persistent profile is configured; nothing to reset",
                code="PROFILE_RESET_NOT_CONFIGURED",
            )

        data_root = Path(self._settings.data_dir).expanduser().resolve()
        resolved = path.expanduser().resolve()

        if data_root not in resolved.parents:
            raise WorkerError(
                f"Refusing to delete {resolved}: it is outside the data "
                f"directory {data_root}",
                code="PROFILE_RESET_UNSAFE_PATH",
            )
        if resolved == data_root:
            raise WorkerError(
                "Refusing to delete the data directory itself",
                code="PROFILE_RESET_UNSAFE_PATH",
            )

        self._logger.warning(
            "Resetting browser profile at %s; all providers will need "
            "signing in again",
            resolved,
        )
        await self.close()

        if not resolved.exists():
            self._logger.info("Profile directory did not exist")
            return False

        shutil.rmtree(resolved)
        self._logger.info("Browser profile removed: %s", resolved)
        return True

    # ------------------------------------------------------------------
    # Health monitoring and recovery
    # ------------------------------------------------------------------

    def _start_monitor(self) -> None:
        """Start the background health monitor if it is not running."""
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        if self._settings.session_monitor_interval <= 0:
            self._logger.info("Provider health monitoring disabled")
            return

        self._closing = False
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._logger.info(
            "Provider health monitor started (every %.0fs)",
            self._settings.session_monitor_interval,
        )

    async def _stop_monitor(self) -> None:
        """Cancel the background health monitor and wait for it to exit."""
        task = self._monitor_task
        self._monitor_task = None
        if task is None or task.done():
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.warning("Health monitor exited with: %s", exc)

    async def _monitor_loop(self) -> None:
        """
        Periodically check providers and recover the ones that died.

        Only unhealthy providers are touched. A healthy tab is never
        reloaded, which is what preserves conversational context across
        the lifetime of the session.
        """
        interval = self._settings.session_monitor_interval

        while not self._closing:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

            if self._closing:
                return

            try:
                await self.recover_unhealthy()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning(
                    "Health monitor iteration failed: %s", exc
                )

    async def recover_unhealthy(self) -> list[str]:
        """
        Reopen tabs for providers that are no longer usable.

        Providers that are busy are skipped: a provider mid-answer is not
        unhealthy, and restarting it would discard a response in flight.

        Returns:
            Names of providers that were recovered.
        """
        recovered: list[str] = []

        for name in list(self._provider_names):
            # Disabled providers are invisible to health monitoring: no
            # check, no restart, no log line. This is what stops a
            # provider behind an unsolvable challenge from generating
            # endless recovery activity.
            if not self._registry.is_enabled(name):
                continue
            if name not in self._workers.worker_names:
                continue
            # A paused provider is awaiting manual action, not broken.
            # Restarting it would discard the authenticated tab the user
            # needs in order to solve the challenge.
            if name in self._paused:
                continue

            worker = self._provider(name)
            if worker.state in {WorkerState.BUSY, WorkerState.STARTING}:
                continue
            if worker.state is WorkerState.STOPPED:
                continue

            # Classify before acting. An unauthenticated provider is not
            # unhealthy, and restarting it would discard the page the
            # user needs in order to sign in.
            status = await worker.check_auth()

            if status.is_ready:
                continue

            if status.state.needs_human:
                self.pause_provider(
                    name, status.detail or status.state.value
                )
                if status.action:
                    print(f"\n[!] {status.action}\n")
                continue

            if not status.state.is_recoverable:
                continue

            self._logger.warning(
                "%s tab is offline; recovering it", worker.display_name
            )
            try:
                await worker.restart()
                recovered.append(name)
                self._stats.recoveries += 1
                self._failures.pop(name, None)
            except (ProviderAuthError, ProviderChallengeError) as exc:
                self.pause_provider(name, str(exc))
            except Exception as exc:
                self._failures[name] = str(exc)
                self._logger.error(
                    "Failed to recover %s: %s", worker.display_name, exc
                )

        return recovered


__all__ = [
    "BrowserSessionManager",
    "ProviderStatus",
    "SessionStats",
]
