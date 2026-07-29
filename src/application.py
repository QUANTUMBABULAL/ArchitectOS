"""
Application runtime for the AI Research Operator.

ResearchOperatorApp is the composition root of the system. It wires the
existing components together — MemoryStore, OllamaClient, DecisionEngine,
Planner, ConsensusEngine, BrowserManager, WorkerManager, ChatGPTWorker,
and ResearchOrchestrator — and runs the interactive terminal loop.

The application follows the system pipeline exactly:

1. A user request arrives from the terminal.
2. The DecisionEngine (backed by local Ollama) classifies its complexity.
3. Simple requests are answered locally without touching the browser.
4. Complex requests lazily start the browser research stack (persistent
   Chrome profile + ChatGPT worker) and run the full orchestrated
   plan → consult → consensus → synthesize flow.

The browser is never launched for simple requests, and authentication is
never automated: the user signs in to chatgpt.com manually once inside
the persistent Chrome profile.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.brain import DecisionEngine, OllamaClient, TaskComplexity
from src.browser import BrowserLaunchConfig, BrowserManager
from src.config import Settings, get_settings
from src.consensus import ConsensusEngine
from src.exceptions import (
    AIResearchOperatorError,
    BrainError,
    ValidationError,
)
from src.logger import get_logger
from src.memory import MemoryStore
from src.debate import (
    DebateEngine,
    DebateOutcome,
    FinalAnswer,
    build_final_answer,
)
from src.orchestrator import ResearchOrchestrator, ResearchOutcome
from src.planner import Planner
from src.research import (
    ExecutiveReportBuilder,
    ResearchOperator,
    ResearchPlanner,
    ResearchResult,
)
from src.session import BrowserSessionManager
from src.routing import (
    FastRouter,
    RouteTarget,
    RoutingDecision,
    evaluate_expression,
    format_result,
)
from src.workers import WorkerManager


@dataclass(frozen=True, slots=True)
class RequestResult:
    """
    Outcome of handling one user request.

    Attributes:
        answer: Text to present to the user.
        route: Routing decision that produced this result.
        outcome: Full research outcome when the request went through the
            orchestrator, otherwise None.
    """

    answer: str
    route: RoutingDecision
    outcome: Optional[ResearchOutcome] = None
    debate: Optional[DebateOutcome] = None
    final: Optional[FinalAnswer] = None
    research: Optional[ResearchResult] = None

    @property
    def used_research(self) -> bool:
        """
        Return whether the orchestrator ran for this request.

        Returns:
            True when a research outcome is attached.
        """
        return self.outcome is not None


_BANNER = r"""
======================================================================
                        ResearchOS Ready
======================================================================
Type a research request and press Enter.

Commands:
  /research <q>    Multi-round debate across every live provider
  /open            Open the persistent session (launch Chrome, all tabs)
  /providers       Provider readiness, turns, and dispatch counters
  /conversations   Per-provider conversation state
  /reset           Clear conversation history on all providers
  /dashboard       Provider status board (refreshes pending sign-ins)
  /auth            Show each provider's authentication state
  /login           Wait for pending manual sign-ins to complete
  /profile         Show the persistent browser profile backing sessions
  /reset-profile   Delete the profile (ONLY way to remove sign-ins)
  /recover         Reopen any tab that has died
  /resume          Re-check providers paused by a CAPTCHA challenge
  /help            Show this help text
  /status          Show Ollama, browser, and worker health
  /routes          Show how requests have been routed this session
  /model <name>    Switch the local Ollama model
  /exit            Shut down and quit (also: /quit, Ctrl+C)
======================================================================
""".strip("\n")

_HELP_TEXT = (
    "Enter any question or research goal. Requests are routed adaptively:\n"
    "  greetings, commands and arithmetic  -> answered instantly, no model\n"
    "  short questions                     -> local Ollama model\n"
    "  shopping, current info, comparisons -> browser research\n"
    "  anything ambiguous                  -> Decision Engine decides\n\n"
    "Commands: /help, /status, /routes, /model <name>, /exit, /quit"
)

_GREETING_REPLY = (
    "Hello. Ask me a question, or type /help to see how requests are "
    "routed."
)


class ResearchOperatorApp:
    """
    Composition root and interactive runtime of the AI Research Operator.

    The application owns every long-lived component and their lifecycle:
    it initializes storage and local inference eagerly at startup, starts
    the browser research stack lazily on the first complex request, and
    releases everything on shutdown. All coordination logic stays in the
    existing components; this class only wires and drives them.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Build the application object graph.

        Construction is cheap and side-effect free; call
        :meth:`initialize` before handling requests.

        Args:
            settings: Optional application settings. Loaded from the
                environment when omitted.
        """
        self._settings = settings or get_settings()
        self._logger = get_logger(__name__)

        self._memory = MemoryStore(settings=self._settings)
        self._ollama = OllamaClient(settings=self._settings)
        self._engine = DecisionEngine(self._ollama, settings=self._settings)
        self._planner = Planner(self._engine, settings=self._settings)
        self._consensus = ConsensusEngine(
            decision_engine=self._engine,
            settings=self._settings,
        )
        self._browser = BrowserManager(settings=self._settings)
        self._workers = WorkerManager()
        self._router = FastRouter()
        self._session = BrowserSessionManager(
            browser=self._browser,
            workers=self._workers,
            settings=self._settings,
            launch_config=self._research_launch_config(),
        )
        self._debate = DebateEngine(
            session=self._session,
            consensus=self._consensus,
            settings=self._settings,
        )
        # The research operator is the default path: it plans a request
        # into subtasks and distributes them, rather than broadcasting one
        # question to everybody. The debate engine remains available
        # behind RESEARCH_MODE=debate.
        self._operator = ResearchOperator(
            session=self._session,
            consensus=self._consensus,
            planner=ResearchPlanner(
                ollama=self._ollama, settings=self._settings
            ),
            report_builder=ExecutiveReportBuilder(
                synthesize=self._write_summary
                if self._settings.final_answer_synthesis
                else None
            ),
            settings=self._settings,
        )
        self._orchestrator = ResearchOrchestrator(
            planner=self._planner,
            decision_engine=self._engine,
            worker_manager=self._workers,
            consensus_engine=self._consensus,
            memory=self._memory,
        )

        self._initialized = False
        self._research_stack_ready = False

    @property
    def orchestrator(self) -> ResearchOrchestrator:
        """
        Return the research orchestrator.

        Returns:
            Research orchestrator instance.
        """
        return self._orchestrator

    @property
    def is_initialized(self) -> bool:
        """
        Return whether the application has been initialized.

        Returns:
            True after :meth:`initialize` completed successfully.
        """
        return self._initialized

    async def initialize(self) -> None:
        """
        Initialize the application core.

        Creates required directories, initializes persistent memory, and
        verifies that the local Ollama server is reachable. The browser
        stack is intentionally NOT started here — it launches lazily on
        the first request that needs external research.

        Raises:
            BrainError: If the Ollama server is unreachable.
            AIResearchOperatorError: If any core component fails to
                initialize.
        """
        if self._initialized:
            return

        self._logger.info("Initializing AI Research Operator runtime...")

        self._settings.create_directories()
        await self._memory.initialize()
        self._logger.info("Memory store initialized")

        if not await self._ollama.health_check():
            raise BrainError(
                f"Ollama server is not reachable at "
                f"{self._settings.ollama_host}. Start the Ollama service "
                f"and try again.",
                code="OLLAMA_UNREACHABLE",
            )
        self._logger.info(
            "Connected to Ollama at %s", self._settings.ollama_host
        )

        # Participation is resolved and reported before any browser work,
        # so an operator sees which providers will and will not be used
        # regardless of whether the session opens at startup.
        print(self._session.registry.startup_summary())
        self._session.registry.log_summary()

        await self._verify_model()

        # Load the model now so the first user request does not pay
        # multi-second model load cost, which would otherwise be
        # indistinguishable from a slow generation.
        await self._ollama.preload()

        self._initialized = True

        if self._settings.open_session_on_start:
            self._logger.info("Opening persistent research session")
            await self.open_session()
        self._logger.info("AI Research Operator runtime initialized")

    async def research_now(
        self,
        query: str,
        new_conversation: Optional[bool] = None,
    ) -> RequestResult:
        """
        Run the explicit-research fast path.

        The user has already declared intent, so nothing is classified and
        nothing is planned. The sequence is deliberately front-loaded on
        the browser: Chrome launches first, a tab opens per configured
        provider, all prompts are submitted concurrently, and the answers
        are combined by the consensus engine.

        Providers that fail to start or fail to answer are skipped. The
        call succeeds as long as one provider responds.

        Args:
            query: Research query, sent verbatim to every provider.
            new_conversation: Force providers to start fresh threads for
                this run. Defaults to the configured policy (continue
                existing conversations).

        Returns:
            Result carrying the unified answer.

        Raises:
            AIResearchOperatorError: If the browser cannot start or every
                provider fails.
        """
        if not self._initialized:
            raise AIResearchOperatorError(
                "Application is not initialized; call initialize() first",
                code="APP_NOT_INITIALIZED",
            )

        cleaned = query.strip()
        if not cleaned:
            raise ValidationError(
                "Research query cannot be empty",
                code="RESEARCH_QUERY_EMPTY",
            )

        decision = RoutingDecision(
            target=RouteTarget.RESEARCH,
            confidence=1.0,
            rule="explicit_research_command",
            reason="User invoked the research command directly",
        )

        ready = await self.open_session()
        if not ready:
            raise AIResearchOperatorError(
                "No providers could be started. Check /providers, and sign "
                "in manually once in the automation Chrome window.",
                code="RESEARCH_NO_PROVIDERS",
            )

        await self._session.maybe_reset_for_context()

        if not self._settings.debate_enabled:
            print(f"[*] Consulting {len(ready)} provider(s) concurrently...")
            outcome = await self._orchestrator.run_direct_consultation(
                cleaned,
                synthesize=self._settings.research_synthesize,
            )
            return RequestResult(
                answer=outcome.report,
                route=decision,
                outcome=outcome,
            )

        if self._settings.research_mode == "operator":
            print(
                f"[*] Planning the request and distributing subtasks "
                f"across {len(ready)} provider(s)..."
            )
            result = await self._operator.run(cleaned)
            return RequestResult(
                answer=self._render_executive(result),
                route=decision,
                research=result,
            )

        print(
            f"[*] Debating across {len(ready)} provider(s), up to "
            f"{self._settings.debate_max_rounds} round(s)..."
        )
        debate = await self._debate.run(
            cleaned, fresh_conversation=new_conversation
        )

        # The user reads ONE concise answer, not five transcripts. The
        # full report and raw responses remain attached for the
        # expandable section and the terminal user who wants them.
        final = await build_final_answer(
            debate,
            synthesize=(
                self._engine.answer_locally
                if self._settings.final_answer_synthesis
                else None
            ),
        )
        return RequestResult(
            answer=final.summary,
            route=decision,
            debate=debate,
            final=final,
        )

    async def open_session(self) -> list[str]:
        """
        Open the persistent research session.

        Launches the browser once and opens one tab per configured
        provider. Idempotent: on an already-open session this returns the
        ready providers without relaunching or reloading anything, which
        is what preserves conversation context between requests.

        Returns:
            Names of providers that are ready.
        """
        already_open = self._session.is_open
        if not already_open:
            print("[*] Launching Chrome...")

        ready = await self._session.open()

        if not already_open:
            for status in await self._session.provider_status():
                if status.ready:
                    print(f"[+] {status.display_name} ready")
                else:
                    print(
                        f"[-] {status.display_name} unavailable: "
                        f"{status.detail or 'not ready'}"
                    )

        self._research_stack_ready = bool(ready)
        return ready

    async def providers_report(self) -> str:
        """
        Summarize provider readiness and conversation state.

        Returns:
            Multi-line human-readable report.
        """
        statuses = await self._session.provider_status()
        if not statuses:
            return (
                self._session.registry.startup_summary()
                + "\n\nSession not open. Run /open to launch it."
            )

        lines = ["Providers:"]
        for status in statuses:
            mark = (
                "ready"
                if status.ready
                else ("PAUSED" if status.paused else "down")
            )
            tag = "" if status.verified else " (unverified selectors)"
            detail = (
                f"  {status.detail}" if status.detail and not status.ready else ""
            )
            lines.append(
                f"  {status.display_name:<12} {mark:<6} "
                f"turns={status.turns}{tag}{detail}"
            )

        disabled = self._session.registry.disabled_names()
        if disabled:
            lines.append(f"Disabled (never launched): {', '.join(disabled)}")

        stats = self._session.stats
        lines.append(
            f"  dispatched={stats.prompts_dispatched} "
            f"received={stats.responses_received} "
            f"failures={stats.provider_failures} "
            f"recoveries={stats.recoveries}"
        )
        return "\n".join(lines)

    def _research_launch_config(self) -> BrowserLaunchConfig:
        """
        Build the launch configuration for provider consultation.

        Returns:
            Launch configuration with headless resolved for first-run
            sign-in.
        """
        launch_config = BrowserLaunchConfig.from_settings(self._settings)
        return launch_config.model_copy(
            update={"headless": self._resolve_headless(launch_config)}
        )

    async def ensure_research_stack(self) -> bool:
        """
        Ensure the persistent research session is available.

        Delegates to the session manager so there is exactly one code path
        that launches the browser and opens provider tabs. Idempotent and
        never raises: when the session cannot be opened the failure is
        logged and research degrades to local-only answers.

        Returns:
            True when at least one provider is ready.
        """
        if self._session.is_open:
            return True

        try:
            return bool(await self.open_session())
        except Exception as exc:
            self._logger.warning(
                "Research session unavailable (%s); complex requests will "
                "be answered by the local model only",
                exc,
            )
            return False

    async def handle_request(self, goal: str) -> RequestResult:
        """
        Handle one user request end to end.

        Routing is adaptive. The FastRouter inspects the request with
        deterministic rules first; only requests it declines to claim
        reach the Decision Engine, and only genuinely complex ones reach
        the orchestrator and the browser. Commands, greetings, and
        arithmetic are answered with no inference at all.

        Args:
            goal: User request or research goal.

        Returns:
            Result carrying the answer and the routing path taken.

        Raises:
            AIResearchOperatorError: If the request cannot be processed.
        """
        if not self._initialized:
            raise AIResearchOperatorError(
                "Application is not initialized; call initialize() first",
                code="APP_NOT_INITIALIZED",
            )

        decision = self._router.route(goal)

        if decision.target is RouteTarget.GREETING:
            return RequestResult(
                answer=_GREETING_REPLY,
                route=decision,
            )

        if decision.target is RouteTarget.COMMAND:
            return RequestResult(answer=_HELP_TEXT, route=decision)

        if decision.target is RouteTarget.CALCULATE:
            return RequestResult(
                answer=self._calculate(decision.payload or goal),
                route=decision,
            )

        if decision.target is RouteTarget.BROWSER:
            return await self._handle_browser_request(goal, decision)

        if decision.target is RouteTarget.LOCAL_ANSWER:
            answer = await self._engine.answer_locally(goal)
            return RequestResult(answer=answer, route=decision)

        if decision.target is RouteTarget.RESEARCH:
            await self._prepare_research()
            outcome = await self._orchestrator.run_research(goal)
            return RequestResult(
                answer=outcome.report,
                route=decision,
                outcome=outcome,
            )

        # DECISION_ENGINE: the router declined to claim this request, so
        # the semantic classifier decides, exactly as before.
        assessment = await self._engine.classify_complexity(goal)
        self._logger.info(
            "Decision Engine classified request as %s (confidence %.2f, "
            "source=%s)",
            assessment.complexity.value,
            assessment.confidence,
            assessment.source,
        )

        if assessment.complexity == TaskComplexity.COMPLEX:
            await self._prepare_research()

        outcome = await self._orchestrator.run_research(goal)
        return RequestResult(
            answer=outcome.report,
            route=decision,
            outcome=outcome,
        )

    async def _prepare_research(self) -> None:
        """
        Ensure the browser research stack is available before research.

        Prints operator guidance when the stack cannot start, so a
        degraded run is visible rather than silent.
        """
        if await self.ensure_research_stack():
            return

        print(
            "[!] Browser research is unavailable; answering with "
            "the local model only.\n"
            "    ResearchOS uses a dedicated automation Chrome "
            "profile, so your normal\n"
            "    Chrome can stay open. Keep the launched window "
            "open, sign in to\n"
            "    chatgpt.com once, and retry your request.\n"
            f"    To reuse an already-running Chrome instead, start it "
            f"with\n"
            f"    --remote-debugging-port="
            f"{self._settings.remote_debug_port}."
        )

    async def _handle_browser_request(
        self,
        goal: str,
        decision: RoutingDecision,
    ) -> "RequestResult":
        """
        Handle a request routed to the browser stack.

        Starts the browser and reports its state. ArchitectOS currently
        exposes browser capability through the ChatGPT worker rather than
        a general automation vocabulary, so the request is then handed to
        the orchestrator with the stack already warm.

        Args:
            goal: Original request text.
            decision: Routing decision that selected this path.

        Returns:
            Result for the request.
        """
        started = await self.ensure_research_stack()
        if not started:
            return RequestResult(
                answer=(
                    "The browser stack could not be started, so this "
                    "request cannot be served. Check /status for details."
                ),
                route=decision,
            )

        health = await self._browser.health_check()
        self._logger.info(
            "Browser ready for routed request (state=%s, pages=%d)",
            health.state.value,
            health.page_count,
        )

        outcome = await self._orchestrator.run_research(goal)
        return RequestResult(
            answer=outcome.report,
            route=decision,
            outcome=outcome,
        )

    def _calculate(self, expression: str) -> str:
        """
        Evaluate an arithmetic request locally.

        Args:
            expression: Expression extracted by the router.

        Returns:
            Formatted result, or an explanation when evaluation fails.
        """
        try:
            value = evaluate_expression(expression)
        except ValidationError as exc:
            return f"Could not evaluate that expression: {exc}"
        return f"{expression.strip()} = {format_result(value)}"

    async def _write_summary(self, system: str, prompt: str) -> str:
        """
        Write prose with the local model on behalf of a report builder.

        Args:
            system: System prompt describing the writing task.
            prompt: The structured material to write from.

        Returns:
            Generated prose.

        Raises:
            BrainError: If generation fails.
        """
        return await self._ollama.generate(
            prompt=prompt,
            system=system,
            num_predict=1400,
            operation="executive_summary",
        )

    @staticmethod
    def _render_executive(result: ResearchResult) -> str:
        """
        Render an executive report for terminal and chat display.

        The structured payload travels separately over the event stream;
        this is the plain-text rendering, which stays short by design.

        Args:
            result: Completed research result.

        Returns:
            Report text.
        """
        report = result.report
        lines = [
            f"{report.headline_label}: {report.headline}",
            f"Confidence: {report.confidence:.0%}",
            "",
            report.summary,
        ]

        if report.evidence_points:
            lines.extend(["", "Evidence:"])
            lines.extend(f"  • {point}" for point in report.evidence_points)

        if report.weaknesses:
            lines.extend(["", "Potential weakness:"])
            lines.extend(f"  • {item}" for item in report.weaknesses)

        if report.alternatives:
            lines.extend(["", "Alternatives:"])
            lines.extend(
                f"  • {alt.priority}: {alt.choice}"
                for alt in report.alternatives
            )

        if report.sources:
            lines.extend(["", "Sources:"])
            lines.extend(
                f"  • {source['url']}" for source in report.sources[:8]
            )

        return "\n".join(lines)

    async def runtime_metrics(self) -> dict[str, object]:
        """
        Assemble engine-side metrics for the Mission Control dashboard.

        Every probe is local and cheap (Ollama health is one HTTP call to
        localhost). Failures degrade to null fields rather than raising,
        because a metrics tick must never disturb research.

        Returns:
            JSON-compatible engine metrics.
        """
        ollama_ok: Optional[bool] = None
        try:
            ollama_ok = await self._ollama.health_check()
        except Exception:
            ollama_ok = False

        browser: dict[str, object] = {"state": "not_started"}
        try:
            health = await self._browser.health_check()
            if health.session_id is not None:
                browser = {
                    "state": health.state.value,
                    "healthy": health.healthy,
                    "pages": health.page_count,
                }
        except Exception:
            browser = {"state": "unknown"}

        stats = self._session.stats
        return {
            "ollamaHealthy": ollama_ok,
            "ollamaModel": self._ollama.model,
            "browser": browser,
            "browserHidden": self._session.browser_hidden,
            "workerCount": len(self._session.ready_providers()),
            "registeredWorkers": len(self._workers.worker_names),
            "sessionOpen": self._session.is_open,
            "paused": self._session.paused_providers,
            "responseTimes": self._session.response_times,
            "stats": {
                "promptsDispatched": stats.prompts_dispatched,
                "responsesReceived": stats.responses_received,
                "providerFailures": stats.provider_failures,
                "recoveries": stats.recoveries,
            },
        }

    async def status(self) -> str:
        """
        Build a human-readable status summary of every subsystem.

        Returns:
            Multi-line status text for terminal display.
        """
        lines: list[str] = ["System status:"]

        ollama_ok = await self._ollama.health_check()
        lines.append(
            f"  Ollama       : "
            f"{'healthy' if ollama_ok else 'UNREACHABLE'} "
            f"({self._settings.ollama_host}, model={self._ollama.model})"
        )

        browser_health = await self._browser.health_check()
        if browser_health.session_id is None:
            lines.append(
                "  Browser      : not started (launches on first "
                "complex request)"
            )
        else:
            lines.append(
                f"  Browser      : "
                f"{'healthy' if browser_health.healthy else 'UNHEALTHY'} "
                f"(state={browser_health.state.value}, "
                f"pages={browser_health.page_count}, "
                f"url={browser_health.current_url or 'n/a'})"
            )

        if not self._workers.worker_names:
            lines.append(
                "  Workers      : none registered (registered on first "
                "complex request)"
            )
        else:
            health = await self._workers.health_check_all()
            for name, snapshot in sorted(health.items()):
                detail = f" - {snapshot.detail}" if snapshot.detail else ""
                lines.append(
                    f"  Worker '{name}': "
                    f"{'healthy' if snapshot.healthy else 'UNHEALTHY'} "
                    f"(state={snapshot.state.value}){detail}"
                )

        return "\n".join(lines)

    def switch_model(self, model: str) -> None:
        """
        Switch the local Ollama model at runtime.

        Args:
            model: Model name to activate.

        Raises:
            BrainError: If the model name is empty.
        """
        self._engine.switch_model(model)

    async def shutdown(self) -> None:
        """
        Release all resources.

        Stops every worker, then the browser session. Shutdown never
        raises; individual failures are logged so the process can always
        exit cleanly.
        """
        self._logger.info("Shutting down AI Research Operator...")

        # Closing the session stops health monitoring, then providers,
        # then the browser, in that order.
        try:
            await self._session.close()
        except Exception as exc:
            self._logger.warning("Session shutdown reported: %s", exc)

        self._research_stack_ready = False
        self._initialized = False
        self._logger.info("Shutdown complete")

    async def run_repl(self) -> None:
        """
        Run the interactive terminal loop until the user exits.

        Reads requests from stdin without blocking the event loop,
        dispatches them through :meth:`handle_request`, and prints the
        final report. Component errors are reported to the user and the
        loop continues; only EOF, an exit command, or KeyboardInterrupt
        end the loop.
        """
        print(_BANNER)

        while True:
            try:
                line = await asyncio.to_thread(input, "research> ")
            except EOFError:
                print()
                break

            request = line.strip()
            if not request:
                continue

            if request.lower() in {"/exit", "/quit", "exit", "quit"}:
                break

            if request.lower() in {"/help", "help"}:
                print(_HELP_TEXT)
                continue

            if request.lower() == "/status":
                print(await self.status())
                continue

            if request.lower() in {"/routes", "/routing"}:
                print(self.routing_report())
                continue

            if request.lower() in {"/providers", "/session"}:
                print(await self.providers_report())
                continue

            if request.lower() == "/conversations":
                print(self._session.conversation_report())
                continue

            if request.lower() == "/open":
                await self.open_session()
                continue

            if request.lower() in {"/reset", "/clear"}:
                reset = await self._session.reset_conversations()
                print(
                    f"Cleared conversation history for "
                    f"{len(reset)} provider(s)."
                    if reset
                    else "No conversations to clear."
                )
                continue

            if request.lower() == "/recover":
                recovered = await self._session.recover_unhealthy()
                print(
                    f"Recovered: {', '.join(recovered)}"
                    if recovered
                    else "All providers healthy."
                )
                continue

            if request.lower() in {"/dashboard", "/status-board"}:
                promoted = await self._session.poll_pending_auth()
                if promoted:
                    print()
                print(self._session.dashboard())
                continue

            if request.lower() in {"/auth", "/login"}:
                await self._handle_auth_command(
                    wait=request.lower() == "/login"
                )
                continue

            if request.lower() == "/profile":
                print(self._session.profile_report())
                continue

            if request.lower() == "/reset-profile":
                await self._handle_reset_profile()
                continue

            if request.lower() == "/resume":
                paused = self._session.paused_providers
                if not paused:
                    print("No providers are paused.")
                    continue
                resumed = await self._session.resume_paused()
                print(
                    f"Resumed: {', '.join(resumed)}"
                    if resumed
                    else "Still blocked: "
                    + ", ".join(self._session.paused_providers)
                )
                continue

            if request.lower().startswith("/research"):
                query = request[len("/research"):].strip()
                if not query:
                    print("Usage: /research <query>")
                    continue
                await self._run_explicit_research(query)
                continue

            if request.lower().startswith("/model"):
                model = request[len("/model"):].strip()
                if not model:
                    print("Usage: /model <name>")
                    continue
                try:
                    self.switch_model(model)
                    print(f"Local model switched to '{model}'")
                except AIResearchOperatorError as exc:
                    print(f"[!] {exc}")
                continue

            await self._run_request(request)

    async def _handle_auth_command(self, wait: bool) -> None:
        """
        Report authentication state, optionally waiting for sign-ins.

        Args:
            wait: When True, wait for outstanding manual sign-ins to
                complete rather than only reporting them.
        """
        if not self._session.is_open and not await self.open_session():
            print("Could not open the session; see /providers.")
            return

        if wait:
            signed_in = await self._session.await_pending_logins()
            if signed_in:
                print(f"[+] Signed in: {', '.join(signed_in)}")

        statuses = await self._session.login_status()
        if not statuses:
            print("No providers registered.")
            return

        print("Authentication:")
        for name, status in statuses.items():
            print(f"  {status.describe()}")

        pending = [s for s in statuses.values() if s.state.needs_human]
        if pending and not wait:
            print("\nRun /login to sign in to the pending providers.")

    async def _handle_reset_profile(self) -> None:
        """
        Delete the persistent browser profile after explicit confirmation.

        The only operation in the system that removes authentication.
        Requires the operator to type the profile name, so a mistyped
        command cannot silently destroy every signed-in session.
        """
        path = self._session.profile_path()
        if path is None:
            print("No persistent profile is configured.")
            return

        print(self._session.profile_report())
        print(
            "\nThis deletes every signed-in session. You will need to log "
            "into each provider again."
        )
        answer = await asyncio.to_thread(
            input, f"Type '{path.name}' to confirm, anything else to abort: "
        )

        if answer.strip() != path.name:
            print("Aborted. The profile was not modified.")
            return

        try:
            removed = await self._session.reset_profile(confirmed=True)
        except AIResearchOperatorError as exc:
            print(f"[!] {exc}")
            return

        print(
            "Profile removed. Run /open to launch a fresh profile and sign "
            "in again."
            if removed
            else "No profile directory existed; nothing was removed."
        )

    async def _run_explicit_research(self, query: str) -> None:
        """
        Execute the explicit research command and print its report.

        Args:
            query: Research query supplied after the command.
        """
        print("Working on it...")
        try:
            result = await self.research_now(query)
        except AIResearchOperatorError as exc:
            print(f"[!] Research failed: {exc}")
            return
        except Exception as exc:  # never kill the REPL
            self._logger.exception("Unexpected error during research")
            print(f"[!] Unexpected error: {exc}")
            return

        print("\n" + "-" * 70)
        print(result.answer)
        print("-" * 70)

        outcome = result.outcome
        if outcome is not None:
            answered = sum(
                1
                for step in outcome.step_results
                for response in step.responses
                if response.success
            )
            total = sum(
                len(step.responses) for step in outcome.step_results
            )
            print(
                f"[route=explicit_research providers={answered}/{total} "
                f"research_id={outcome.research_id}]\n"
            )

    async def _run_request(self, request: str) -> None:
        """
        Execute one research request and print its report.

        Args:
            request: User request text.
        """
        try:
            result = await self.handle_request(request)
        except AIResearchOperatorError as exc:
            print(f"[!] Request failed: {exc}")
            return
        except Exception as exc:  # never kill the REPL
            self._logger.exception("Unexpected error handling request")
            print(f"[!] Unexpected error: {exc}")
            return

        # Instant paths print bare; only substantive answers get a frame.
        if result.outcome is None and len(result.answer) < 200:
            print(result.answer + "\n")
            return

        print("\n" + "-" * 70)
        print(result.answer)
        print("-" * 70)

        trailer = f"[route={result.route.target.value}"
        if result.outcome is not None:
            trailer += (
                f" research_id={result.outcome.research_id}"
                f" steps={len(result.outcome.step_results)}"
                f" consultations="
                f"{result.outcome.consensus_summary.get('consultations', 0)}"
            )
        print(trailer + "]\n")

    def routing_report(self) -> str:
        """
        Summarize how requests have been routed this session.

        Returns:
            Human-readable routing statistics.
        """
        counts = self._router.counts
        if not counts:
            return "No requests routed yet."

        total = sum(counts.values())
        lines = [f"Routing over {total} request(s):"]
        for target, count in sorted(
            counts.items(), key=lambda item: -item[1]
        ):
            share = count / total * 100
            model = "model" if RouteTarget(target).uses_model else "instant"
            lines.append(
                f"  {target:<16} {count:>4}  ({share:5.1f}%)  {model}"
            )

        rate = self._router.model_avoidance_rate
        if rate is not None:
            lines.append(f"  inference avoided: {rate * 100:.1f}%")
        return "\n".join(lines)

    async def _verify_model(self) -> None:
        """
        Resolve the active Ollama model against the installed models.

        Delegates to :meth:`OllamaClient.resolve_model`, which keeps the
        configured model when installed and otherwise falls back to the
        first installed compatible model with a warning instead of
        failing.
        """
        active = await self._ollama.resolve_model()
        self._logger.info("Using Ollama model '%s'", active)

    def _resolve_headless(self, launch_config: BrowserLaunchConfig) -> bool:
        """
        Resolve the headless mode for the research browser.

        Interactive sign-in cannot be completed without a visible window,
        so a profile that has never been signed in is always launched
        headed — even when headless was explicitly requested. Once the
        profile carries state, the configured preference is honoured.

        Args:
            launch_config: Launch configuration for the pending session.

        Returns:
            True when the browser should run headless.
        """
        requested = (
            self._settings.headless
            if self._settings.headless is not None
            else False
        )

        if requested and not self._profile_has_state(launch_config):
            self._logger.warning(
                "Headless was requested but the automation profile has no "
                "saved session; forcing a visible window so the one-time "
                "sign-in can be completed"
            )
            return False

        return requested

    @staticmethod
    def _profile_has_state(launch_config: BrowserLaunchConfig) -> bool:
        """
        Report whether an automation profile carries prior session state.

        A Chrome profile that has been used writes a ``Preferences`` file
        into its profile directory. Its presence is used as the signal
        that a sign-in has plausibly already happened; absence means the
        profile is new.

        Args:
            launch_config: Launch configuration for the pending session.

        Returns:
            True when the profile directory shows evidence of prior use.
            True also when no explicit directory is configured, since no
            claim can be made about an unknown profile.
        """
        if launch_config.user_data_dir is None:
            return True

        profile_path = (
            Path(launch_config.user_data_dir)
            / launch_config.profile_directory
        )
        return (profile_path / "Preferences").exists()


__all__ = [
    "ResearchOperatorApp",
]
