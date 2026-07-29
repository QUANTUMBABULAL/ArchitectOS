/**
 * Mission Control.
 *
 * Everything the operator is doing and everything it costs, on one
 * screen: the current research stage and per-worker task, host and
 * engine health, and the session-level counters. Values arrive from the
 * engine's Metrics event and the diagnostics command; nothing here polls
 * a browser directly.
 *
 * Every tile carries a text label and a unit — status is never conveyed
 * by colour alone, and numbers wear text tokens rather than series
 * colours.
 */

import { useEffect } from "react";

import type { Command, EngineState, Worker } from "../lib/types";

interface Props {
  state: EngineState;
  workers: Worker[];
  send: (command: Command) => void;
}

const DIAGNOSTICS_REFRESH_MS = 6000;

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function formatRate(bytesPerSec: number | null | undefined): string {
  if (bytesPerSec == null) return "—";
  return `${formatBytes(bytesPerSec)}/s`;
}

function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  }
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function Tile({
  label,
  value,
  sub,
  meter,
  tone = "default",
}: {
  label: string;
  value: string;
  sub?: string;
  meter?: number | null;
  tone?: "default" | "ok" | "warn" | "bad";
}) {
  const valueTone =
    tone === "ok"
      ? "text-ok"
      : tone === "warn"
        ? "text-warn"
        : tone === "bad"
          ? "text-bad"
          : "text-slate-100";
  return (
    <div className="card p-4">
      <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500">
        {label}
      </div>
      <div
        className={`mt-1.5 text-lg font-semibold tabular-nums truncate ${valueTone}`}
        title={value}
      >
        {value}
      </div>
      {sub && (
        <div className="mt-0.5 text-[11px] text-slate-500 truncate" title={sub}>
          {sub}
        </div>
      )}
      {meter != null && (
        <div className="mt-2 h-1 rounded-full bg-white/[.06] overflow-hidden">
          <div
            className="h-full rounded-full bg-accent transition-all duration-500"
            style={{ width: `${Math.min(Math.max(meter, 0), 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: unknown;
}) {
  return (
    <section>
      <h2 className="text-[10px] uppercase tracking-[0.16em] text-slate-600 mb-3">
        {title}
      </h2>
      {children as never}
    </section>
  );
}

export default function DashboardPage({ state, workers, send }: Props) {
  // Browser memory and session age come from the diagnostics probe, so
  // the dashboard keeps it warm while it is the visible page.
  useEffect(() => {
    if (!state.connected) return;
    send({ command: "diagnostics" });
    const timer = window.setInterval(
      () => send({ command: "diagnostics" }),
      DIAGNOSTICS_REFRESH_MS,
    );
    return () => window.clearInterval(timer);
  }, [state.connected, send]);

  const metrics = state.metrics;
  const system = metrics?.system;
  const engine = metrics?.engine;
  const research = metrics?.research;

  const activeWorkers = workers.filter(
    (worker) => worker.phase === "Thinking" || worker.phase === "Generating",
  );
  const currentTask =
    state.plan?.tasks.find((task) => task.status === "running") ??
    state.plan?.tasks.find((task) => task.status === "pending");
  const tasksDone =
    state.plan?.tasks.filter((task) => task.status === "done").length ?? 0;
  const taskTotal = state.plan?.tasks.length ?? 0;

  const responseTimes = Object.values(engine?.responseTimes ?? {});
  const averageResponse = responseTimes.length
    ? responseTimes.reduce((sum, item) => sum + item.avg, 0) /
      responseTimes.length
    : null;

  const browserMemory = state.diagnostics.reduce(
    (sum, record) => sum + (record.jsHeapBytes ?? 0),
    0,
  );
  const oldestSession = state.diagnostics.reduce(
    (max, record) => Math.max(max, record.sessionAgeSeconds ?? 0),
    0,
  );

  const answerChars = workers.reduce(
    (sum, worker) => sum + worker.answerChars,
    0,
  );
  const approxTokens = Math.round(answerChars / 4);

  const healthy = state.diagnostics.filter((record) => record.loggedIn).length;
  const stale =
    metrics != null && Date.now() - metrics.at > 10_000 ? " (stale)" : "";

  return (
    <div className="h-full overflow-y-auto">
      <header className="px-8 py-4 border-b border-white/[.06] flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold text-slate-100">
            Mission Control
          </h1>
          <p className="text-[11px] text-slate-500">
            Live operator, host, and session telemetry{stale}
            {metrics == null && " — waiting for the first metrics tick"}
          </p>
        </div>
        {state.researching && (
          <span className="text-[11px] font-mono text-accent-soft">
            {state.stage} · {Math.round(state.progress * 100)}%
          </span>
        )}
      </header>

      <div className="px-8 py-6 space-y-6">
        <Section title="Research">
          <div className="grid grid-cols-4 gap-3">
            <Tile
              label="Research stage"
              value={state.stage || "Idle"}
              sub={state.stageDetail || "No run in progress"}
              tone={state.researching ? "warn" : "default"}
              meter={state.researching ? state.progress * 100 : null}
            />
            <Tile
              label="Current worker task"
              value={currentTask?.title ?? (state.researching ? "—" : "None")}
              sub={
                currentTask?.assignedTo
                  ? `${currentTask.assignedTo} · ${currentTask.kind}`
                  : activeWorkers.map((w) => w.displayName).join(", ") ||
                    undefined
              }
            />
            <Tile
              label="Research progress"
              value={`${Math.round(state.progress * 100)}%`}
              sub={
                taskTotal > 0
                  ? `${tasksDone}/${taskTotal} subtasks complete`
                  : "no plan yet"
              }
              meter={state.progress * 100}
            />
            <Tile
              label="Consensus progress"
              value={
                state.consensus
                  ? `${Math.round(state.consensus.confidence * 100)}%`
                  : "—"
              }
              sub={
                state.consensus
                  ? `agreement ${Math.round(state.consensus.agreement * 100)}% · ${state.evidenceCount} evidence items`
                  : `${state.evidenceCount} evidence items`
              }
              meter={
                state.consensus ? state.consensus.confidence * 100 : null
              }
            />
          </div>
        </Section>

        <Section title="Host">
          <div className="grid grid-cols-4 gap-3">
            <Tile
              label="CPU"
              value={
                system?.cpuPercent != null
                  ? `${system.cpuPercent.toFixed(0)}%`
                  : "—"
              }
              meter={system?.cpuPercent}
            />
            <Tile
              label="RAM"
              value={
                system?.ramPercent != null
                  ? `${system.ramPercent.toFixed(0)}%`
                  : "—"
              }
              sub={`${formatBytes(system?.ramAvailableBytes)} available of ${formatBytes(system?.ramTotalBytes)}`}
              meter={system?.ramPercent}
            />
            <Tile
              label="GPU"
              value={
                system?.gpu
                  ? `${system.gpu.utilizationPercent.toFixed(0)}%`
                  : "n/a"
              }
              sub={
                system?.gpu
                  ? `${system.gpu.memoryUsedMb.toFixed(0)} / ${system.gpu.memoryTotalMb.toFixed(0)} MB`
                  : "no NVIDIA GPU detected"
              }
              meter={system?.gpu?.utilizationPercent ?? null}
            />
            <Tile
              label="Network"
              value={`↓ ${formatRate(system?.netRecvBytesPerSec)}`}
              sub={`↑ ${formatRate(system?.netSentBytesPerSec)}`}
            />
          </div>
        </Section>

        <Section title="Engine">
          <div className="grid grid-cols-4 gap-3">
            <Tile
              label="Ollama"
              value={
                engine?.ollamaHealthy == null
                  ? "—"
                  : engine.ollamaHealthy
                    ? "Healthy"
                    : "Unreachable"
              }
              sub={engine?.ollamaModel ?? ""}
              tone={
                engine?.ollamaHealthy == null
                  ? "default"
                  : engine.ollamaHealthy
                    ? "ok"
                    : "bad"
              }
            />
            <Tile
              label="Browser status"
              value={engine?.browser?.state ?? "not started"}
              sub={
                engine?.browser?.pages != null
                  ? `${engine.browser.pages} tabs · ${engine.browserHidden ? "hidden" : "visible"}`
                  : "WebSocket " + (state.connected ? "connected" : "offline")
              }
              tone={
                engine?.browser?.healthy === false
                  ? "bad"
                  : engine?.browser?.healthy
                    ? "ok"
                    : "default"
              }
            />
            <Tile
              label="Browser memory"
              value={browserMemory > 0 ? formatBytes(browserMemory) : "—"}
              sub={`JS heap across ${state.diagnostics.length} provider tab(s)`}
            />
            <Tile
              label="Worker status"
              value={`${activeWorkers.length} active`}
              sub={`${engine?.workerCount ?? workers.length} ready of ${engine?.registeredWorkers ?? workers.length} registered`}
            />
          </div>
        </Section>

        <Section title="Performance & cost">
          <div className="grid grid-cols-4 gap-3">
            <Tile
              label="Average response time"
              value={
                averageResponse != null
                  ? `${averageResponse.toFixed(1)}s`
                  : "—"
              }
              sub={`across ${responseTimes.length} provider(s)`}
            />
            <Tile
              label="Estimated remaining"
              value={formatSeconds(research?.estimatedRemainingSeconds)}
              sub={
                research?.averageDurationSeconds != null
                  ? `avg run ${formatSeconds(research.averageDurationSeconds)}`
                  : "no completed runs yet"
              }
            />
            <Tile
              label="Token usage"
              value={approxTokens > 0 ? `≈ ${approxTokens.toLocaleString()}` : "—"}
              sub="estimated from answer length (chars ÷ 4)"
            />
            <Tile
              label="Research cost"
              value="$0.00"
              sub="browser sessions, no metered API calls"
              tone="ok"
            />
          </div>
        </Section>

        <Section title="Provider health">
          <div className="card divide-y divide-white/[.05]">
            {workers.length === 0 && (
              <p className="p-4 text-[12px] text-slate-500">
                No providers yet — open a session first.
              </p>
            )}
            {workers.map((worker) => {
              const timing = engine?.responseTimes?.[worker.name];
              const paused = engine?.paused?.[worker.name];
              const diagnostics = state.diagnostics.find(
                (record) => record.name === worker.name,
              );
              const task = state.plan?.tasks.find(
                (item) => item.assignedTo === worker.name,
              );
              return (
                <div
                  key={worker.name}
                  className="flex items-center gap-3 px-4 py-2.5"
                >
                  <span
                    className={`w-1.5 h-1.5 shrink-0 rounded-full ${
                      paused
                        ? "bg-warn"
                        : worker.phase === "Failed"
                          ? "bg-bad"
                          : diagnostics?.loggedIn
                            ? "bg-ok"
                            : "bg-slate-600"
                    }`}
                  />
                  <span className="text-[12px] text-slate-200 w-24 truncate">
                    {worker.displayName}
                  </span>
                  <span className="text-[11px] text-slate-500 flex-1 truncate">
                    {paused
                      ? `paused — ${paused}`
                      : task
                        ? `${task.title} · ${worker.stage ?? worker.status}`
                        : (worker.stage ?? worker.status)}
                  </span>
                  <span className="text-[11px] font-mono text-slate-600 tabular-nums w-24 text-right">
                    {diagnostics?.sessionAgeSeconds != null
                      ? `age ${formatSeconds(diagnostics.sessionAgeSeconds)}`
                      : "—"}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400 tabular-nums w-40 text-right">
                    {timing
                      ? `${timing.last.toFixed(1)}s · avg ${timing.avg.toFixed(1)}s`
                      : "no answers yet"}
                  </span>
                </div>
              );
            })}
            {workers.length > 0 && (
              <div className="px-4 py-2 text-[11px] text-slate-600">
                {healthy}/{state.diagnostics.length || workers.length} signed
                in · {engine?.stats?.promptsDispatched ?? 0} dispatched ·{" "}
                {engine?.stats?.responsesReceived ?? 0} answered ·{" "}
                {engine?.stats?.providerFailures ?? 0} failed ·{" "}
                {engine?.stats?.recoveries ?? 0} recovered
              </div>
            )}
          </div>
        </Section>

        {state.plan && (
          <Section title="Research plan">
            <div className="card divide-y divide-white/[.05]">
              <div className="px-4 py-2.5 text-[11px] text-slate-500">
                {state.plan.objective}
                <span className="ml-2 text-slate-600">
                  ({state.plan.generatedBy})
                </span>
              </div>
              {state.plan.tasks.map((task) => (
                <div
                  key={`${task.taskId}-${task.assignedTo}`}
                  className="flex items-center gap-3 px-4 py-2"
                >
                  <span
                    className={`text-[11px] w-4 shrink-0 ${
                      task.status === "done"
                        ? "text-ok"
                        : task.status === "failed"
                          ? "text-bad"
                          : task.status === "running"
                            ? "text-accent animate-pulse-soft"
                            : "text-slate-600"
                    }`}
                  >
                    {task.status === "done"
                      ? "✓"
                      : task.status === "failed"
                        ? "×"
                        : "○"}
                  </span>
                  <span className="text-[12px] text-slate-300 flex-1 truncate">
                    {task.title}
                  </span>
                  <span className="text-[10px] font-mono text-slate-600">
                    {task.kind}
                  </span>
                  <span className="text-[11px] text-slate-400 w-20 text-right truncate">
                    {task.assignedTo ?? "—"}
                  </span>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>
    </div>
  );
}
