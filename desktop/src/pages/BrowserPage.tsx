/**
 * Browser Manager.
 *
 * Per-provider diagnostics of the hidden execution engine: sign-in state,
 * current URL, conversation identity, session age, cookies, memory, and
 * the recovery actions. This page is how the user inspects Chrome without
 * Chrome ever being the interface.
 */

import { useEffect, useState } from "react";
import type { Command, EngineState, ProviderDiagnostics } from "../lib/types";

interface Props {
  state: EngineState;
  send: (command: Command) => void;
}

const REFRESH_MS = 5000;

function formatAge(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatHeap(bytes: number | null): string {
  if (bytes == null) return "—";
  return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-[10px] uppercase tracking-wider text-slate-600 shrink-0">
        {label}
      </span>
      <span
        className="text-[11px] font-mono text-slate-300 truncate text-right"
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

function ProviderCard({
  record,
  screenshot,
  send,
}: {
  record: ProviderDiagnostics;
  screenshot?: { image: string; at: number };
  send: (command: Command) => void;
}) {
  const [showShot, setShowShot] = useState(false);

  const loginText =
    record.loggedIn == null
      ? record.tabOpen
        ? "Unknown"
        : "No tab"
      : record.loggedIn
        ? "Logged in"
        : "Signed out";
  const loginTone =
    record.loggedIn == null
      ? "text-slate-500"
      : record.loggedIn
        ? "text-ok"
        : "text-warn";

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              record.paused
                ? "bg-warn"
                : record.loggedIn
                  ? "bg-ok"
                  : record.tabOpen
                    ? "bg-slate-500"
                    : "bg-bad"
            }`}
          />
          <h3 className="text-[13px] font-semibold text-slate-100">
            {record.displayName}
          </h3>
        </div>
        <span className={`text-[11px] font-medium ${loginTone}`}>
          {record.paused ? "Paused" : loginText}
        </span>
      </div>

      {record.pauseReason && (
        <p className="mt-1.5 text-[10px] text-warn/90 leading-snug">
          {record.pauseReason}
        </p>
      )}

      <div className="mt-3">
        <Row label="Tab" value={record.tabOpen ? "Open" : "Closed"} />
        <Row label="URL" value={record.url ?? "—"} />
        <Row
          label="Conversation"
          value={
            record.conversationId
              ? `${record.conversationId.slice(0, 18)} · ${record.turns} turns`
              : record.turns > 0
                ? `${record.turns} turns`
                : "fresh"
          }
        />
        <Row label="Session age" value={formatAge(record.sessionAgeSeconds)} />
        <Row
          label="Cookies"
          value={
            record.cookieCount != null ? String(record.cookieCount) : "—"
          }
        />
        <Row label="JS heap" value={formatHeap(record.jsHeapBytes)} />
        <Row
          label="Last activity"
          value={
            record.lastActivity
              ? new Date(record.lastActivity).toLocaleTimeString()
              : "—"
          }
        />
        <Row
          label="Response time"
          value={
            record.responseTimes
              ? `last ${record.responseTimes.last.toFixed(1)}s · avg ${record.responseTimes.avg.toFixed(1)}s`
              : "—"
          }
        />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-1.5">
        <button className="btn-ghost" onClick={() => send({ command: "recover" })}>
          Recover
        </button>
        <button
          className="btn-ghost"
          onClick={() => {
            send({ command: "screenshot", provider: record.name });
            setShowShot(true);
          }}
        >
          Screenshot
        </button>
        <button
          className="btn-ghost"
          onClick={() => send({ command: "reloadProvider", provider: record.name })}
        >
          Reload session
        </button>
        <button
          className="btn-ghost"
          onClick={() => send({ command: "focusProvider", provider: record.name })}
        >
          Open browser
        </button>
      </div>

      {showShot && screenshot && (
        <div className="mt-3">
          <img
            src={`data:image/png;base64,${screenshot.image}`}
            alt={`${record.displayName} tab`}
            className="rounded-lg border border-white/[.08] w-full"
          />
          <div className="mt-1 flex items-center justify-between">
            <span className="text-[10px] text-slate-600">
              {new Date(screenshot.at).toLocaleTimeString()}
            </span>
            <button
              className="text-[10px] text-slate-500 hover:text-slate-300"
              onClick={() => setShowShot(false)}
            >
              Hide
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function BrowserPage({ state, send }: Props) {
  useEffect(() => {
    if (!state.connected) return;
    send({ command: "diagnostics" });
    const timer = window.setInterval(
      () => send({ command: "diagnostics" }),
      REFRESH_MS,
    );
    return () => window.clearInterval(timer);
  }, [state.connected, send]);

  return (
    <div className="h-full overflow-y-auto">
      <header className="px-8 py-4 border-b border-white/[.06] flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold text-slate-100">
            Browser Manager
          </h1>
          <p className="text-[11px] text-slate-500">
            The hidden execution engine, one card per provider
            {state.diagnosticsAt &&
              ` · refreshed ${new Date(state.diagnosticsAt).toLocaleTimeString()}`}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            className="btn-ghost"
            onClick={() =>
              send({
                command: state.browserHidden ? "showBrowser" : "hideBrowser",
              })
            }
          >
            {state.browserHidden ? "Show browser" : "Hide browser"}
          </button>
          <button
            className="btn-ghost"
            onClick={() => send({ command: "diagnostics" })}
          >
            Refresh
          </button>
        </div>
      </header>

      <div className="px-8 py-6">
        {state.diagnostics.length === 0 ? (
          <div className="card p-6 text-center">
            <p className="text-[13px] text-slate-400">
              {state.connected
                ? "No diagnostics yet — open a session first."
                : "Engine offline."}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
            {state.diagnostics.map((record) => (
              <ProviderCard
                key={record.name}
                record={record}
                screenshot={state.screenshots[record.name]}
                send={send}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
