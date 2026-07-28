/**
 * Settings and developer pane.
 *
 * Read-mostly by design. Provider enablement, the Chrome profile, and
 * parallelism live in the engine's .env so a single source of truth
 * governs a run; the UI surfaces current values and offers the actions the
 * engine exposes as commands.
 */

import type { Command, EngineState, Worker } from "../lib/types";

interface Props {
  state: EngineState;
  workers: Worker[];
  send: (command: Command) => void;
}

export default function SettingsPage({ state, workers, send }: Props) {
  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-4 border-b border-white/[.06]">
        <h1 className="text-sm font-semibold text-slate-100">Settings</h1>
        <p className="text-[11px] text-slate-500">
          Engine configuration lives in .env — shown here as the engine
          reports it
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          <section className="card p-5">
            <h2 className="text-[12px] font-semibold text-slate-200">
              Providers
            </h2>
            <div className="mt-3 space-y-2">
              {workers.map((worker) => (
                <div
                  key={worker.name}
                  className="flex items-center justify-between text-[12px]"
                >
                  <span className="text-slate-300">{worker.displayName}</span>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-md ${
                      worker.authState === "READY"
                        ? "bg-ok/15 text-ok"
                        : "bg-warn/15 text-warn"
                    }`}
                  >
                    {worker.authState.replace(/_/g, " ")}
                  </span>
                </div>
              ))}
              {state.disabled.map((name) => (
                <div
                  key={name}
                  className="flex items-center justify-between text-[12px]"
                >
                  <span className="text-slate-500">{name}</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-white/[.05] text-slate-600">
                    DISABLED
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-3 text-[10px] text-slate-600 leading-relaxed">
              Change with ENABLED_PROVIDERS and DISABLED_PROVIDERS in .env,
              then restart the engine.
            </p>
          </section>

          <section className="card p-5">
            <h2 className="text-[12px] font-semibold text-slate-200">
              Authentication
            </h2>
            <p className="mt-2 text-[11px] text-slate-500 leading-relaxed">
              Sessions live only inside the ArchitectOS Chrome profile.
              ArchitectOS never stores usernames, passwords, or tokens. Sign
              in once per provider in the browser window.
            </p>
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => send({ command: "checkAuth" })}
                className="px-3 py-1.5 rounded-lg text-[11px] text-slate-300
                           bg-white/[.05] hover:bg-white/[.09] border border-white/[.06]
                           transition-colors"
              >
                Re-check sign-ins
              </button>
              <button
                onClick={() => send({ command: "resetConversations" })}
                className="px-3 py-1.5 rounded-lg text-[11px] text-slate-300
                           bg-white/[.05] hover:bg-white/[.09] border border-white/[.06]
                           transition-colors"
              >
                Clear conversations
              </button>
            </div>
            <p className="mt-3 text-[10px] text-slate-600">
              Removing sign-ins is deliberately terminal-only:{" "}
              <code className="text-slate-500">/reset-profile</code>
            </p>
          </section>

          <section className="card p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-[12px] font-semibold text-slate-200">
                Developer mode
              </h2>
              <span className="text-[10px] text-slate-600">
                mirrored from the terminal
              </span>
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
              The terminal remains the authoritative log. These are copies.
            </p>
            <div className="mt-3 h-64 overflow-y-auto rounded-xl bg-ink-950/70 border border-white/[.05] p-3 font-mono text-[10px] leading-relaxed">
              {state.logs.length === 0 ? (
                <p className="text-slate-600">No log lines yet.</p>
              ) : (
                state.logs.map((line) => (
                  <div key={line.id} className="flex gap-2">
                    <span
                      className={
                        line.level === "ERROR"
                          ? "text-bad"
                          : line.level === "WARNING"
                            ? "text-warn"
                            : "text-slate-600"
                      }
                    >
                      {line.level.slice(0, 4)}
                    </span>
                    <span className="text-slate-500 break-all">
                      {line.message}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
