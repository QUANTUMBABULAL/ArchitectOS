/**
 * Live Workers.
 *
 * Every provider is shown simultaneously as a card driven by engine
 * events. Selecting one opens its detail panel.
 */

import { useState } from "react";
import type { EngineState, Command, Worker } from "../lib/types";
import WorkerCard from "../components/WorkerCard";
import ProgressBar from "../components/ProgressBar";

interface Props {
  state: EngineState;
  workers: Worker[];
  send: (command: Command) => void;
}

export default function WorkersPage({ state, workers, send }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const active = selected ? state.workers[selected] : null;

  return (
    <div className="flex h-full">
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="px-8 py-4 border-b border-white/[.06] flex items-center justify-between">
          <div>
            <h1 className="text-sm font-semibold text-slate-100">Live Workers</h1>
            <p className="text-[11px] text-slate-500">
              {workers.length} provider{workers.length === 1 ? "" : "s"} · updates
              stream from the engine
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => send({ command: "checkAuth" })}
              className="px-3 py-1.5 rounded-lg text-[11px] text-slate-300
                         bg-white/[.05] hover:bg-white/[.09] border border-white/[.06]
                         transition-colors"
            >
              Check sign-ins
            </button>
            <button
              onClick={() => send({ command: "recover" })}
              className="px-3 py-1.5 rounded-lg text-[11px] text-slate-300
                         bg-white/[.05] hover:bg-white/[.09] border border-white/[.06]
                         transition-colors"
            >
              Recover tabs
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6">
          {workers.length === 0 ? (
            <div className="pt-24 text-center">
              <p className="text-[13px] text-slate-500">No providers yet.</p>
              <button
                onClick={() => send({ command: "openSession" })}
                className="mt-4 px-4 py-2 rounded-xl text-[12px] font-medium
                           bg-accent text-ink-950 hover:bg-accent-soft transition-colors"
              >
                Open browser session
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {workers.map((worker) => (
                <WorkerCard
                  key={worker.name}
                  worker={worker}
                  selected={selected === worker.name}
                  onSelect={(name) =>
                    setSelected(name === selected ? null : name)
                  }
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {active && (
        <aside className="w-[340px] shrink-0 border-l border-white/[.06] bg-ink-900/50 backdrop-blur-xl overflow-y-auto">
          <div className="p-5">
            <div className="flex items-start justify-between">
              <h2 className="text-sm font-semibold text-slate-100">
                {active.displayName}
              </h2>
              <button
                onClick={() => setSelected(null)}
                className="text-slate-500 hover:text-slate-300 text-lg leading-none"
              >
                ×
              </button>
            </div>

            <ProgressBar value={active.progress} className="mt-4" />

            <dl className="mt-5 space-y-3">
              {[
                ["Current phase", active.phase],
                ["Status", active.status],
                ["Authentication", active.authState.replace(/_/g, " ")],
                ["Elapsed", `${active.elapsedSeconds.toFixed(1)}s`],
                ["Response size", `${active.answerChars.toLocaleString()} chars`],
                ["Prompt size", `${active.promptChars.toLocaleString()} chars`],
                ["Conversation turn", String(active.turn)],
                ["Current URL", active.currentUrl ?? "—"],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt className="text-[10px] uppercase tracking-wide text-slate-600">
                    {label}
                  </dt>
                  <dd className="text-[12px] text-slate-300 break-words">
                    {value}
                  </dd>
                </div>
              ))}
            </dl>

            {active.detail && (
              <div className="mt-5 p-3 rounded-xl bg-warn/[.08] border border-warn/20">
                <p className="text-[11px] text-warn/90 leading-relaxed">
                  {active.detail}
                </p>
              </div>
            )}

            {active.error && (
              <div className="mt-3 p-3 rounded-xl bg-bad/[.08] border border-bad/20">
                <p className="text-[10px] uppercase tracking-wide text-bad/70">
                  Last error
                </p>
                <p className="mt-1 text-[11px] text-bad/90 leading-relaxed">
                  {active.error}
                </p>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}
