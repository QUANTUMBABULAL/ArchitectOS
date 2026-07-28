/**
 * Live research timeline.
 *
 * A chronological trace of engine events for the current session. Useful
 * both as a progress view and as a first stop when a run behaves oddly.
 */

import type { EngineState } from "../lib/types";

const KIND_STYLE: Record<string, { dot: string; text: string }> = {
  research: { dot: "bg-accent", text: "text-slate-200" },
  provider: { dot: "bg-ok", text: "text-slate-300" },
  consensus: { dot: "bg-warn", text: "text-slate-300" },
  error: { dot: "bg-bad", text: "text-bad/90" },
};

export default function TimelinePage({ state }: { state: EngineState }) {
  const entries = [...state.timeline].reverse();

  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-4 border-b border-white/[.06]">
        <h1 className="text-sm font-semibold text-slate-100">
          Research Timeline
        </h1>
        <p className="text-[11px] text-slate-500">
          {entries.length} event{entries.length === 1 ? "" : "s"} · newest first
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        {entries.length === 0 ? (
          <p className="pt-20 text-center text-[13px] text-slate-500">
            Nothing yet. Start a research request to see the trace.
          </p>
        ) : (
          <ol className="max-w-2xl mx-auto relative">
            <div className="absolute left-[5px] top-2 bottom-2 w-px bg-white/[.07]" />
            {entries.map((item) => {
              const style = KIND_STYLE[item.kind] ?? KIND_STYLE.provider;
              return (
                <li
                  key={item.id}
                  className="relative pl-6 pb-5 animate-fade-up"
                >
                  <span
                    className={`absolute left-0 top-[6px] w-[11px] h-[11px] rounded-full
                                ring-4 ring-ink-950 ${style.dot}`}
                  />
                  <div className="flex items-baseline justify-between gap-3">
                    <p className={`text-[13px] font-medium ${style.text}`}>
                      {item.label}
                    </p>
                    <time className="text-[10px] font-mono text-slate-600 shrink-0">
                      {new Date(item.at).toLocaleTimeString()}
                    </time>
                  </div>
                  {item.detail && (
                    <p className="mt-0.5 text-[11px] text-slate-500 leading-relaxed">
                      {item.detail}
                    </p>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </div>
  );
}
