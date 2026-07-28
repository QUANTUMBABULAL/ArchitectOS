/**
 * Architect Chat.
 *
 * The user converses with ArchitectOS only — never with an individual
 * provider. Provider activity appears as an inline consultation strip so
 * the conversation stays the focus.
 */

import { useEffect, useRef, useState } from "react";
import type { EngineState, Worker } from "../lib/types";
import ProgressBar from "../components/ProgressBar";

interface Props {
  state: EngineState;
  workers: Worker[];
  onSubmit: (query: string) => void;
}

export default function ChatPage({ state, workers, onSubmit }: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.messages.length, state.researching, state.progress]);

  const submit = () => {
    if (!draft.trim() || state.researching) return;
    onSubmit(draft);
    setDraft("");
  };

  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-4 border-b border-white/[.06] flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold text-slate-100">Architect Chat</h1>
          <p className="text-[11px] text-slate-500">
            One conversation. ArchitectOS consults every provider for you.
          </p>
        </div>
        {state.researching && (
          <span className="text-[11px] text-accent-soft font-mono">
            round {state.round} · {Math.round(state.progress * 100)}%
          </span>
        )}
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {state.messages.length === 0 && !state.researching && (
            <div className="pt-20 text-center animate-fade-up">
              <div className="w-12 h-12 mx-auto rounded-2xl bg-gradient-to-br from-accent to-accent-dim grid place-items-center text-ink-950 font-bold shadow-xl shadow-accent/20">
                A
              </div>
              <h2 className="mt-5 text-lg font-semibold text-slate-200">
                What should we research?
              </h2>
              <p className="mt-1.5 text-[13px] text-slate-500">
                Every enabled provider answers in parallel, then debates
                until they converge.
              </p>
            </div>
          )}

          {state.messages.map((message) => (
            <div key={message.id} className="animate-fade-up">
              <div className="text-[11px] font-medium text-slate-500 mb-1.5">
                {message.role === "user" ? "You" : "ArchitectOS"}
              </div>
              <div
                className={
                  message.role === "user"
                    ? "text-[14px] leading-relaxed text-slate-200"
                    : "text-[14px] leading-relaxed text-slate-300 whitespace-pre-wrap"
                }
              >
                {message.content}
              </div>
            </div>
          ))}

          {state.researching && (
            <div className="animate-fade-up">
              <div className="text-[11px] font-medium text-slate-500 mb-1.5">
                ArchitectOS
              </div>
              <div className="card p-4">
                <p className="text-[13px] text-slate-300">
                  Research started. Consulting providers:
                </p>
                <div className="mt-3 space-y-1.5">
                  {workers.map((worker) => {
                    const done = worker.phase === "Finished";
                    const failed =
                      worker.phase === "Failed" || worker.phase === "Blocked";
                    return (
                      <div
                        key={worker.name}
                        className="flex items-center gap-2 text-[12px]"
                      >
                        <span
                          className={
                            done
                              ? "text-ok"
                              : failed
                                ? "text-bad"
                                : "text-accent animate-pulse-soft"
                          }
                        >
                          {done ? "✓" : failed ? "×" : "○"}
                        </span>
                        <span className="text-slate-400 flex-1">
                          {worker.displayName}
                        </span>
                        <span className="text-[10px] text-slate-600 font-mono">
                          {worker.status}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <ProgressBar value={state.progress} className="mt-4" />
                <p className="mt-2 text-[10px] text-slate-600 font-mono">
                  {Math.round(state.progress * 100)}%
                </p>
              </div>
            </div>
          )}

          {state.error && (
            <div className="card p-4 border-bad/30">
              <p className="text-[12px] text-bad/90">{state.error}</p>
            </div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      <div className="px-8 pb-6">
        <div className="max-w-3xl mx-auto">
          <div className="card p-2 flex items-end gap-2">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submit();
                }
              }}
              rows={1}
              placeholder={
                state.connected
                  ? "Ask ArchitectOS to research something..."
                  : "Engine offline — start python -m scripts.serve"
              }
              disabled={!state.connected}
              className="flex-1 bg-transparent resize-none px-3 py-2.5 text-[14px]
                         text-slate-200 placeholder:text-slate-600 outline-none
                         max-h-40 disabled:opacity-50"
            />
            <button
              onClick={submit}
              disabled={!draft.trim() || state.researching || !state.connected}
              className="px-4 py-2.5 rounded-xl text-[13px] font-medium
                         bg-accent text-ink-950 hover:bg-accent-soft
                         disabled:opacity-30 disabled:cursor-not-allowed
                         transition-all duration-200"
            >
              {state.researching ? "Working" : "Send"}
            </button>
          </div>
          <p className="mt-2 text-[10px] text-slate-600 text-center">
            Shift+Enter for a new line · backend logs stay in your terminal
          </p>
        </div>
      </div>
    </div>
  );
}
