/**
 * Architect Chat.
 *
 * The user converses with ArchitectOS only — never with an individual
 * provider. Provider activity appears as an inline consultation strip so
 * the conversation stays the focus.
 */

import { useEffect, useRef, useState } from "react";
import type { EngineState, FinalAnswer, Worker } from "../lib/types";
import ExecutiveReportCard from "../components/ExecutiveReportCard";
import ProgressBar from "../components/ProgressBar";

/**
 * The single concise research result: recommendation, confidence,
 * supporters, key disagreement, sources — with the five raw provider
 * responses tucked behind an expandable section.
 */
function FinalAnswerCard({ final }: { final: FinalAnswer }) {
  const [showRaw, setShowRaw] = useState(false);
  const confidence = Math.round(final.confidence * 100);
  const rawEntries = Object.entries(final.rawAnswers);

  return (
    <div className="card p-5 mt-3">
      <div className="flex items-center gap-3 flex-wrap">
        <span
          className={`text-[11px] font-semibold px-2 py-0.5 rounded-md ${
            final.converged
              ? "bg-ok/15 text-ok"
              : "bg-warn/15 text-warn"
          }`}
        >
          {final.converged ? "Consensus" : "No full consensus"}
        </span>
        <span className="text-[11px] font-mono text-slate-400 tabular-nums">
          confidence {confidence}%
        </span>
        <span className="text-[11px] text-slate-500">
          {final.rounds} round{final.rounds === 1 ? "" : "s"}
        </span>
      </div>

      {final.supporting.length > 0 && (
        <p className="mt-2.5 text-[11px] text-slate-500">
          Supported by{" "}
          <span className="text-slate-300">
            {final.supporting.join(", ")}
          </span>
          {final.opposing.length > 0 && (
            <>
              {" · disagreeing: "}
              <span className="text-warn/90">
                {final.opposing.join(", ")}
              </span>
            </>
          )}
        </p>
      )}

      {final.disagreements.length > 0 && (
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          <span className="text-slate-400 font-medium">
            Key disagreement:
          </span>{" "}
          {final.disagreements[0]}
        </p>
      )}

      {final.sources.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {final.sources.slice(0, 8).map((source) => (
            <a
              key={source.url}
              href={source.url}
              target="_blank"
              rel="noreferrer"
              className="text-[10px] px-2 py-1 rounded-md bg-white/[.05]
                         border border-white/[.07] text-slate-400
                         hover:text-accent-soft hover:border-accent/30
                         transition-colors truncate max-w-[220px]"
              title={source.url}
            >
              {source.title}
            </a>
          ))}
        </div>
      )}

      {rawEntries.length > 0 && (
        <div className="mt-4 border-t border-white/[.06] pt-3">
          <button
            onClick={() => setShowRaw((value) => !value)}
            className="text-[11px] text-slate-500 hover:text-slate-300
                       transition-colors flex items-center gap-1.5"
          >
            <span
              className={`inline-block transition-transform ${
                showRaw ? "rotate-90" : ""
              }`}
            >
              ▸
            </span>
            {showRaw ? "Hide" : "Show"} raw provider responses (
            {rawEntries.length})
          </button>
          {showRaw && (
            <div className="mt-3 space-y-3">
              {rawEntries.map(([name, answer]) => (
                <details
                  key={name}
                  className="rounded-xl bg-ink-900/60 border border-white/[.05]"
                >
                  <summary className="px-3 py-2 text-[12px] font-medium text-slate-300 cursor-pointer select-none">
                    {name}
                    <span className="ml-2 text-[10px] text-slate-600 font-mono">
                      {answer.length.toLocaleString()} chars
                    </span>
                  </summary>
                  <div className="px-3 pb-3 text-[12px] leading-relaxed text-slate-400 whitespace-pre-wrap max-h-80 overflow-y-auto">
                    {answer}
                  </div>
                </details>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

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
              {/* An operator result renders as its report card; the
                  plain text form would duplicate it. */}
              {!message.research && (
                <div
                  className={
                    message.role === "user"
                      ? "text-[14px] leading-relaxed text-slate-200"
                      : "text-[14px] leading-relaxed text-slate-300 whitespace-pre-wrap"
                  }
                >
                  {message.content}
                </div>
              )}
              {message.research && (
                <ExecutiveReportCard research={message.research} />
              )}
              {message.final && !message.research && (
                <FinalAnswerCard final={message.final} />
              )}
            </div>
          ))}

          {state.researching && (
            <div className="animate-fade-up">
              <div className="text-[11px] font-medium text-slate-500 mb-1.5">
                ArchitectOS
              </div>
              <div className="card p-4">
                <p className="text-[13px] text-slate-300">
                  {state.stage
                    ? `${state.stage}${state.stageDetail ? ` — ${state.stageDetail}` : ""}`
                    : "Research started."}
                </p>

                {state.plan && (
                  <div className="mt-3 space-y-1">
                    {state.plan.tasks.map((task) => (
                      <div
                        key={`${task.taskId}-${task.assignedTo}`}
                        className="flex items-center gap-2 text-[12px]"
                      >
                        <span
                          className={
                            task.status === "done"
                              ? "text-ok"
                              : task.status === "failed"
                                ? "text-bad"
                                : "text-slate-600"
                          }
                        >
                          {task.status === "done"
                            ? "✓"
                            : task.status === "failed"
                              ? "×"
                              : "○"}
                        </span>
                        <span className="text-slate-400 flex-1 truncate">
                          {task.title}
                        </span>
                        <span className="text-[10px] text-slate-600 font-mono">
                          {task.assignedTo ?? "—"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

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
