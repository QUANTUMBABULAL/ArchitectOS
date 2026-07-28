/**
 * One provider's live card.
 *
 * Renders engine events only. Nothing here embeds, screenshots, or
 * iframes a browser: Chrome stays an implementation detail the user never
 * interacts with.
 */

import type { Worker } from "../lib/types";
import ProgressBar from "./ProgressBar";

interface Props {
  worker: Worker;
  onSelect?: (name: string) => void;
  selected?: boolean;
}

const PHASE_TONE: Record<string, "accent" | "ok" | "warn" | "bad"> = {
  Idle: "accent",
  Waiting: "warn",
  Thinking: "accent",
  Generating: "accent",
  Finished: "ok",
  Failed: "bad",
  Blocked: "warn",
};

const DOT: Record<string, string> = {
  Idle: "bg-slate-600",
  Waiting: "bg-warn animate-pulse-soft",
  Thinking: "bg-accent animate-pulse-soft",
  Generating: "bg-accent animate-pulse-soft",
  Finished: "bg-ok",
  Failed: "bg-bad",
  Blocked: "bg-warn",
};

export default function WorkerCard({ worker, onSelect, selected }: Props) {
  const tone = PHASE_TONE[worker.phase] ?? "accent";

  return (
    <button
      onClick={() => onSelect?.(worker.name)}
      className={`card w-full text-left p-4 animate-fade-up transition-all duration-300
                  hover:border-white/[.12] hover:-translate-y-[1px]
                  ${selected ? "border-accent/40 ring-1 ring-accent/20" : ""}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${DOT[worker.phase]}`} />
            <h3 className="text-[13px] font-semibold text-slate-100 truncate">
              {worker.displayName}
            </h3>
          </div>
          <p className="mt-1 text-[11px] text-slate-500 truncate">
            {worker.status}
          </p>
        </div>

        {worker.elapsedSeconds > 0 && (
          <span className="text-[10px] font-mono text-slate-500 shrink-0 tabular-nums">
            {worker.elapsedSeconds.toFixed(1)}s
          </span>
        )}
      </div>

      <ProgressBar value={worker.progress} tone={tone} className="mt-3" />

      <div className="mt-3 flex items-center justify-between text-[10px] text-slate-600">
        <span className="font-mono">
          {worker.answerChars > 0
            ? `${worker.answerChars.toLocaleString()} chars`
            : worker.phase === "Blocked"
              ? worker.authState.replace(/_/g, " ")
              : "—"}
        </span>
        {worker.turn > 0 && <span className="font-mono">turn {worker.turn}</span>}
      </div>

      {worker.error && (
        <p className="mt-2 text-[10px] leading-snug text-bad/80 line-clamp-2">
          {worker.error}
        </p>
      )}
    </button>
  );
}
