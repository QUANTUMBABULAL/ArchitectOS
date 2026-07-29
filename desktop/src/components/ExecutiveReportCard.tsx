/**
 * The executive report — the deliverable of one research run.
 *
 * The decision comes first and is readable in seconds: headline,
 * confidence, the evidence behind it, what argues against it, what to
 * choose instead under a different priority, and the sources. Everything
 * upstream of that — extracted evidence and the raw provider answers —
 * is collapsed by default, because a research operator hands over a
 * recommendation, not five transcripts.
 */

import { useState } from "react";

import type { ResearchResultView } from "../lib/types";

interface Props {
  research: ResearchResultView;
}

function confidenceTone(confidence: number): string {
  if (confidence >= 0.75) return "text-ok";
  if (confidence >= 0.5) return "text-accent-soft";
  return "text-warn";
}

export default function ExecutiveReportCard({ research }: Props) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  const report = research.report;
  const confidence = Math.round(report.confidence * 100);
  const rawEntries = Object.entries(research.evidence.rawAnswers);
  const evidence = research.evidence.items;

  return (
    <div className="card mt-3 overflow-hidden">
      <div className="px-5 pt-5 pb-4 border-b border-white/[.06]">
        <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
          {report.headlineLabel}
        </div>
        <h3 className="mt-1 text-[19px] font-semibold text-slate-100 leading-tight">
          {report.headline}
        </h3>

        <div className="mt-3 flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span
              className={`text-[13px] font-semibold tabular-nums ${confidenceTone(report.confidence)}`}
            >
              {confidence}%
            </span>
            <span className="text-[11px] text-slate-500">confidence</span>
          </div>
          <div className="h-1 w-28 rounded-full bg-white/[.06] overflow-hidden">
            <div
              className="h-full rounded-full bg-accent"
              style={{ width: `${confidence}%` }}
            />
          </div>
          <span className="text-[11px] text-slate-600">
            {research.plan.tasks.length} subtasks ·{" "}
            {research.evidence.providers.length} sources ·{" "}
            {Math.round(research.elapsedSeconds)}s
          </span>
        </div>

        {report.supportingProviders.length > 0 && (
          <p className="mt-2 text-[11px] text-slate-500">
            Supported by{" "}
            <span className="text-slate-300">
              {report.supportingProviders.join(", ")}
            </span>
            {report.dissentingProviders.length > 0 && (
              <>
                {" · other views from "}
                <span className="text-warn/90">
                  {report.dissentingProviders.join(", ")}
                </span>
              </>
            )}
          </p>
        )}
      </div>

      <div className="px-5 py-4 text-[13.5px] leading-relaxed text-slate-300 whitespace-pre-wrap">
        {report.summary}
      </div>

      {report.evidencePoints.length > 0 && (
        <div className="px-5 pb-4">
          <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-2">
            Evidence
          </div>
          <ul className="space-y-1.5">
            {report.evidencePoints.map((point) => (
              <li key={point} className="flex gap-2 text-[12.5px] text-slate-400">
                <span className="text-ok shrink-0">•</span>
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.weaknesses.length > 0 && (
        <div className="px-5 pb-4">
          <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-2">
            Potential weakness
          </div>
          <ul className="space-y-1.5">
            {report.weaknesses.map((item) => (
              <li key={item} className="flex gap-2 text-[12.5px] text-slate-400">
                <span className="text-warn shrink-0">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.alternatives.length > 0 && (
        <div className="px-5 pb-4">
          <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-2">
            Alternatives
          </div>
          <div className="grid grid-cols-2 gap-2">
            {report.alternatives.map((alt) => (
              <div
                key={`${alt.priority}-${alt.choice}`}
                className="rounded-lg bg-white/[.03] border border-white/[.05] px-3 py-2"
              >
                <div className="text-[10px] text-slate-500 capitalize">
                  {alt.priority}
                </div>
                <div className="text-[12.5px] text-slate-200">{alt.choice}</div>
                {alt.rationale && (
                  <div className="text-[10px] text-slate-600 truncate">
                    {alt.rationale}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {report.sources.length > 0 && (
        <div className="px-5 pb-4">
          <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-2">
            Sources
          </div>
          <div className="flex flex-wrap gap-1.5">
            {report.sources.map((source) => (
              <a
                key={source.url}
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="text-[10.5px] px-2 py-1 rounded-md bg-white/[.05]
                           border border-white/[.07] text-slate-400
                           hover:text-accent-soft hover:border-accent/30
                           transition-colors truncate max-w-[220px]"
                title={source.url}
              >
                {source.title}
              </a>
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-white/[.06] px-5 py-3 space-y-2">
        <button
          onClick={() => setShowEvidence((value) => !value)}
          className="text-[11px] text-slate-500 hover:text-slate-300
                     transition-colors flex items-center gap-1.5"
        >
          <span
            className={`inline-block transition-transform ${showEvidence ? "rotate-90" : ""}`}
          >
            ▸
          </span>
          {showEvidence ? "Hide" : "Show"} extracted evidence ({evidence.length})
        </button>

        {showEvidence && (
          <div className="space-y-2 pt-1">
            {evidence.map((item, index) => (
              <div
                key={`${item.provider}-${index}`}
                className="rounded-lg bg-ink-900/60 border border-white/[.05] px-3 py-2"
              >
                <div className="text-[12px] text-slate-300">{item.fact}</div>
                <div className="mt-1 flex items-center gap-2 flex-wrap text-[10px] text-slate-600">
                  <span className="text-accent-soft/80">{item.provider}</span>
                  <span>· {item.taskTitle}</span>
                  {item.source && <span>· {item.source}</span>}
                  <span>· {Math.round(item.confidence * 100)}% stated</span>
                  {item.links.map((link) => (
                    <a
                      key={link}
                      href={link}
                      target="_blank"
                      rel="noreferrer"
                      className="text-slate-500 hover:text-accent-soft truncate max-w-[180px]"
                    >
                      {link}
                    </a>
                  ))}
                </div>
                {item.caveats.length > 0 && (
                  <div className="mt-1 text-[10.5px] text-warn/80">
                    caveat: {item.caveats.join("; ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <button
          onClick={() => setShowRaw((value) => !value)}
          className="text-[11px] text-slate-500 hover:text-slate-300
                     transition-colors flex items-center gap-1.5"
        >
          <span
            className={`inline-block transition-transform ${showRaw ? "rotate-90" : ""}`}
          >
            ▸
          </span>
          {showRaw ? "Hide" : "Show"} raw provider answers ({rawEntries.length})
        </button>

        {showRaw && (
          <div className="space-y-2 pt-1">
            {rawEntries.map(([label, answer]) => (
              <details
                key={label}
                className="rounded-lg bg-ink-900/60 border border-white/[.05]"
              >
                <summary className="px-3 py-2 text-[12px] font-medium text-slate-300 cursor-pointer select-none">
                  {label}
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
    </div>
  );
}
