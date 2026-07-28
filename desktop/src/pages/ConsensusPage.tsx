/**
 * Consensus view.
 *
 * Shows agreement, disagreement, and merged recommendations for the
 * current run. A run that ended without converging is labelled as such:
 * presenting unresolved positions as consensus would be misleading.
 */

import type { EngineState } from "../lib/types";
import ProgressBar from "../components/ProgressBar";

export default function ConsensusPage({ state }: { state: EngineState }) {
  const consensus = state.consensus;

  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-4 border-b border-white/[.06]">
        <h1 className="text-sm font-semibold text-slate-100">Consensus</h1>
        <p className="text-[11px] text-slate-500">
          {consensus
            ? `Round ${consensus.round} · ${consensus.opinionCount} provider(s) answering`
            : "No analysis yet"}
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        {!consensus ? (
          <p className="pt-20 text-center text-[13px] text-slate-500">
            Run a research request to see cross-provider consensus.
          </p>
        ) : (
          <div className="max-w-3xl mx-auto space-y-5">
            <div className="grid grid-cols-2 gap-4">
              {[
                ["Confidence", consensus.confidence],
                ["Agreement", consensus.agreement],
              ].map(([label, value]) => (
                <div key={String(label)} className="card p-4">
                  <p className="text-[10px] uppercase tracking-wide text-slate-600">
                    {label}
                  </p>
                  <p className="mt-1 text-2xl font-semibold text-slate-100 tabular-nums">
                    {(Number(value) * 100).toFixed(0)}%
                  </p>
                  <ProgressBar
                    value={Number(value)}
                    tone={Number(value) > 0.7 ? "ok" : "warn"}
                    className="mt-3"
                  />
                </div>
              ))}
            </div>

            {consensus.stopReason && (
              <div
                className={`card p-4 ${
                  consensus.converged ? "border-ok/25" : "border-warn/25"
                }`}
              >
                <p className="text-[12px] text-slate-200">
                  {consensus.converged
                    ? "Providers converged."
                    : "Ended without convergence — treat the positions below as unresolved."}
                </p>
                <p className="mt-1 text-[10px] font-mono text-slate-600">
                  {consensus.stopReason}
                </p>
              </div>
            )}

            {consensus.products.length > 0 && (
              <section className="card p-5">
                <h2 className="text-[12px] font-semibold text-slate-200">
                  Merged recommendations
                </h2>
                <div className="mt-3 space-y-2.5">
                  {consensus.products.map((product) => (
                    <div key={product.name} className="flex items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <p className="text-[12px] text-slate-200 truncate">
                          {product.name}
                        </p>
                        <p className="text-[10px] text-slate-500">
                          {product.supporters.join(", ")}
                          {product.dissenters.length > 0 && (
                            <span className="text-warn/70">
                              {" "}
                              · omitted by {product.dissenters.join(", ")}
                            </span>
                          )}
                        </p>
                      </div>
                      <span className="text-[11px] font-mono text-slate-400 tabular-nums shrink-0">
                        {(product.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {consensus.contradictions.length > 0 && (
              <section className="card p-5 border-warn/20">
                <h2 className="text-[12px] font-semibold text-slate-200">
                  Contradictions
                </h2>
                <div className="mt-3 space-y-2">
                  {consensus.contradictions.map((item, index) => (
                    <div key={index} className="text-[11px]">
                      <span className="text-warn/90 font-medium">
                        {item.sourceA} vs {item.sourceB}
                      </span>
                      <p className="text-slate-500 leading-relaxed">
                        {item.description}
                      </p>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
