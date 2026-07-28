/**
 * Left navigation rail.
 *
 * Also carries the engine connection indicator, because a disconnected
 * engine explains every other empty panel and should be visible from
 * anywhere in the app.
 */

export type PageId = "chat" | "workers" | "sessions" | "consensus" | "settings";

interface Props {
  page: PageId;
  onNavigate: (page: PageId) => void;
  connected: boolean;
  workerCount: number;
  activeCount: number;
  researching: boolean;
}

const ITEMS: Array<{ id: PageId; label: string; icon: string }> = [
  { id: "chat", label: "Architect Chat", icon: "◆" },
  { id: "workers", label: "Live Workers", icon: "▦" },
  { id: "sessions", label: "Research Timeline", icon: "◇" },
  { id: "consensus", label: "Consensus", icon: "◈" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

export default function Sidebar({
  page,
  onNavigate,
  connected,
  workerCount,
  activeCount,
  researching,
}: Props) {
  return (
    <aside className="w-[248px] shrink-0 h-full flex flex-col border-r border-white/[.06] bg-ink-900/60 backdrop-blur-xl">
      <div className="px-5 pt-6 pb-5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent to-accent-dim grid place-items-center text-ink-950 font-bold text-sm shadow-lg shadow-accent/20">
            A
          </div>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold text-slate-100">
              ArchitectOS
            </div>
            <div className="text-[11px] text-slate-500">Research Engine</div>
          </div>
        </div>
      </div>

      <button
        onClick={() => onNavigate("chat")}
        className="mx-3 mb-4 px-3 py-2.5 rounded-xl text-sm font-medium text-slate-100
                   bg-white/[.06] hover:bg-white/[.10] border border-white/[.07]
                   transition-all duration-200 flex items-center gap-2"
      >
        <span className="text-accent">+</span> New Chat
      </button>

      <nav className="px-3 space-y-1 flex-1 overflow-y-auto">
        {ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={`nav-item w-full ${page === item.id ? "nav-item-active" : ""}`}
          >
            <span className="w-4 text-center opacity-70">{item.icon}</span>
            <span className="flex-1 text-left">{item.label}</span>
            {item.id === "workers" && workerCount > 0 && (
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded-md font-mono ${
                  activeCount > 0
                    ? "bg-accent/20 text-accent-soft"
                    : "bg-white/[.06] text-slate-500"
                }`}
              >
                {activeCount > 0 ? `${activeCount}/${workerCount}` : workerCount}
              </span>
            )}
          </button>
        ))}
      </nav>

      <div className="p-3 mt-2 border-t border-white/[.06]">
        <div className="flex items-center gap-2 px-2 py-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              connected
                ? researching
                  ? "bg-accent animate-pulse-soft"
                  : "bg-ok"
                : "bg-bad"
            }`}
          />
          <span className="text-[11px] text-slate-500">
            {connected
              ? researching
                ? "Researching"
                : "Engine connected"
              : "Engine offline"}
          </span>
        </div>
        {!connected && (
          <p className="px-2 text-[10px] leading-snug text-slate-600">
            Start it with{" "}
            <code className="text-slate-500">python -m scripts.serve</code>
          </p>
        )}
      </div>
    </aside>
  );
}
