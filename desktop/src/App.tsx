/**
 * Application shell.
 *
 * Owns the single engine connection and routes between pages. State lives
 * in `useEngine`; pages are presentational, so a page can be added
 * without touching transport or reduction logic.
 */

import { useState } from "react";
import Sidebar, { type PageId } from "./components/Sidebar";
import ChatPage from "./pages/ChatPage";
import WorkersPage from "./pages/WorkersPage";
import ConsensusPage from "./pages/ConsensusPage";
import SettingsPage from "./pages/SettingsPage";
import TimelinePage from "./pages/TimelinePage";
import { useEngine } from "./lib/useEngine";

export default function App() {
  const [page, setPage] = useState<PageId>("chat");
  const { state, workers, send, submit } = useEngine();

  const activeCount = workers.filter(
    (worker) => worker.phase === "Thinking" || worker.phase === "Generating",
  ).length;

  return (
    <div className="flex h-full w-full overflow-hidden">
      <Sidebar
        page={page}
        onNavigate={setPage}
        connected={state.connected}
        workerCount={workers.length}
        activeCount={activeCount}
        researching={state.researching}
      />

      <main className="flex-1 min-w-0 flex flex-col">
        {page === "chat" && (
          <ChatPage state={state} workers={workers} onSubmit={submit} />
        )}
        {page === "workers" && (
          <WorkersPage state={state} workers={workers} send={send} />
        )}
        {page === "sessions" && <TimelinePage state={state} />}
        {page === "consensus" && <ConsensusPage state={state} />}
        {page === "settings" && (
          <SettingsPage state={state} workers={workers} send={send} />
        )}
      </main>
    </div>
  );
}
