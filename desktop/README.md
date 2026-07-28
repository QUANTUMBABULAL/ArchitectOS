# ArchitectOS Desktop Interface

The desktop UI is a **client** of the Python engine, not a replacement for
it. The engine is unchanged: it still runs standalone, still owns all
research logic, and still prints every log line to the terminal.

```
You  →  Desktop UI  →  WebSocket  →  Python engine  →  Browser workers
                                                       ChatGPT · Gemini
                                                       Claude · Grok · DeepSeek
```

Chrome is an implementation detail. The UI never embeds, screenshots, or
iframes a browser — providers emit events and the UI renders them.

## Running it

Two processes. The terminal stays open and keeps logging.

**1. Engine** (from the repository root)

```
.venv\Scripts\python -m scripts.serve
```

Serves `ws://127.0.0.1:8777/ws` and continues printing logs. Loopback only
by design: the engine drives an authenticated browser session and must not
be exposed to a network.

**2. Desktop app**

```
cd desktop
npm install
npm run tauri:dev
```

For browser-only development without the Tauri shell, `npm run dev` and
open `http://localhost:5173`.

The terminal REPL still works unchanged (`python -m src`) — it is simply
another client of the same engine.

## Structure

| Path | Purpose |
| --- | --- |
| `src/lib/types.ts` | Event contract mirroring `src/events/models.py` |
| `src/lib/useEngine.ts` | Socket, reconnection, and event reduction |
| `src/components/` | Presentational pieces (sidebar, worker card, progress) |
| `src/pages/` | Chat, Live Workers, Timeline, Consensus, Settings |
| `src-tauri/` | Desktop shell — hosts the web view and nothing else |

All engine knowledge lives in `useEngine.ts`. Components take props, so a
page can be added without touching transport.

## Adding an event

1. Add the case to `EventType` in `src/events/models.py`.
2. Emit it through the injected `EventEmitter`.
3. Add the literal to `EventType` in `src/lib/types.ts`.
4. Handle it in the reducer in `useEngine.ts`.

Unknown event types fall through the reducer's default branch rather than
crashing, so a version mismatch degrades instead of breaking.

## Notes on honesty in the UI

Worker progress is rendered as **discrete phase blocks**, not a percentage.
The engine drives a real browser and cannot observe token-level progress,
so a precise-looking bar would be inventing precision it does not have.

A debate that ends without converging is labelled as unresolved on the
Consensus page. Presenting unresolved positions as agreement would
misrepresent the result.
