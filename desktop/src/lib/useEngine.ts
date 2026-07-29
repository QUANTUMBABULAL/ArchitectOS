/**
 * Engine connection and state reduction.
 *
 * One WebSocket carries every engine event. This hook owns the socket,
 * reconnects when it drops, and folds the event stream into the state the
 * pages render. All engine knowledge lives here so components stay
 * presentational.
 *
 * Reduction is additive and order-tolerant: an event for an unknown
 * provider creates it rather than being dropped, because the engine's
 * replay buffer may deliver a provider's first event after a later one.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import type {
  AuthState,
  ChatMessage,
  Command,
  EngineEvent,
  EngineState,
  FinalAnswer,
  MetricsSnapshot,
  PlanTask,
  ProviderDiagnostics,
  ResearchPlanView,
  ResearchResultView,
  TimelineEntry,
  Worker,
  WorkerPhase,
} from "./types";

const DEFAULT_URL = "ws://127.0.0.1:8777/ws";
const RECONNECT_MS = 1500;
const MAX_TIMELINE = 200;
const MAX_LOGS = 400;

const initialState: EngineState = {
  connected: false,
  engineReady: false,
  workers: {},
  order: [],
  disabled: [],
  timeline: [],
  messages: [],
  consensus: null,
  logs: [],
  researching: false,
  progress: 0,
  round: 0,
  question: "",
  error: null,
  stage: "",
  stageDetail: "",
  plan: null,
  evidenceCount: 0,
  metrics: null,
  browserHidden: false,
  diagnostics: [],
  diagnosticsAt: null,
  screenshots: {},
};

/** Human-readable status text per phase, shown on worker cards. */
const PHASE_STATUS: Record<WorkerPhase, string> = {
  Idle: "Idle",
  Waiting: "Waiting...",
  Thinking: "Thinking...",
  Generating: "Generating...",
  Finished: "Finished",
  Failed: "Failed",
  Blocked: "Action needed",
};

/** Nominal progress per phase. The engine cannot report true token
 *  progress from a browser, so these are honest coarse stages rather
 *  than a fake percentage. */
const PHASE_PROGRESS: Record<WorkerPhase, number> = {
  Idle: 0,
  Waiting: 0.15,
  Thinking: 0.45,
  Generating: 0.75,
  Finished: 1,
  Failed: 1,
  Blocked: 0,
};

function makeWorker(name: string, displayName?: string): Worker {
  return {
    name,
    displayName: displayName ?? name,
    phase: "Idle",
    authState: "UNKNOWN",
    status: PHASE_STATUS.Idle,
    elapsedSeconds: 0,
    answerChars: 0,
    promptChars: 0,
    turn: 0,
    progress: 0,
  };
}

function entry(
  kind: TimelineEntry["kind"],
  label: string,
  detail?: string,
): TimelineEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    at: Date.now(),
    kind,
    label,
    detail,
  };
}

function withTimeline(
  state: EngineState,
  next: TimelineEntry,
): TimelineEntry[] {
  return [...state.timeline, next].slice(-MAX_TIMELINE);
}

function str(event: EngineEvent, key: string, fallback = ""): string {
  const value = event[key];
  return typeof value === "string" ? value : fallback;
}

function num(event: EngineEvent, key: string, fallback = 0): number {
  const value = event[key];
  return typeof value === "number" ? value : fallback;
}

function updateWorker(
  state: EngineState,
  name: string,
  patch: Partial<Worker>,
  displayName?: string,
): EngineState {
  const existing = state.workers[name] ?? makeWorker(name, displayName);
  const merged: Worker = { ...existing, ...patch };
  if (patch.phase) {
    merged.status = patch.status ?? PHASE_STATUS[patch.phase];
    merged.progress = patch.progress ?? PHASE_PROGRESS[patch.phase];
  }
  return {
    ...state,
    workers: { ...state.workers, [name]: merged },
    order: state.order.includes(name)
      ? state.order
      : [...state.order, name],
  };
}

type Action =
  | { kind: "connected"; value: boolean }
  | { kind: "event"; event: EngineEvent }
  | { kind: "userMessage"; content: string }
  | { kind: "clearError" };

function reduce(state: EngineState, action: Action): EngineState {
  if (action.kind === "connected") {
    return { ...state, connected: action.value };
  }

  if (action.kind === "clearError") {
    return { ...state, error: null };
  }

  if (action.kind === "userMessage") {
    const message: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: action.content,
      at: Date.now(),
    };
    return {
      ...state,
      messages: [...state.messages, message],
      question: action.content,
      researching: true,
      progress: 0,
      round: 0,
      consensus: null,
      error: null,
      stage: "Planning",
      stageDetail: "Decomposing the request",
      plan: null,
      evidenceCount: 0,
      timeline: withTimeline(state, entry("research", "Request submitted", action.content)),
    };
  }

  const event = action.event;
  const provider = event.provider ? String(event.provider) : undefined;

  switch (event.type) {
    case "Snapshot": {
      const providers = (event.providers ?? []) as Array<{
        name: string;
        displayName: string;
        authState: AuthState;
        detail?: string;
      }>;
      const workers: Record<string, Worker> = {};
      const order: string[] = [];
      for (const item of providers) {
        workers[item.name] = {
          ...makeWorker(item.name, item.displayName),
          authState: item.authState,
          detail: item.detail,
          phase: item.authState === "READY" ? "Idle" : "Blocked",
          status:
            item.authState === "READY"
              ? PHASE_STATUS.Idle
              : item.authState.replace(/_/g, " "),
        };
        order.push(item.name);
      }
      return {
        ...state,
        workers,
        order,
        disabled: (event.disabled ?? []) as string[],
        browserHidden: Boolean(event.browserHidden ?? state.browserHidden),
      };
    }

    case "EngineReady":
      return {
        ...state,
        engineReady: true,
        disabled: (event.disabled ?? []) as string[],
        timeline: withTimeline(state, entry("research", "Engine ready")),
      };

    case "ResearchStarted":
      return {
        ...state,
        researching: true,
        progress: 0.02,
        round: 1,
        timeline: withTimeline(
          state,
          entry("research", "Research started", str(event, "question")),
        ),
      };

    case "ResearchRoundStarted":
      return {
        ...state,
        round: num(event, "round", state.round),
        timeline: withTimeline(
          state,
          entry(
            "research",
            `Round ${num(event, "round")} started`,
            ((event.providers ?? []) as string[]).join(", "),
          ),
        ),
      };

    case "ResearchProgress": {
      const stage = str(event, "stage") || state.stage;
      const detail = str(event, "detail") || state.stageDetail;
      const next = {
        ...state,
        progress: num(event, "progress", state.progress),
        round: num(event, "round", state.round),
        stage,
        stageDetail: event.detail ? detail : state.stageDetail,
      };
      // Only stage entries are worth a timeline row; the completion
      // ticks that follow each stage would double every entry.
      if (event.detail && stage !== state.stage) {
        return { ...next, timeline: withTimeline(next, entry("research", stage, detail)) };
      }
      return next;
    }

    case "ResearchPlanned": {
      const plan = event.plan as ResearchPlanView | undefined;
      if (!plan) return state;
      return {
        ...state,
        plan: { ...plan, tasks: plan.tasks.map((task) => ({ ...task, status: "pending" as const })) },
        stage: "Planning",
        timeline: withTimeline(
          state,
          entry(
            "research",
            `Plan: ${plan.tasks.length} subtasks`,
            plan.objective,
          ),
        ),
      };
    }

    case "TaskAssigned": {
      const task = event.task as PlanTask | undefined;
      if (!task || !state.plan) return state;
      const tasks = [...state.plan.tasks];
      const index = tasks.findIndex((item) => item.taskId === task.taskId);
      const merged: PlanTask = { ...task, status: "pending" };
      if (index >= 0) tasks[index] = merged;
      else tasks.push(merged);
      return { ...state, plan: { ...state.plan, tasks } };
    }

    case "TaskFinished": {
      if (!state.plan) return state;
      const taskId = num(event, "taskId", -1);
      const success = Boolean(event.success);
      const tasks = state.plan.tasks.map((task) =>
        task.taskId === taskId
          ? { ...task, status: (success ? "done" : "failed") as PlanTask["status"] }
          : task,
      );
      return {
        ...state,
        plan: { ...state.plan, tasks },
        timeline: withTimeline(
          state,
          entry(
            success ? "provider" : "error",
            `${str(event, "taskTitle") || "Subtask"} ${success ? "complete" : "failed"}`,
            str(event, "provider"),
          ),
        ),
      };
    }

    case "EvidenceExtracted":
      return {
        ...state,
        evidenceCount: num(event, "itemCount", state.evidenceCount),
        timeline: withTimeline(
          state,
          entry(
            "research",
            `${num(event, "itemCount")} evidence items extracted`,
            ((event.providers ?? []) as string[]).join(", "),
          ),
        ),
      };

    case "ProviderStarted":
      return updateWorker(
        state,
        provider ?? "unknown",
        {
          phase: "Thinking",
          startedAt: Date.now(),
          stage: str(event, "stage") || "Opening conversation…",
          stageAt: Date.now(),
          elapsedSeconds: 0,
          promptChars: num(event, "promptChars"),
          turn: num(event, "turn"),
          error: undefined,
        },
        str(event, "displayName") || undefined,
      );

    case "ProviderTyping":
      return updateWorker(state, provider ?? "unknown", {
        phase: "Thinking",
        stage: str(event, "stage") || "Typing prompt…",
        stageAt: Date.now(),
      });

    case "ProviderWaiting":
      return updateWorker(state, provider ?? "unknown", {
        phase: "Waiting",
        stage: str(event, "stage") || "Waiting for response…",
        stageAt: Date.now(),
      });

    case "ProviderStreaming":
      return updateWorker(state, provider ?? "unknown", {
        phase: "Generating",
        stage: str(event, "stage") || "Streaming answer…",
        stageAt: Date.now(),
      });

    case "ProviderFinished": {
      const next = updateWorker(
        state,
        provider ?? "unknown",
        {
          phase: "Finished",
          stage: "Finished",
          stageAt: Date.now(),
          answerChars: num(event, "answerChars"),
          elapsedSeconds: num(event, "elapsedSeconds"),
        },
        str(event, "displayName") || undefined,
      );
      const label = next.workers[provider ?? ""]?.displayName ?? provider;
      return {
        ...next,
        timeline: withTimeline(
          next,
          entry(
            "provider",
            `${label} finished`,
            `${num(event, "elapsedSeconds")}s`,
          ),
        ),
      };
    }

    case "ProviderError": {
      const next = updateWorker(state, provider ?? "unknown", {
        phase: "Failed",
        error: str(event, "error", "Unknown error"),
        elapsedSeconds: num(event, "elapsedSeconds"),
      });
      return {
        ...next,
        timeline: withTimeline(
          next,
          entry("error", `${provider} failed`, str(event, "error")),
        ),
      };
    }

    case "ProviderLoginRequired":
    case "ProviderCaptchaRequired": {
      const needsCaptcha = event.type === "ProviderCaptchaRequired";
      const next = updateWorker(state, provider ?? "unknown", {
        phase: "Blocked",
        authState: needsCaptcha ? "CAPTCHA_REQUIRED" : "LOGIN_REQUIRED",
        status: needsCaptcha ? "Verification needed" : "Sign-in needed",
        detail: str(event, "reason"),
      });
      return {
        ...next,
        timeline: withTimeline(
          next,
          entry(
            "error",
            `${provider} needs attention`,
            needsCaptcha ? "Human verification" : "Sign-in required",
          ),
        ),
      };
    }

    case "ConsensusUpdated":
      return {
        ...state,
        consensus: {
          round: num(event, "round", state.round),
          agreement: num(event, "agreement"),
          confidence: num(event, "confidence"),
          opinionCount: num(event, "opinionCount"),
          contradictions: (event.contradictions ?? []) as never[],
          products: (event.products ?? []) as never[],
          supporting: state.consensus?.supporting ?? [],
          opposing: state.consensus?.opposing ?? [],
        },
        timeline: withTimeline(
          state,
          entry(
            "consensus",
            "Consensus updated",
            `confidence ${num(event, "confidence").toFixed(2)}`,
          ),
        ),
      };

    case "ContradictionDetected":
      return {
        ...state,
        timeline: withTimeline(
          state,
          entry(
            "consensus",
            `${str(event, "sourceA")} vs ${str(event, "sourceB")}`,
            str(event, "description"),
          ),
        ),
      };

    case "ResearchFinished":
      return {
        ...state,
        researching: false,
        progress: 1,
        stage: "Done",
        stageDetail: str(event, "headline") || state.stageDetail,
        consensus: state.consensus
          ? {
              ...state.consensus,
              converged: Boolean(event.converged),
              stopReason: str(event, "stopReason"),
              supporting: (event.supporting ?? []) as string[],
              opposing: (event.opposing ?? []) as string[],
            }
          : null,
        timeline: withTimeline(
          state,
          entry(
            "research",
            event.converged ? "Consensus reached" : "Finished without consensus",
            str(event, "stopReason"),
          ),
        ),
      };

    case "ResearchFailed":
      return {
        ...state,
        researching: false,
        stage: "Failed",
        error: str(event, "error", "Research failed"),
        timeline: withTimeline(
          state,
          entry("error", "Research failed", str(event, "error")),
        ),
      };

    case "AssistantMessage": {
      const message: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: str(event, "content"),
        at: Date.now(),
        final: (event.final as FinalAnswer | undefined) ?? undefined,
        research: (event.research as ResearchResultView | undefined) ?? undefined,
      };
      return {
        ...state,
        researching: false,
        progress: 1,
        stage: "Done",
        messages: [...state.messages, message],
      };
    }

    case "Metrics": {
      const snapshot: MetricsSnapshot = {
        at: Date.now(),
        system: event.system as MetricsSnapshot["system"],
        engine: event.engine as MetricsSnapshot["engine"],
        research: event.research as MetricsSnapshot["research"],
      };
      return {
        ...state,
        metrics: snapshot,
        browserHidden: snapshot.engine?.browserHidden ?? state.browserHidden,
      };
    }

    case "BrowserVisibility":
      return { ...state, browserHidden: Boolean(event.hidden) };

    case "Diagnostics":
      return {
        ...state,
        diagnostics: (event.providers ?? []) as ProviderDiagnostics[],
        diagnosticsAt: Date.now(),
      };

    case "Screenshot": {
      const name = str(event, "provider");
      if (!name) return state;
      return {
        ...state,
        screenshots: {
          ...state.screenshots,
          [name]: { image: str(event, "image"), at: Date.now() },
        },
      };
    }

    case "Log": {
      const line = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        level: str(event, "level", "INFO"),
        source: str(event, "source"),
        message: str(event, "message"),
        at: Date.now(),
      };
      return { ...state, logs: [...state.logs, line].slice(-MAX_LOGS) };
    }

    case "Error":
      return { ...state, error: str(event, "message", "Engine error") };

    default:
      return state;
  }
}

/**
 * Connect to the engine and expose reduced state plus a command sender.
 *
 * @param url - WebSocket endpoint. Defaults to the local engine.
 * @returns Engine state, a command sender, and a chat submit helper.
 */
export function useEngine(url: string = DEFAULT_URL) {
  const [state, dispatch] = useReducer(reduce, initialState);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;

    const connect = () => {
      if (closedRef.current) return;

      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => dispatch({ kind: "connected", value: true });

      socket.onmessage = (raw) => {
        try {
          dispatch({ kind: "event", event: JSON.parse(raw.data) });
        } catch {
          // A malformed frame must not tear down the connection.
        }
      };

      socket.onclose = () => {
        dispatch({ kind: "connected", value: false });
        if (!closedRef.current) {
          timerRef.current = window.setTimeout(connect, RECONNECT_MS);
        }
      };

      socket.onerror = () => socket.close();
    };

    connect();

    return () => {
      closedRef.current = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
      socketRef.current?.close();
    };
  }, [url]);

  const send = useCallback((command: Command) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(command));
    }
  }, []);

  const submit = useCallback(
    (query: string) => {
      const trimmed = query.trim();
      if (!trimmed) return;
      dispatch({ kind: "userMessage", content: trimmed });
      send({ command: "research", query: trimmed });
    },
    [send],
  );

  const workers = useMemo(
    () => state.order.map((name) => state.workers[name]).filter(Boolean),
    [state.order, state.workers],
  );

  return { state, workers, send, submit };
}
