/**
 * Event and state types mirroring the Python engine's contract.
 *
 * These mirror `src/events/models.py`. The UI never reaches into engine
 * internals and never sees a browser — it renders what these events
 * describe, which is what keeps Chrome an implementation detail.
 *
 * Keep this file in sync with the Python `EventType` enum. A mismatch
 * shows up as events falling through to the default branch rather than as
 * a crash, so the reducer logs unknown types in developer mode.
 */

export type EventType =
  | "EngineReady"
  | "EngineShutdown"
  | "Log"
  | "ProviderRegistered"
  | "ProviderStarted"
  | "ProviderTyping"
  | "ProviderWaiting"
  | "ProviderStreaming"
  | "ProviderFinished"
  | "ProviderError"
  | "ProviderLoginRequired"
  | "ProviderCaptchaRequired"
  | "ProviderStateChanged"
  | "ResearchStarted"
  | "ResearchProgress"
  | "ResearchRoundStarted"
  | "ResearchFinished"
  | "ResearchFailed"
  | "ConsensusStarted"
  | "ConsensusUpdated"
  | "ContradictionDetected"
  | "AssistantMessage"
  | "AssistantToken"
  | "Snapshot"
  | "Error"
  | "ResearchAccepted"
  | "SessionOpened"
  | "AuthChecked"
  | "Recovered"
  | "ConversationsReset";

/** Coarse phase shown on a worker card. */
export type WorkerPhase =
  | "Idle"
  | "Waiting"
  | "Thinking"
  | "Generating"
  | "Finished"
  | "Failed"
  | "Blocked";

/** Authentication state reported by the engine. */
export type AuthState =
  | "READY"
  | "LOGIN_REQUIRED"
  | "CAPTCHA_REQUIRED"
  | "OFFLINE"
  | "RECOVERING"
  | "UNKNOWN";

/** One event received over the socket. */
export interface EngineEvent {
  type: EventType;
  eventId?: string;
  timestamp?: string;
  provider?: string;
  researchId?: string;
  [key: string]: unknown;
}

/** Live view of one provider, accumulated from its events. */
export interface Worker {
  name: string;
  displayName: string;
  phase: WorkerPhase;
  authState: AuthState;
  status: string;
  detail?: string;
  startedAt?: number;
  elapsedSeconds: number;
  answerChars: number;
  promptChars: number;
  turn: number;
  error?: string;
  currentUrl?: string;
  progress: number;
}

/** One entry in the live research timeline. */
export interface TimelineEntry {
  id: string;
  at: number;
  label: string;
  detail?: string;
  kind: "research" | "provider" | "consensus" | "error";
}

/** A message in the Architect chat. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  at: number;
}

/** A merged recommendation reported by the consensus engine. */
export interface ConsensusProduct {
  name: string;
  supporters: string[];
  dissenters: string[];
  confidence: number;
}

/** A disagreement between two providers. */
export interface Contradiction {
  sourceA: string;
  sourceB: string;
  description: string;
}

/** Current consensus state for the active research run. */
export interface ConsensusState {
  round: number;
  agreement: number;
  confidence: number;
  opinionCount: number;
  contradictions: Contradiction[];
  products: ConsensusProduct[];
  converged?: boolean;
  stopReason?: string;
  supporting: string[];
  opposing: string[];
}

/** A forwarded copy of a terminal log line. */
export interface LogLine {
  id: string;
  level: string;
  source: string;
  message: string;
  at: number;
}

/** Aggregate application state derived from the event stream. */
export interface EngineState {
  connected: boolean;
  engineReady: boolean;
  workers: Record<string, Worker>;
  order: string[];
  disabled: string[];
  timeline: TimelineEntry[];
  messages: ChatMessage[];
  consensus: ConsensusState | null;
  logs: LogLine[];
  researching: boolean;
  progress: number;
  round: number;
  question: string;
  error: string | null;
}

/** Commands the UI may send to the engine. */
export type Command =
  | { command: "research"; query: string }
  | { command: "openSession" }
  | { command: "snapshot" }
  | { command: "recover" }
  | { command: "checkAuth" }
  | { command: "resetConversations" };
