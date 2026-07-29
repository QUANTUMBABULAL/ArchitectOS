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
  | "ConversationsReset"
  | "ResearchPlanned"
  | "TaskAssigned"
  | "TaskFinished"
  | "EvidenceExtracted"
  | "Metrics"
  | "BrowserVisibility"
  | "Diagnostics"
  | "Screenshot"
  | "ProviderReloaded";

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
  /** Engine-reported live stage, e.g. "Typing prompt…". */
  stage?: string;
  /** When the current stage began (client clock). */
  stageAt?: number;
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

/** One subtask of the research plan. */
export interface PlanTask {
  taskId: number;
  title: string;
  question: string;
  kind: string;
  assignedTo: string | null;
  status?: "pending" | "running" | "done" | "failed";
}

/** The decomposition of a research request. */
export interface ResearchPlanView {
  question: string;
  objective: string;
  generatedBy: string;
  tasks: PlanTask[];
}

/** One structured claim extracted from a provider answer. */
export interface EvidenceItemView {
  provider: string;
  taskId: number;
  taskTitle: string;
  fact: string;
  source: string;
  confidence: number;
  links: string[];
  caveats: string[];
}

/** A runner-up choice under a different priority. */
export interface AlternativeView {
  priority: string;
  choice: string;
  rationale: string;
}

/** The executive report produced by the research operator. */
export interface ExecutiveReportView {
  question: string;
  headline: string;
  headlineLabel: string;
  summary: string;
  confidence: number;
  evidencePoints: string[];
  weaknesses: string[];
  alternatives: AlternativeView[];
  sources: Array<{ title: string; url: string }>;
  supportingProviders: string[];
  dissentingProviders: string[];
  contradictions: string[];
  wordCount: number;
  synthesized: boolean;
}

/** Everything one research run produced. */
export interface ResearchResultView {
  researchId: string;
  plan: ResearchPlanView;
  evidence: {
    items: EvidenceItemView[];
    rawAnswers: Record<string, string>;
    providers: string[];
  };
  report: ExecutiveReportView;
  agreement: number;
  elapsedSeconds: number;
}

/** One concise final answer with its evidence. */
export interface FinalAnswer {
  summary: string;
  confidence: number;
  supporting: string[];
  opposing: string[];
  disagreements: string[];
  sources: Array<{ title: string; url: string }>;
  rawAnswers: Record<string, string>;
  rounds: number;
  converged: boolean;
  stopReason: string;
  synthesized: boolean;
}

/** Host + engine telemetry published by the Metrics event. */
export interface MetricsSnapshot {
  at: number;
  system: {
    cpuPercent: number | null;
    ramPercent: number | null;
    ramAvailableBytes: number | null;
    ramTotalBytes: number | null;
    netSentBytesPerSec: number | null;
    netRecvBytesPerSec: number | null;
    gpu: {
      utilizationPercent: number;
      memoryUsedMb: number;
      memoryTotalMb: number;
    } | null;
  };
  engine: {
    ollamaHealthy: boolean | null;
    ollamaModel: string;
    browser: { state: string; healthy?: boolean; pages?: number };
    browserHidden: boolean;
    workerCount: number;
    registeredWorkers: number;
    sessionOpen: boolean;
    paused: Record<string, string>;
    responseTimes: Record<
      string,
      { last: number; avg: number; count: number }
    >;
    stats: {
      promptsDispatched: number;
      responsesReceived: number;
      providerFailures: number;
      recoveries: number;
    };
  };
  research: {
    running: boolean;
    elapsedSeconds: number | null;
    estimatedRemainingSeconds: number | null;
    averageDurationSeconds: number | null;
  };
}

/** One provider record on the Browser Manager page. */
export interface ProviderDiagnostics {
  name: string;
  displayName: string;
  registered: boolean;
  loggedIn: boolean | null;
  url: string | null;
  conversationId: string | null;
  turns: number;
  sessionAgeSeconds: number | null;
  cookieCount: number | null;
  tabOpen: boolean;
  jsHeapBytes: number | null;
  lastActivity: string | null;
  paused: boolean;
  pauseReason: string | null;
  responseTimes: { last: number; avg: number; count: number } | null;
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
  /** Structured final answer, present on debate-mode results. */
  final?: FinalAnswer;
  /** Full research result, present on operator-mode results. */
  research?: ResearchResultView;
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
  /** Current pipeline stage, e.g. "Gathering". */
  stage: string;
  /** One-line description of what the stage is doing. */
  stageDetail: string;
  /** The plan being executed, with live per-task status. */
  plan: ResearchPlanView | null;
  /** Evidence items extracted so far this run. */
  evidenceCount: number;
  metrics: MetricsSnapshot | null;
  browserHidden: boolean;
  diagnostics: ProviderDiagnostics[];
  diagnosticsAt: number | null;
  screenshots: Record<string, { image: string; at: number }>;
}

/** Commands the UI may send to the engine. */
export type Command =
  | { command: "research"; query: string; newChat?: boolean }
  | { command: "openSession" }
  | { command: "snapshot" }
  | { command: "recover" }
  | { command: "checkAuth" }
  | { command: "resetConversations" }
  | { command: "newChat" }
  | { command: "showBrowser" }
  | { command: "hideBrowser" }
  | { command: "diagnostics" }
  | { command: "screenshot"; provider: string }
  | { command: "reloadProvider"; provider: string }
  | { command: "focusProvider"; provider: string };
